"""The outbound proxy of confined generated tools, on its UNIX socket; offline except the TLS check."""
import errno
import http.server
import ipaddress
import os
import socket
import ssl
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from openvlharness.toolsmith import proxy

REAL_CONNECT = socket.socket.connect  # before the suite's offline guard replaces it
pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="the proxy listens on a UNIX socket")


def allow_all(addresses):
    """Stands in for destination_refused: the tests' servers listen on loopback."""
    return None


class Echo(http.server.BaseHTTPRequestHandler):
    """Answer with the request line, headers and body the server received."""

    def do_GET(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = (f"{self.command} {self.path}\n" + "".join(f"{name}: {value}\n" for name, value in self.headers.items())
                + self.rfile.read(length).decode())
        self.send_response(200)
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    do_POST = do_GET

    def log_message(self, *args):
        pass


def _rest(client):
    """What a client still reads before the connection ends; a UNIX socket closed with unread data resets."""
    data = b""
    try:
        while chunk := client.recv(65536):
            data += chunk
    except ConnectionResetError:
        pass
    return data


def _length(response):
    head = response.split(b"\r\n\r\n", 1)[0].decode("latin-1").lower()
    return next((int(line.split(":", 1)[1]) for line in head.split("\r\n") if line.startswith("content-length:")), 0)


def serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Recorder:
    """A loopback HTTP server that keeps every connection open, whatever Connection says.

    It records each request head and body it reads, and answers each with its label.
    """

    def __init__(self, label):
        self.label, self.requests = label, []
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self):
        self.listener.close()

    def _accept(self):
        while True:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection):
        data = b""
        with connection:
            while True:
                while b"\r\n\r\n" not in data:
                    chunk = connection.recv(65536)
                    if not chunk:
                        return
                    data += chunk
                head, data = data.split(b"\r\n\r\n", 1)
                lines = head.decode("latin-1").split("\r\n")
                length = next((int(line.split(":", 1)[1]) for line in lines
                               if line.lower().startswith("content-length:")), 0)
                while len(data) < length:
                    data += connection.recv(65536)
                self.requests.append((lines, data[:length]))
                data = data[length:]
                body = f"{self.label} served {lines[0]}".encode()
                connection.sendall(b"HTTP/1.1 200 OK\r\nKeep-Alive: timeout=60\r\nConnection: keep-alive\r\n"
                                   b"Content-Length: %d\r\n\r\n%s" % (len(body), body))


class OutboundProxyTests(unittest.TestCase):
    def setUp(self):
        environment = {name: value for name, value in os.environ.items() if not name.lower().endswith("_proxy")}
        patcher = patch.dict(os.environ, environment, clear=True)  # connect directly, as the router does here
        patcher.start()
        self.addCleanup(patcher.stop)
        self.proxy = self.start()

    def start(self, *args, **options):
        outbound = proxy.OutboundProxy(*args, **options)
        self.addCleanup(outbound.close)
        return outbound

    def connect(self, outbound=None):
        client = socket.socket(socket.AF_UNIX)
        client.settimeout(30)
        client.connect((outbound or self.proxy).path)
        return client

    def exchange(self, request, outbound=None):
        """Send a request and return everything the proxy sends back until it closes."""
        with self.connect(outbound) as client:
            client.sendall(request)
            data = b""
            while chunk := client.recv(65536):
                data += chunk
        return data

    def ask(self, request, outbound=None):
        """Send a request and return its response, read up to its Content-Length as a client does."""
        with self.connect(outbound) as client:
            client.sendall(request)
            data = b""
            while b"\r\n\r\n" not in data or len(data.split(b"\r\n\r\n", 1)[1]) < _length(data):
                chunk = client.recv(65536)
                if not chunk:
                    break
                data += chunk
        return data

    def test_addresses_of_this_host_and_private_networks_are_refused_without_naming_them(self):
        for target in ("127.0.0.1:80", "[::1]:80", "0.0.0.0:80", "10.1.2.3:80", "169.254.169.254:80",
                       "[::ffff:127.0.0.1]:80", "2130706433:80", "0177.0.0.1:80", "localhost:80", "100.64.0.1:80",
                       "224.0.0.1:80", "192.168.1.1:443", "[fe80::1]:80", "[fc00::1]:80", "[fec0::1]:80",
                       "[64:ff9b::7f00:1]:80", "[2002:7f00:1::1]:80", "[::127.0.0.1]:443"):
            with self.subTest(target=target):
                answer = self.exchange(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
                self.assertTrue(answer.startswith(b"HTTP/1.1 403 Forbidden"), answer[:120])
                self.assertIn(proxy.REFUSED.encode(), answer)
        answer = self.exchange(b"GET http://localhost/ HTTP/1.1\r\nHost: localhost\r\n\r\n")
        self.assertTrue(answer.startswith(b"HTTP/1.1 403"), answer[:120])
        # The name a tool gave comes back, never the addresses it resolves to.
        self.assertTrue(answer.endswith(b"\r\n\r\nlocalhost " + proxy.REFUSED.encode() + b"\n"), answer)
        own = sorted(str(address) for address in proxy.local_addresses() if address.is_global)
        for address in own[:2]:  # a public address of this host is refused like loopback
            target = f"[{address}]:80" if ":" in address else f"{address}:80"
            with self.subTest(own=address):
                answer = self.exchange(f"CONNECT {target} HTTP/1.1\r\n\r\n".encode())
                self.assertTrue(answer.startswith(b"HTTP/1.1 403"), answer[:120])
        for address in ("::7f00:1", "::ffff:0:7f00:1", "2001:0:4136:e378:8000:63bf:3fff:fdd2", "64:ff9b:1::a00:5"):
            self.assertIsNotNone(proxy.destination_refused([address]), address)  # IPv4 forms in IPv6 ones
        self.assertEqual(proxy.destination_refused(["168.63.129.16"]), proxy.REFUSED)  # a cloud platform's own
        public = {"8.8.8.8", "2001:4860:4860::8888", "::ffff:8.8.8.8"}
        if not public & {str(address) for address in proxy.local_addresses()}:
            self.assertIsNone(proxy.destination_refused(sorted(public)))

    def test_networks_this_host_reaches_without_a_gateway_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            route, ipv6_route = Path(directory, "route"), Path(directory, "ipv6_route")
            route.write_text(
                "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
                "eno1\t00000000\t01040808\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"  # default via 8.8.4.1
                "eno1\t00040808\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n"  # 8.8.4.0/24
                "wg0\t00000000\t00000000\t0001\t0\t0\t0\t00000080\t0\t0\t0\n"  # 0.0.0.0/1 through a VPN
                "eno1\t00000909\t00000000\t0201\t0\t0\t0\t0000FFFF\t0\t0\t0\n")  # 9.9.0.0/16, unreachable
            ipv6_route.write_text(
                "20014860486000000000000000000000 40 00000000000000000000000000000000 00 "
                "00000000000000000000000000000000 00000100 00000001 00000000 00000001 eno1\n"
                "00000000000000000000000000000000 00 00000000000000000000000000000000 00 "
                "fe800000000000000000000000000001 00000400 00000001 00000000 00000003 eno1\n")
            with patch.multiple(proxy, ROUTE=str(route), IPV6_ROUTE=str(ipv6_route)), \
                    patch.dict(proxy._host, {"value": None, "until": 0.0}):
                self.assertEqual(proxy.link_networks(), [ipaddress.ip_network("8.8.4.0/24"),
                                                         ipaddress.ip_network("2001:4860:4860::/64")])
                for address in ("8.8.4.1", "8.8.4.50", "2001:4860:4860::5", "::ffff:8.8.4.50"):
                    self.assertEqual(proxy.destination_refused([address]), proxy.REFUSED, address)
                for address in ("8.8.5.1", "2001:4860:4861::5", "9.9.1.1"):
                    if ipaddress.ip_address(address) not in proxy.local_addresses():
                        self.assertIsNone(proxy.destination_refused([address]), address)
                answer = self.exchange(b"CONNECT 8.8.4.50:443 HTTP/1.1\r\n\r\n")
        self.assertTrue(answer.startswith(b"HTTP/1.1 403"), answer[:120])
        self.assertEqual(answer.split(b"\r\n\r\n", 1)[1], b"8.8.4.50 " + proxy.REFUSED.encode() + b"\n")

    def test_a_connection_field_cannot_strip_the_body_length(self):
        # Without its Content-Length the body would be the next request on the server's connection.
        for value in (b"content-length", b"close, Content-Length", b"\tCONTENT-LENGTH\t,x"):
            with self.subTest(value=value):
                answer = self.exchange(b"POST http://8.8.8.8/ HTTP/1.1\r\nConnection: " + value
                                       + b"\r\nContent-Length: 5\r\n\r\nhello")
                self.assertTrue(answer.startswith(b"HTTP/1.1 400"), answer[:80])
                self.assertIn(b"Connection lists Content-Length", answer)

    def test_this_hosts_networks_are_read_at_most_every_few_seconds_and_unknown_ones_refuse_all(self):
        calls = []
        real = proxy.local_addresses

        def counted():
            calls.append(1)
            return real()
        with patch.object(proxy, "local_addresses", counted), patch.dict(proxy._host, {"value": None, "until": 0.0}):
            for _ in range(3):
                proxy.destination_refused(["8.8.8.8"])
            self.assertEqual(len(calls), 1)
        with patch.object(proxy, "FIB_TRIE", "/nonexistent/fib_trie"), \
                patch.dict(proxy._host, {"value": None, "until": 0.0}):
            answer = self.exchange(b"CONNECT 8.8.8.8:443 HTTP/1.1\r\n\r\n")
        self.assertTrue(answer.startswith(b"HTTP/1.1 503"), answer[:120])
        self.assertIn(b"the proxy cannot check destinations: cannot read this host's IPv4 addresses", answer)

    def test_only_allowed_ports_and_networks_outside_deny_networks_are_reached(self):
        outbound = self.start(ports=[443], deny=["8.8.8.0/24", "2001:4860::/32"])
        for request, reason in (
                (b"CONNECT 8.8.8.8:443 HTTP/1.1\r\n\r\n", b"8.8.8.8 " + proxy.REFUSED.encode()),
                (b"CONNECT [::ffff:8.8.8.8]:443 HTTP/1.1\r\n\r\n", proxy.REFUSED.encode()),
                (b"CONNECT [2001:4860:4860::8888]:443 HTTP/1.1\r\n\r\n", proxy.REFUSED.encode()),
                (b"CONNECT 8.8.4.4:22 HTTP/1.1\r\n\r\n", b" 22 is not allowed (toolsmith.worker.network_ports: 443)\n"),
                (b"GET http://8.8.4.4/ HTTP/1.1\r\n\r\n", b"port 80 is not allowed"),
                (b"GET http://8.8.4.4:8080/ HTTP/1.1\r\n\r\n", b"port 8080 is not allowed")):
            with self.subTest(request=request):
                answer = self.exchange(request, outbound)
                self.assertTrue(answer.startswith(b"HTTP/1.1 403"), answer[:120])
                self.assertIn(reason, answer)

    def test_an_allowed_destination_is_relayed_and_absolute_requests_go_on_in_origin_form(self):
        server = serve(Echo)
        self.addCleanup(server.shutdown)
        port = server.server_port
        with patch.object(proxy, "destination_refused", allow_all):
            outbound = self.start(ports=[port])
            tunnel = self.exchange(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\nGET /tunnelled HTTP/1.1\r\n"
                                   "Host: example\r\nConnection: close\r\n\r\n".encode(), outbound)
            direct = self.exchange(f"POST http://127.0.0.1:{port}/direct?q=1#part HTTP/1.1\r\nHost: elsewhere\r\n"
                                   "Proxy-Connection: keep-alive\r\nConnection: keep-alive, X-Hop\r\nX-Hop: 1\r\n"
                                   "Proxy-Authorization: Basic c2VjcmV0\r\nContent-Length: 5\r\n\r\nhello".encode(),
                                   outbound)
        self.assertTrue(tunnel.startswith(b"HTTP/1.1 200 Connection established\r\n\r\nHTTP/1.0 200 OK"), tunnel[:80])
        self.assertIn(b"GET /tunnelled", tunnel)
        self.assertTrue(direct.startswith(b"HTTP/1.0 200 OK"), direct[:80])
        # Host comes from the target, as RFC 9112 asks of a proxy; hop-by-hop fields stay here.
        self.assertTrue(direct.endswith(f"POST /direct?q=1\nHost: 127.0.0.1:{port}\nContent-Length: 5\n"
                                        "Connection: close\nhello".encode()), direct)
        self.assertEqual(direct.split(b"\r\n\r\n", 1)[0].count(b"Connection:"), 1)
        for field in (b"Proxy-Connection", b"X-Hop", b"Proxy-Authorization", b"elsewhere"):
            self.assertNotIn(field, direct.split(b"\r\n\r\n", 1)[1])

    def test_loose_or_malformed_requests_are_refused_with_400_and_no_traceback(self):
        errors = []
        with patch.object(threading, "excepthook", lambda args: errors.append(args.exc_value)):
            for request, status in (
                    # Bare LF, bare CR, NUL, folded and spaced header lines: parsers disagree on them.
                    (b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\nX-A: 1\n\n"
                     b"GET http://127.0.0.1/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET http://example.com/ HTTP/1.1\nHost: example.com\r\n\r\n", 400),
                    (b"GET http://example.com/ HTTP/1.1\r\nX-A: 1\rX-B: 2\r\n\r\n", 400),
                    (b"GET http://example.com/ HTTP/1.1\r\nX-A: 1\x00\r\n\r\n", 400),
                    (b"GET http://example.com/ HTTP/1.1\r\nX-A: 1\r\n continued\r\n\r\n", 400),
                    (b"GET http://example.com/ HTTP/1.1\r\nHost : example.com\r\n\r\n", 400),
                    (b"GET http://example.com/ HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n", 400),
                    (b"GET\thttp://example.com/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET http://example.com/  HTTP/1.1\r\n\r\n", 400),
                    (b"GET http://example.com/ HTTP/2.0\r\n\r\n", 400),
                    (b"\r\nGET http://example.com/ HTTP/1.1\r\n\r\n", 400),
                    (b"POST http://example.com/ HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", 411),
                    (b"POST http://example.com/ HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 5\r\n\r\nhello", 400),
                    (b"POST http://example.com/ HTTP/1.1\r\nContent-Length: +5\r\n\r\nhello", 400),
                    # Targets that are not an absolute http:// URL or a host and port.
                    (b"GET http://[::1/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET http://[example.com]/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET http://a\x00b.example/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET http://127.0.0.1:0/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET http://127.0.0.1:65536/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET http://user:pass@example.com/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET https://example.com/ HTTP/1.1\r\n\r\n", 400),
                    (b"GET /relative HTTP/1.1\r\nHost: example.com\r\n\r\n", 400),
                    (b"CONNECT example.com HTTP/1.1\r\n\r\n", 400),
                    (b"CONNECT a\x00b.example:443 HTTP/1.1\r\n\r\n", 400),
                    (b"CONNECT [fe80::1%25eth0]:443 HTTP/1.1\r\n\r\n", 400)):
                with self.subTest(request=request):
                    answer = self.exchange(request)
                    self.assertTrue(answer.startswith(f"HTTP/1.1 {status} ".encode()), answer[:120])
            time.sleep(0.2)
        self.assertEqual(errors, [])

    def test_one_request_per_connection_however_the_client_or_server_behave(self):
        first, second = Recorder("first"), Recorder("second")
        for server in (first, second):
            self.addCleanup(server.close)
        with patch.object(proxy, "destination_refused", allow_all):
            outbound = self.start(ports=[first.port, second.port])
            # A pipelined second request, after a body, never goes on.
            answer = self.ask(f"POST http://127.0.0.1:{first.port}/upload HTTP/1.1\r\nContent-Length: 5\r\n\r\n"
                              f"helloGET http://127.0.0.1:{second.port}/smuggled HTTP/1.1\r\n\r\n".encode(), outbound)
            self.assertEqual([(lines[0], body) for lines, body in first.requests],
                             [("POST /upload HTTP/1.1", b"hello")])
            # The server would keep the connection, but the client hears Connection: close and goes elsewhere.
            self.assertEqual(answer, b"HTTP/1.1 200 OK\r\nContent-Length: 34\r\nConnection: close\r\n\r\n"
                                     b"first served POST /upload HTTP/1.1")
            relay = f"http://127.0.0.1:{proxy.forward(outbound.path)}"
            import requests

            with requests.Session() as session:
                session.trust_env = False
                proxies = {"http": relay}
                one = session.get(f"http://127.0.0.1:{first.port}/one", proxies=proxies, timeout=10)
                two = session.get(f"http://127.0.0.1:{second.port}/two", proxies=proxies,
                                  headers={"X-Api-Key": "for-the-second-only"}, timeout=10)
            # A client that sends another request on the connection anyway loses the connection.
            with self.connect(outbound) as client:
                client.sendall(f"GET http://127.0.0.1:{first.port}/three HTTP/1.1\r\n\r\n".encode())
                head = b""
                while not head.endswith(b"GET /three HTTP/1.1"):
                    head += client.recv(65536)
                client.sendall(f"GET http://127.0.0.1:{second.port}/four HTTP/1.1\r\n\r\n".encode())
                rest = _rest(client)
        self.assertEqual((one.text, two.text), ("first served GET /one HTTP/1.1", "second served GET /two HTTP/1.1"))
        self.assertEqual(rest, b"")
        self.assertEqual([lines[0] for lines, _ in first.requests],
                         ["POST /upload HTTP/1.1", "GET /one HTTP/1.1", "GET /three HTTP/1.1"])
        self.assertEqual([lines[0] for lines, _ in second.requests], ["GET /two HTTP/1.1"])
        self.assertFalse(any("X-Api-Key" in line for lines, _ in first.requests for line in lines))

    def test_an_address_that_is_also_a_broadcast_address_is_read_as_this_hosts(self):
        # 198.51.100.255 is this host's own and the broadcast address of 198.51.100.0/24: its leaf
        # lists the broadcast route first.
        fib = ("Local:\n  +-- 0.0.0.0/0 2 0 2\n     +-- 127.0.0.0/8 2 0 2\n        +-- 127.0.0.0/31 1 0 0\n"
               "           |-- 127.0.0.1\n              /32 host LOCAL\n        |-- 127.255.255.255\n"
               "           /32 link BROADCAST\n     +-- 198.51.100.0/24 2 0 2\n           |-- 198.51.100.7\n"
               "              /32 host LOCAL\n        |-- 198.51.100.255\n           /32 link BROADCAST\n"
               "           /32 host LOCAL\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "fib_trie")
            path.write_text(fib)
            with patch.multiple(proxy, FIB_TRIE=str(path), IF_INET6=str(Path(directory, "none"))):
                self.assertEqual(sorted(str(address) for address in proxy.local_addresses()),
                                 ["127.0.0.1", "198.51.100.255", "198.51.100.7"])

    def test_a_client_that_ends_its_upload_still_gets_the_response(self):
        server = serve(Echo)
        self.addCleanup(server.shutdown)
        with patch.object(proxy, "destination_refused", allow_all):
            outbound = self.start(ports=[server.server_port])
            with self.connect(outbound) as client:
                client.sendall(f"POST http://127.0.0.1:{server.server_port}/up HTTP/1.1\r\nContent-Length: 3\r\n"
                               "\r\nabc".encode())
                client.shutdown(socket.SHUT_WR)  # as a client that half-closes after its request does
                answer = _rest(client)
        self.assertTrue(answer.startswith(b"HTTP/1.0 200 OK"), answer[:80])
        self.assertTrue(answer.endswith(b"Connection: close\nabc"), answer[-80:])

    def test_an_interim_100_continue_reaches_a_client_that_waits_for_it(self):
        class Continuing(Echo):
            protocol_version = "HTTP/1.1"  # http.server answers Expect: 100-continue from HTTP/1.1 on
        server = serve(Continuing)
        self.addCleanup(server.shutdown)
        with patch.object(proxy, "destination_refused", allow_all):
            outbound = self.start(ports=[server.server_port])
            with self.connect(outbound) as client:
                client.settimeout(5)
                client.sendall(f"POST http://127.0.0.1:{server.server_port}/up HTTP/1.1\r\nContent-Length: 3\r\n"
                               "Expect: 100-continue\r\n\r\n".encode())
                interim = client.recv(4096)  # before the body is sent
                client.sendall(b"abc")
                answer = _rest(client)
        self.assertEqual(interim, b"HTTP/1.1 100 Continue\r\n\r\n")
        self.assertTrue(answer.startswith(b"HTTP/1.1 200 OK"), answer[:80])
        self.assertTrue(answer.endswith(b"abc"), answer[-80:])

    def test_tunnels_relay_both_ways_at_once(self):
        size = 8 << 20  # far more than the sockets buffer: one side waits for the other unless both directions run
        listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(listener.close)

        def stream():
            connection, _ = listener.accept()
            with connection:
                reader = threading.Thread(target=lambda: [None for _ in iter(lambda: connection.recv(65536), b"")],
                                          daemon=True)
                reader.start()
                connection.sendall(b"s" * size)
                connection.shutdown(socket.SHUT_WR)
                reader.join(30)
        threading.Thread(target=stream, daemon=True).start()
        port = listener.getsockname()[1]
        with patch.object(proxy, "destination_refused", allow_all):
            outbound = self.start(ports=[port])
            with self.connect(outbound) as client:
                client.sendall(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
                head = b""
                while not head.endswith(b"\r\n\r\n"):
                    head += client.recv(1)
                client.sendall(b"c" * size)  # sent before reading anything back, as an upload would
                client.shutdown(socket.SHUT_WR)
                received = 0
                while chunk := client.recv(65536):
                    received += len(chunk)
        self.assertEqual((head, received), (proxy.ESTABLISHED, size))

    def test_heads_refusals_and_connections_are_bounded(self):
        answer = self.exchange(b"A" * (proxy.MAX_HEADER_BYTES + 1000))  # all read before the refusal
        self.assertTrue(answer.startswith(b"HTTP/1.1 431"), answer[:80])
        with patch.object(proxy, "HEAD_SECONDS", 1), patch.object(proxy, "REPLY_SECONDS", 0.2):
            # A head sent a byte at a time gets HEAD_SECONDS.
            with self.connect() as client:
                start = time.monotonic()
                answer = b""
                try:
                    for byte in b"CONNECT 127.0.0.1:9 HTTP/1.1\r\n\r\n":
                        client.send(bytes([byte]))
                        time.sleep(0.2)
                except OSError:
                    pass
                answer = client.recv(4096)
        self.assertTrue(answer.startswith(b"HTTP/1.1 408"), answer[:80])
        self.assertLess(time.monotonic() - start, 3)
        with patch.object(proxy, "REPLY_SECONDS", 0.5), self.connect() as client:  # a refusal's reading is bounded
            client.sendall(b"CONNECT 127.0.0.1:9 HTTP/1.1\r\nContent-Length: 99999\r\n\r\n")
            start = time.monotonic()
            try:
                while time.monotonic() - start < 5:
                    client.send(b"x")
                    time.sleep(0.05)
            except OSError:
                pass
            answer = _rest(client)  # the proxy closes with this client's last bytes unread, so the end may reset
        self.assertTrue(answer.startswith(b"HTTP/1.1 403"), answer[:80])
        self.assertLess(time.monotonic() - start, 3)
        with patch.object(proxy, "MAX_CONNECTIONS", 1):
            bounded = self.start()
        with self.connect(bounded) as held:  # holds the only slot while its header never ends
            held.sendall(b"CONNECT ")
            time.sleep(0.2)
            threads = threading.active_count()
            for _ in range(20):  # refused at once: no thread each, and nothing read
                with self.connect(bounded) as client:
                    try:
                        client.sendall(b"CONNECT 127.0.0.1:9 HTTP/1.1\r\n\r\n")
                    except BrokenPipeError:  # closed before the request went out; the answer is there
                        pass
                    answer = _rest(client)
                    self.assertTrue(answer.startswith(b"HTTP/1.1 503 Service Unavailable\r\n"), answer)
                    self.assertTrue(answer.endswith(b"\r\n\r\nmore than 1 connections are open\n"), answer)
            self.assertLessEqual(threading.active_count(), threads)

    def test_accepting_survives_running_out_of_descriptors(self):
        class Listener:
            def __init__(self):
                self.calls, self.connection = 0, socket.socketpair()

            def accept(self):
                self.calls += 1
                if self.calls <= 3:
                    raise OSError(errno.EMFILE, "Too many open files")
                if self.calls == 4:
                    return self.connection[0], None
                raise OSError(errno.EBADF, "Bad file descriptor")
        listener, admitted = Listener(), []
        self.addCleanup(lambda: [end.close() for end in listener.connection])
        with self.assertLogs(proxy.logger, "WARNING") as logs:
            proxy._accept(listener, admitted.append, lambda: listener.calls > 4)
        self.assertEqual((admitted, len(logs.output)), ([listener.connection[0]], 1))
        self.assertIn("Too many open files", logs.output[0])

    def test_a_thread_that_cannot_start_releases_its_slot_and_closes_the_connection(self):
        real = threading.Thread.start
        failed = []

        def start(thread):
            if getattr(thread, "_target", None) == self.proxy._handle and not failed:
                failed.append(thread)
                raise RuntimeError("can't start new thread")
            return real(thread)
        with patch.object(threading.Thread, "start", start):
            try:
                first = self.exchange(b"CONNECT 127.0.0.1:9 HTTP/1.1\r\n\r\n")
            except OSError:  # closed before the request went out
                first = b""
        second = self.exchange(b"CONNECT 127.0.0.1:9 HTTP/1.1\r\n\r\n")
        self.assertEqual((len(failed), first), (1, b""))
        self.assertTrue(second.startswith(b"HTTP/1.1 403"), second[:80])
        time.sleep(0.2)
        self.assertEqual(self.proxy.slots._value, proxy.MAX_CONNECTIONS)

    def test_the_socket_lives_in_a_short_private_directory_whatever_tmpdir_says(self):
        with tempfile.TemporaryDirectory() as directory:
            deep = Path(directory, "d" * 100)
            deep.mkdir()
            with patch.dict(os.environ, {"TMPDIR": str(deep)}), patch.object(tempfile, "tempdir", None):
                outbound = self.start()
            self.assertEqual(Path(outbound.path).parent.parent, Path(proxy.SOCKET_BASE))
            self.assertTrue(Path(outbound.path).parent.name.startswith("openvlharness-proxy-"))
            self.assertEqual(Path(outbound.path).parent.stat().st_mode & 0o777, 0o700)
            outbound.close()
            self.assertFalse(Path(outbound.path).parent.exists())
            with patch.object(proxy, "SOCKET_BASE", str(deep)), \
                    self.assertRaisesRegex(ValueError, "socket path .* is longer than 107 bytes"):
                proxy.OutboundProxy()
            self.assertEqual(list(deep.iterdir()), [])  # the directory made for it is gone
            given = Path(directory, "given.sock")
            outbound = proxy.OutboundProxy(str(given))
            outbound.close()
            self.assertFalse(given.exists())

    def test_the_routers_own_proxy_tunnels_to_the_checked_address_and_keeps_what_follows_its_answer(self):
        seen = []
        upstream = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(upstream.close)

        def answer():  # the router's proxy, and behind its tunnels the destination
            while True:
                try:
                    connection, _ = upstream.accept()
                except OSError:
                    return
                with connection:
                    data = b""
                    while b"\r\n\r\n" not in data:
                        data += connection.recv(4096)
                    head, data = data.split(b"\r\n\r\n", 1)
                    seen.append(head.decode())
                    port = head.split(b" ")[1].rsplit(b":", 1)[1]
                    if port == b"443":
                        connection.sendall(proxy.ESTABLISHED + b"SSH-2.0-banner\r\n")  # in the same read
                        time.sleep(0.5)
                        continue
                    if port == b"8443":
                        connection.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                        continue
                    connection.sendall(proxy.ESTABLISHED)
                    while b"\r\n\r\n" not in data:
                        data += connection.recv(4096)
                    seen.append(data.split(b"\r\n\r\n", 1)[0].decode())
                    connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        threading.Thread(target=answer, daemon=True).start()
        address = f"http://user:secret@127.0.0.1:{upstream.getsockname()[1]}"
        with patch.dict(os.environ, {"HTTPS_PROXY": address, "HTTP_PROXY": address}), \
                patch.object(proxy, "destination_refused", allow_all):
            outbound = self.start(ports=[443, 8080, 8443])
            with self.connect(outbound) as client:
                client.sendall(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n")
                tunnel = b""
                while b"banner" not in tunnel:
                    tunnel += client.recv(4096)
            plain = self.exchange(b"GET http://localhost:8080/page?x=1 HTTP/1.1\r\nHost: localhost:8080\r\n\r\n",
                                  outbound)
            refused = self.exchange(b"CONNECT localhost:8443 HTTP/1.1\r\n\r\n", outbound)
        self.assertEqual(tunnel, proxy.ESTABLISHED + b"SSH-2.0-banner\r\n")
        connect = [head.split("\r\n") for head in seen if head.startswith("CONNECT")]
        self.assertIn(connect[0][0], ("CONNECT 127.0.0.1:443 HTTP/1.1", "CONNECT [::1]:443 HTTP/1.1"))
        self.assertIn("Proxy-Authorization: Basic dXNlcjpzZWNyZXQ=", connect[0])
        # A plain request goes through a tunnel to the checked address too, in origin form with its own Host.
        self.assertIn(connect[1][0], ("CONNECT 127.0.0.1:8080 HTTP/1.1", "CONNECT [::1]:8080 HTTP/1.1"))
        self.assertEqual(next(head for head in seen if head.startswith("GET")).split("\r\n"),
                         ["GET /page?x=1 HTTP/1.1", "Host: localhost:8080", "Connection: close"])
        self.assertTrue(plain.startswith(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"), plain)
        self.assertTrue(refused.startswith(b"HTTP/1.1 502"), refused[:80])
        self.assertIn(b"the router's proxy refused a tunnel to localhost:8443 (403)", refused)

    def test_a_name_that_does_not_resolve_gets_the_same_answer_as_one_that_is_refused(self):
        real = socket.getaddrinfo

        def resolve(host, *args, **options):
            if host == "db.internal.example":
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            return real(host, *args, **options)
        outbound = self.start(deny=["8.8.8.0/24"])
        with patch.object(proxy.socket, "getaddrinfo", resolve), self.assertLogs(proxy.logger, "DEBUG") as logs:
            answers = {target: self.exchange(f"CONNECT {target}:443 HTTP/1.1\r\n\r\n".encode(), outbound)
                       for target in ("db.internal.example", "localhost", "8.8.8.8")}
        for target, answer in answers.items():  # one status and reason, so a tool cannot tell which names resolve
            with self.subTest(target=target):
                self.assertTrue(answer.startswith(b"HTTP/1.1 403 Forbidden"), answer[:80])
                self.assertTrue(answer.endswith(f"\r\n\r\n{target} {proxy.REFUSED}\n".encode()), answer)
        # The router's log says which it was.
        self.assertEqual([line.split(": ", 1)[1] for line in logs.output],
                         ["it does not resolve",
                          "an address it resolves to is not public, or is this host's or on its local networks",
                          "an address it resolves to is in toolsmith.worker.deny_networks"])

    def test_all_proxy_is_the_routers_proxy_when_neither_http_proxy_nor_https_proxy_is_set(self):
        seen = []
        upstream = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(upstream.close)

        def answer():
            while True:
                try:
                    connection, _ = upstream.accept()
                except OSError:
                    return
                with connection:
                    seen.append(connection.recv(4096).split(b"\r\n", 1)[0].decode())
                    connection.sendall(proxy.ESTABLISHED)
        threading.Thread(target=answer, daemon=True).start()
        address = f"http://127.0.0.1:{upstream.getsockname()[1]}"
        for variables, through in (({"ALL_PROXY": address}, True),
                                   ({"all_proxy": address, "NO_PROXY": "localhost"}, False),  # NO_PROXY still holds
                                   ({"ALL_PROXY": "socks5://127.0.0.1:1", "HTTPS_PROXY": address}, True)):
            with self.subTest(variables=variables), patch.dict(os.environ, variables), \
                    patch.object(proxy, "destination_refused", allow_all):
                seen.clear()
                outbound = self.start(ports=[443])
                answer = self.exchange(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n", outbound)
                if through:
                    self.assertTrue(answer.startswith(proxy.ESTABLISHED), answer[:80])
                    self.assertIn(seen[0], ("CONNECT 127.0.0.1:443 HTTP/1.1", "CONNECT [::1]:443 HTTP/1.1"))
                else:
                    self.assertEqual(seen, [])
        with patch.dict(os.environ, {"ALL_PROXY": "socks5://127.0.0.1:1080"}), \
                patch.object(proxy, "destination_refused", allow_all), \
                self.assertLogs(proxy.logger, "WARNING") as logs:
            outbound = self.start(ports=[443])
            answer = self.exchange(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n", outbound)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("all proxy setting is not an http:// proxy", logs.output[0])
        self.assertTrue(answer.startswith(b"HTTP/1.1 502"), answer[:80])

    def test_a_router_proxy_that_is_not_http_is_reported_and_refused(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "socks5://127.0.0.1:1080"}), \
                patch.object(proxy, "destination_refused", allow_all), \
                self.assertLogs(proxy.logger, "WARNING") as logs:
            outbound = self.start(ports=[443])
            answer = self.exchange(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n", outbound)
        self.assertIn("https proxy setting is not an http:// proxy", logs.output[0])
        self.assertNotIn("1080", logs.output[0])
        self.assertTrue(answer.startswith(b"HTTP/1.1 502"), answer[:80])
        self.assertIn(b"not an http:// proxy (https:// and SOCKS proxies are not supported)", answer)


@pytest.mark.skipif(not os.environ.get("OPENVLHARNESS_NETWORK_TESTS"),
                    reason="reaches the internet; set OPENVLHARNESS_NETWORK_TESTS=1 to run it")
def test_tls_to_a_public_host_verifies_through_the_tunnel(monkeypatch, tmp_path):
    monkeypatch.setattr(socket.socket, "connect", REAL_CONNECT)  # the suite refuses non-loopback connections
    outbound = proxy.OutboundProxy(str(tmp_path / "proxy.sock"))
    host = "www.wikidata.org"
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(30)
            client.connect(outbound.path)
            client.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
            assert client.recv(4096) == proxy.ESTABLISHED
            with ssl.create_default_context().wrap_socket(client, server_hostname=host) as tls:  # no request sent
                names = [value for kind, value in tls.getpeercert()["subjectAltName"] if kind == "DNS"]
    finally:
        outbound.close()
    assert "*.wikidata.org" in names or host in names
