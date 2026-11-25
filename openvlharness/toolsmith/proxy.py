"""The outbound proxy of confined generated tools: their only route out of the sandbox.

A confined worker has its own network namespace with only loopback up. Its HTTP and
HTTPS clients use a relay on that loopback (forward), which passes each connection
through a UNIX socket mounted into the sandbox to this proxy, running in the router.

The proxy serves one request per connection: a CONNECT tunnel, or one absolute-form
http:// request, which it sends on in origin form with exactly its Content-Length body
and answers with the response alone. Request heads are parsed strictly: lines end in
CRLF, hold no other control character than tab, and are not folded. Only the ports in
toolsmith.worker.network_ports (80 and 443 by default) are served. The proxy resolves
the host itself, on the router's host, and refuses when the name does not resolve, or
when any address is not a public address of another host (see destination_refused) or
is in toolsmith.worker.deny_networks, with the same answer (REFUSED) in each case. It then
connects to an address it checked, never resolving the name again: directly, or through
a CONNECT tunnel to that address, an IP literal, from the router's own HTTP proxy
(HTTP_PROXY or HTTPS_PROXY, else ALL_PROXY; NO_PROXY), which must be an http:// proxy
that allows CONNECT to those ports. All the connections of one proxy, that is of one
router, share MAX_CONNECTIONS.
"""
import base64
import ipaddress
import logging
import os
import re
import selectors
import shutil
import socket
import struct
import tempfile
import threading
import time
import urllib.request
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

PORTS = (80, 443)  # the destination ports served unless toolsmith.worker.network_ports says otherwise
MAX_HEADER_BYTES = 64 << 10
HEAD_SECONDS = 30  # to send a whole request head
CONNECT_SECONDS = 15
IDLE_SECONDS = 120  # a connection that moves no byte for this long (twice as long at most) is closed
MAX_CONNECTIONS = 256  # per proxy, that is per router; connections past it get 503 and are closed at once
REPLY_SECONDS = 1  # how long a refused request's further bytes are read before the connection closes
HOST_SECONDS = 5  # how long this host's addresses and networks are used before they are read again
# A UNIX socket path has at most 107 bytes, so the socket directory goes under a short base
# whatever TMPDIR says.
SOCKET_BASE = "/tmp"
MAX_SOCKET_PATH = 107
SHARED = ipaddress.ip_network("100.64.0.0/10")  # carrier-grade NAT
NAT64 = ipaddress.ip_network("64:ff9b::/96")  # the IPv4 address is the last 32 bits
NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")  # local-use translation, any embedding
# Addresses a platform serves its machines at although they are public: Azure's virtual machine agent,
# DNS and health endpoint. Other platforms' belong in toolsmith.worker.deny_networks.
PLATFORM = {ipaddress.ip_address("168.63.129.16")}
# Routes without a gateway that cover more than this count as default routes: a VPN that sends
# everything through an interface installs 0.0.0.0/1 and 128.0.0.0/1 that way.
WIDEST_LINK = {4: 8, 6: 16}
RTF_UP, RTF_GATEWAY, RTF_REJECT = 0x1, 0x2, 0x200
# Where Linux lists this host's addresses and routes.
FIB_TRIE, IF_INET6 = "/proc/net/fib_trie", "/proc/net/if_inet6"
ROUTE, IPV6_ROUTE = "/proc/net/route", "/proc/net/ipv6_route"
# The answer to a name that does not resolve and to one that resolves where tools may not go alike, so a
# tool cannot learn which names the router's resolver knows.
REFUSED = ("does not resolve to an address the proxy may reach: a public one outside this host, its local networks "
           "and toolsmith.worker.deny_networks")
CLOSE_FIELDS = (b"connection", b"keep-alive", b"proxy-connection")  # response fields Connection: close replaces
HOP_BY_HOP = {"connection", "keep-alive", "proxy-connection", "proxy-authorization", "proxy-authenticate", "te",
              "trailer", "transfer-encoding", "upgrade", "host"}
TOKEN = re.compile(rb"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
TARGET = re.compile(rb"[\x21-\x7e]+")
VERSION = re.compile(rb"HTTP/1\.[01]")
CONTROL = re.compile(rb"[\x00-\x08\x0a-\x1f\x7f]")  # control characters but tab, so bare CR and LF too
AUTHORITY = re.compile(r"(?:\[([0-9A-Fa-f:.]+)\]|([A-Za-z0-9._-]+))(?::([0-9]{1,5}))?")
ABSOLUTE = re.compile(r"[Hh][Tt][Tt][Pp]://([^/?#]*)([^#]*)(?:#.*)?")
ESTABLISHED = b"HTTP/1.1 200 Connection established\r\n\r\n"


class _Refusal(Exception):
    """A request the proxy answers with an error status instead of relaying."""

    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


class HostUnknown(RuntimeError):
    """This host's addresses or networks cannot be read, so no destination can be checked."""


def local_addresses():
    """Return the addresses of this host's network interfaces (Linux).

    Raises:
        HostUnknown: FIB_TRIE cannot be read or lists none, or IF_INET6 exists but cannot be read.
    """
    found, leaf = set(), None
    try:
        with open(FIB_TRIE) as stream:
            for line in stream:
                # A leaf ("|-- address") lists its routes below it, a broadcast one before a local
                # one when an address of this host is also another network's broadcast address.
                if line.lstrip().startswith("|-- "):
                    leaf = line.split()[-1]
                elif "/32 host LOCAL" in line and leaf is not None:
                    found.add(ipaddress.ip_address(leaf))
    except (OSError, ValueError, IndexError) as exc:
        raise HostUnknown(f"cannot read this host's IPv4 addresses from {FIB_TRIE} ({exc})") from None
    if not found:
        raise HostUnknown(f"{FIB_TRIE} lists no IPv4 address of this host")
    try:
        with open(IF_INET6) as stream:
            found.update(ipaddress.IPv6Address(int(line.split()[0], 16)) for line in stream if line.strip())
    except FileNotFoundError:
        pass  # without IPv6 this host has no IPv6 address
    except (OSError, ValueError, IndexError) as exc:
        raise HostUnknown(f"cannot read this host's IPv6 addresses from {IF_INET6} ({exc})") from None
    return found


def link_networks():
    """Return the networks this host's main routing table reaches without a gateway: its LANs and links.

    A VPN's device routes count as well, since private services may have addresses in them:
    with a split tunnel that routes public ranges that way, public hosts in them are refused.

    Raises:
        HostUnknown: ROUTE cannot be read, or IPV6_ROUTE exists but cannot be read.
    """
    found = []
    try:
        with open(ROUTE) as stream:
            for line in stream.read().splitlines()[1:]:
                fields = line.split()
                destination, flags, mask = int(fields[1], 16), int(fields[3], 16), int(fields[7], 16)
                if flags & RTF_UP and not flags & (RTF_GATEWAY | RTF_REJECT):
                    # The kernel prints the network-order address as a native integer.
                    found.append(ipaddress.IPv4Network((struct.pack("=I", destination), bin(mask).count("1")),
                                                       strict=False))
    except (OSError, ValueError, IndexError, struct.error) as exc:
        raise HostUnknown(f"cannot read this host's IPv4 routes from {ROUTE} ({exc})") from None
    try:
        with open(IPV6_ROUTE) as stream:
            for line in stream:
                fields = line.split()
                if fields and int(fields[8], 16) & (RTF_UP | RTF_GATEWAY | RTF_REJECT) == RTF_UP:
                    found.append(ipaddress.IPv6Network((int(fields[0], 16), int(fields[1], 16)), strict=False))
    except FileNotFoundError:
        pass
    except (OSError, ValueError, IndexError) as exc:
        raise HostUnknown(f"cannot read this host's IPv6 routes from {IPV6_ROUTE} ({exc})") from None
    return [network for network in found if network.prefixlen >= WIDEST_LINK[network.version]]


_host = {"until": 0.0, "value": None}
_host_lock = threading.Lock()


def host_networks():
    """Return this host's addresses and link networks (see link_networks), read again after HOST_SECONDS.

    Raises:
        HostUnknown: They cannot be read.
    """
    with _host_lock:
        if _host["value"] is None or time.monotonic() >= _host["until"]:
            _host["value"] = None
            _host["value"] = (local_addresses(), link_networks())
            _host["until"] = time.monotonic() + HOST_SECONDS
        return _host["value"]


def _forms(address):
    """The address and the IPv4 addresses an IPv6 address embeds: mapped, compatible, NAT64, 6to4, Teredo."""
    forms = [address]
    if address.version == 6:
        forms += [embedded for embedded in (address.ipv4_mapped, address.sixtofour) if embedded]
        forms += list(address.teredo or ())
        if address in NAT64 or 1 < int(address) < 1 << 32:
            forms.append(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    return forms


def destination_refused(addresses):
    """Return why the proxy refuses to connect to addresses, or None when every one may be reached.

    An address is refused when it, or an IPv4 address it embeds, is loopback, unspecified,
    private, link-local, site-local (fec0::/10), shared (100.64/10), multicast, reserved or
    otherwise not global, an address of this host or of PLATFORM, or in a network this host
    reaches without a gateway: hosts on its LANs are refused even with public addresses. Not
    recognized, and for deny_networks to list: public addresses that lead back to this host
    or its network through 1:1 NAT or hairpin NAT, networks that only policy routing tables
    reach, a site's own NAT64 prefix, and other platforms' addresses. The reason,
    REFUSED, never names the address. Tests replace this function to reach a server of their
    own; no setting changes it, and toolsmith.worker.deny_networks refuses more.

    Raises:
        HostUnknown: This host's addresses or networks cannot be read.
    """
    local, links = host_networks()
    for text in addresses:
        for address in _forms(ipaddress.ip_address(text.split("%", 1)[0])):  # without an IPv6 zone
            if (not address.is_global or address.is_loopback or address.is_unspecified or address.is_private
                    or address.is_link_local or address.is_multicast or address.is_reserved or address in SHARED
                    or (address.version == 6 and (address.is_site_local or address in NAT64_LOCAL))
                    or address in local or address in PLATFORM or any(address in network for network in links)):
                return REFUSED
    return None


def _denied(addresses, networks):
    """Whether an address, or an IPv4 address it embeds, is in one of networks (toolsmith.worker.deny_networks)."""
    return any(address in network for text in addresses
               for address in _forms(ipaddress.ip_address(text.split("%", 1)[0])) for network in networks)


class OutboundProxy:
    """The proxy of one router's confined workers, listening on a UNIX socket.

    Args:
        path: Where the socket goes; by default proxy.sock in a new directory under
            SOCKET_BASE that only this user can open, removed by close().
        ports: The destination ports served; None serves every port.
        deny: Networks (CIDR strings) refused on top of destination_refused.

    Raises:
        ValueError: The socket path is longer than a UNIX socket path can be.
        OSError: The directory or the socket cannot be made.
    """

    def __init__(self, path=None, *, ports=PORTS, deny=()):
        self.ports = None if ports is None else frozenset(ports)
        self.deny = [ipaddress.ip_network(network, strict=False) for network in deny]
        self.directory = None
        self.closed = False
        if path is None:
            self.directory = tempfile.mkdtemp(prefix="openvlharness-proxy-", dir=SOCKET_BASE)
            path = os.path.join(self.directory, "proxy.sock")
        self.path = path
        try:
            if len(os.fsencode(path)) > MAX_SOCKET_PATH:
                raise ValueError(f"the outbound proxy's socket path {path} is longer than {MAX_SOCKET_PATH} bytes")
            self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                self.listener.bind(path)
                self.listener.listen(64)
            except OSError:
                self.listener.close()
                raise
            self.capacity = MAX_CONNECTIONS
            self.slots = threading.BoundedSemaphore(self.capacity)
            threading.Thread(target=_accept, args=(self.listener, self._admit, lambda: self.closed),
                             name="toolsmith-proxy", daemon=True).start()
        except BaseException:
            if self.directory:
                shutil.rmtree(self.directory, ignore_errors=True)
            raise
        proxies = urllib.request.getproxies_environment()
        used = {scheme if proxies.get(scheme) else "all" for scheme in ("http", "https")}  # the settings _upstream reads
        for name in sorted(name for name in used if proxies.get(name) and not _supported(proxies[name])):
            logger.warning("The router's %s proxy setting is not an http:// proxy: generated tools' connections "
                           "through it will fail (https:// and SOCKS proxies are not supported).", name)

    def close(self):
        """Stop accepting connections and remove the socket (and its own directory); open connections run on."""
        self.closed = True
        try:
            self.listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.listener.close()
        if self.directory:
            shutil.rmtree(self.directory, ignore_errors=True)
        else:
            try:
                os.unlink(self.path)
            except OSError:
                pass

    def _admit(self, client):
        """Serve a new connection in a thread of its own, or close it at once when MAX_CONNECTIONS are open."""
        if not self.slots.acquire(blocking=False):
            body = f"more than {self.capacity} connections are open\n".encode()
            try:
                client.setblocking(False)  # a fresh socket's buffer takes the answer; nothing waits for it
                client.send(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nContent-Length: "
                            b"%d\r\nConnection: close\r\n\r\n%s" % (len(body), body))
            except OSError:
                pass
            client.close()
            return
        try:
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()
        except RuntimeError:  # no thread can start now
            self.slots.release()
            client.close()

    def _handle(self, client):
        """Serve one connection: a CONNECT tunnel, or one absolute-form request and its response."""
        server = None
        try:
            try:
                head, rest = _read_head(client, time.monotonic() + HEAD_SECONDS)
                method, target, version, fields = _parse(head)
                client.settimeout(IDLE_SECONDS)
                if method == "CONNECT":
                    host, port = _host_port(target)
                    server, early = self._open(host, port, "https")
                    client.sendall(ESTABLISHED + early)
                    if rest:
                        server.sendall(rest)
                    relay(client, server, IDLE_SECONDS)
                else:
                    host, port, authority, path = _absolute(target)
                    length = _body_length(fields)
                    server, early = self._open(host, port, "http")
                    body = rest[:length]  # bytes past the body would be another request: never sent on
                    server.sendall(_origin_request(method, path, version, fields, authority) + body)
                    _respond(server, client, early, length - len(body))
            except _Refusal as refusal:
                _reply(client, refusal.status, refusal.reason)
        except (OSError, RuntimeError):
            pass  # a peer went away, a connection idled out, or no thread could start
        except Exception as exc:  # a bug: this connection closes, the proxy goes on
            logger.warning("The outbound proxy dropped a connection: %s: %s", type(exc).__name__, exc)
        finally:
            for sock in (server, client):
                if sock is not None:
                    sock.close()
            self.slots.release()

    def _open(self, host, port, scheme):
        """Check the port and every address host resolves to, and connect to a checked address.

        Returns:
            (socket, bytes the router's own proxy sent past its answer to CONNECT).
        """
        if self.ports is not None and port not in self.ports:
            allowed = ", ".join(str(each) for each in sorted(self.ports))
            raise _Refusal(403, f"port {port} is not allowed (toolsmith.worker.network_ports: {allowed})")
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except (OSError, UnicodeError, ValueError):
            infos = []
        addresses = [info[4][0] for info in infos]
        try:
            # Asked for a name that does not resolve too: when this host's networks cannot be read, every
            # request gets the same 503.
            refused = destination_refused(addresses)
        except HostUnknown as exc:
            raise _Refusal(503, f"the proxy cannot check destinations: {exc}") from None
        if not addresses:
            why = "it does not resolve"
        elif refused:
            why = "an address it resolves to is not public, or is this host's or on its local networks"
        elif _denied(addresses, self.deny):
            why = "an address it resolves to is in toolsmith.worker.deny_networks"
        else:
            why = None
        if why:
            logger.debug("The outbound proxy refused %s: %s", host, why)  # the tool gets REFUSED whatever the reason
            raise _Refusal(403, f"{host} {REFUSED}")
        upstream = _upstream(scheme, host)
        failure = None
        for family, _, _, _, address in infos:
            try:
                if upstream is not None:
                    return _tunnel(upstream, family, address[0], port, host)
                server = socket.socket(family, socket.SOCK_STREAM)
                server.settimeout(CONNECT_SECONDS)
                try:
                    server.connect(address)
                except OSError:
                    server.close()
                    raise
                server.settimeout(IDLE_SECONDS)
                return server, b""
            except OSError as exc:
                failure = exc
        raise _Refusal(502, f"cannot connect to {host}:{port}: {getattr(failure, 'strerror', None) or failure}")


def _accept(listener, admit, closed=lambda: False):
    """Accept connections and pass each to admit until closed() tells the listener was closed.

    Running out of file descriptors (EMFILE) or memory fails one accept, not the loop: the
    first failure in a row is logged, and the loop waits a little longer after each.
    """
    pause = 0.0
    while True:
        try:
            client, _ = listener.accept()
        except OSError as exc:
            if closed():
                return
            if not pause:
                logger.warning("The outbound proxy cannot accept a connection, and retries: %s", exc)
            pause = min(1.0, pause * 2 or 0.01)
            time.sleep(pause)
            continue
        pause = 0.0
        admit(client)


def _reply(client, status, reason):
    """Answer a request with an error status and its reason, and no relay.

    What the client still sends is read for at most REPLY_SECONDS before the connection
    closes: a TCP socket closed with unread data resets the connection, which can lose the
    answer, and the relay in the worker passes this one on over TCP.
    """
    body = (reason + "\n").encode()
    try:
        client.sendall(f"HTTP/1.1 {status} {_PHRASES.get(status, 'Error')}\r\nContent-Type: text/plain\r\n"
                       f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        client.shutdown(socket.SHUT_WR)
        deadline, drained = time.monotonic() + REPLY_SECONDS, 0
        while drained < MAX_HEADER_BYTES and (remaining := deadline - time.monotonic()) > 0:
            client.settimeout(remaining)
            chunk = client.recv(65536)
            if not chunk:
                break
            drained += len(chunk)
    except OSError:
        pass


_PHRASES = {400: "Bad Request", 403: "Forbidden", 408: "Request Timeout", 411: "Length Required",
            431: "Request Header Fields Too Large", 502: "Bad Gateway", 503: "Service Unavailable"}


def _read_head(sock, deadline, what="request header"):
    """Read up to the end of a header block by deadline (time.monotonic); return it and the bytes after it."""
    data, start = bytearray(), 0
    while (end := data.find(b"\r\n\r\n", start)) < 0:
        start = max(0, len(data) - 3)  # the end is not in what was searched already
        if len(data) > MAX_HEADER_BYTES:
            raise _Refusal(431, f"the {what} is longer than {MAX_HEADER_BYTES} bytes")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _Refusal(408, f"the {what} did not arrive in time")
        sock.settimeout(remaining)
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            raise _Refusal(408, f"the {what} did not arrive in time") from None
        if not chunk:
            raise _Refusal(400, f"the connection closed before the {what} ended")
        data += chunk
    if end > MAX_HEADER_BYTES:
        raise _Refusal(431, f"the {what} is longer than {MAX_HEADER_BYTES} bytes")
    return bytes(data[:end]), bytes(data[end + 4:])


def _parse(head):
    """Split a request head into method, target, version and header fields; refuse anything loose.

    Every line ends in CRLF and holds no other control character than tab (the request line
    not even that), header names are tokens right before their colon, and no line is
    folded, so the head reads the same to every parser after this one.

    Returns:
        (method, target, version, fields), each field (lowercase name, value, raw line).
    """
    lines = head.split(b"\r\n")
    if any(CONTROL.search(line) for line in lines):
        raise _Refusal(400, "the request head holds a bare CR or LF, or another control character")
    parts = lines[0].split(b" ")
    if len(parts) != 3 or not TOKEN.fullmatch(parts[0]) or not TARGET.fullmatch(parts[1]) \
            or not VERSION.fullmatch(parts[2]):
        raise _Refusal(400, "malformed request line")
    fields = []
    for line in lines[1:]:
        name, colon, value = line.partition(b":")
        if not colon or not TOKEN.fullmatch(name):  # also a folded line, which starts with space or tab
            raise _Refusal(400, "malformed header line")
        fields.append((name.decode("ascii").lower(), value.strip(b" \t").decode("latin-1"), line))
    if sum(name == "host" for name, _, _ in fields) > 1:
        raise _Refusal(400, "more than one Host header")
    return parts[0].decode("ascii"), parts[1].decode("ascii"), parts[2].decode("ascii"), fields


def _host_port(authority, default=None):
    """Split host:port or [IPv6]:port, the port defaulting to default; refuse anything else."""
    match = AUTHORITY.fullmatch(authority)
    port = None
    if match:
        literal, name, digits = match.groups()
        port = int(digits) if digits else default
        if literal is not None:
            try:
                ipaddress.IPv6Address(literal)
            except ValueError:
                match = None
    if not match or port is None or not 0 < port < 65536:
        raise _Refusal(400, f"not a host and port: {authority}")
    return literal if literal is not None else name, port


def _absolute(target):
    """Split an absolute http:// target into host, port, its authority (the Host field) and origin form."""
    match = ABSOLUTE.fullmatch(target)
    if not match:
        raise _Refusal(400, "only CONNECT and absolute http:// requests are proxied")
    authority, rest = match.groups()
    host, port = _host_port(authority, 80)
    return host, port, authority, rest if rest.startswith("/") else "/" + rest


def _body_length(fields):
    """The length of the request body, which only Content-Length may give.

    A Connection field that lists Content-Length would have it dropped on the way, and the
    body read as another request, so it is refused.
    """
    if "content-length" in _connection_options(fields):
        raise _Refusal(400, "Connection lists Content-Length")
    if any(name == "transfer-encoding" for name, _, _ in fields):
        raise _Refusal(411, "request bodies need a Content-Length; Transfer-Encoding is not supported")
    lengths = [value for name, value, _ in fields if name == "content-length"]
    if len(lengths) > 1 or (lengths and not re.fullmatch(r"[0-9]{1,18}", lengths[0])):
        raise _Refusal(400, "a request needs one Content-Length of digits, or none")
    return int(lengths[0]) if lengths else 0


def _connection_options(fields):
    """The lowercase names that the request's Connection fields list."""
    return {token.strip().lower() for name, value, _ in fields if name == "connection" for token in value.split(",")}


def _origin_request(method, path, version, fields, authority):
    """The request head to send on: origin form, Host from the target, no hop-by-hop fields, Connection: close."""
    listed = _connection_options(fields)
    lines = [f"{method} {path} {version}".encode("ascii"), f"Host: {authority}".encode("ascii")]
    lines += [line for name, _, line in fields if name not in HOP_BY_HOP and name not in listed]
    lines.append(b"Connection: close")
    return b"\r\n".join(lines) + b"\r\n\r\n"


def _respond(server, client, data, body=0):
    """Pass on the rest of one request's body, and relay its response; nothing more goes to server.

    The response flows while the body is still sent, so an interim 100 Continue reaches a
    client that waits for it. The final response head (past any interim 1xx ones) goes on
    with Connection: close, so the client sends its next request on a new connection,
    checked like this one. The client's end of sending, once its body is complete, is the
    end of its upload: the response still comes back. Anything it sends past the body, or
    its end before the body is complete, ends the connection; so do the server's end and
    IDLE_SECONDS without a byte. data holds response bytes that came before; body is the
    number of body bytes still to come from the client.
    """
    looking = True
    with selectors.DefaultSelector() as selector:
        selector.register(server, selectors.EVENT_READ)
        selector.register(client, selectors.EVENT_READ)
        while True:
            if looking:
                ready, data, looking = _close_head(data)
                if ready:
                    client.sendall(ready)
            events = selector.select(IDLE_SECONDS)
            if not events:
                return
            for key, _ in events:
                if key.fileobj is client:
                    chunk = client.recv(min(body, 65536) if body else 65536)
                    if chunk and body:
                        server.sendall(chunk)
                        body -= len(chunk)
                    elif chunk or body:
                        return  # a further request, or the body cut short: never sent on
                    else:
                        selector.unregister(client)  # the end of the upload; the response still comes back
                    continue
                chunk = server.recv(65536)
                if not chunk:
                    if data:
                        client.sendall(data)
                    return
                if looking:
                    data += chunk
                else:
                    client.sendall(chunk)


def _close_head(data):
    """Find the final response head in data and make it say Connection: close.

    Returns:
        (bytes to pass on now, bytes to keep until more arrive, whether still looking).
        A head that does not end in CRLF CRLF within MAX_HEADER_BYTES passes on as it is.
    """
    ready = b""
    while True:
        end, bare = data.find(b"\r\n\r\n"), data.find(b"\n\n")
        if end < 0 or 0 <= bare < end:
            if bare >= 0 or len(data) > MAX_HEADER_BYTES:
                return ready + data, b"", False
            return ready, data, True
        head, data = data[:end], data[end + 4:]
        lines = head.split(b"\r\n")
        status = lines[0].split(b" ")
        code = status[1] if len(status) > 1 else b""
        if len(code) == 3 and code.startswith(b"1") and code != b"101":  # interim; the final head follows
            ready += head + b"\r\n\r\n"
            continue
        lines[1:] = [line for line in lines[1:] if line.split(b":", 1)[0].strip().lower() not in CLOSE_FIELDS]
        return ready + b"\r\n".join([*lines, b"Connection: close"]) + b"\r\n\r\n" + data, b"", False


def _upstream(scheme, host):
    """Return the proxy the router's environment sets for scheme and host, or None to connect directly.

    HTTP_PROXY or HTTPS_PROXY, else ALL_PROXY, as clients such as requests read them; NO_PROXY
    exempts host.
    """
    proxies = urllib.request.getproxies_environment()
    setting = proxies.get(scheme) or proxies.get("all")
    if not setting or urllib.request.proxy_bypass_environment(host, proxies):
        return None
    return setting


def _proxy_address(setting):
    """The host and port of an http:// proxy setting (scheme optional), or None for another or a broken one."""
    try:
        url = urlsplit(setting if "://" in setting else "http://" + setting)
        if url.scheme == "http" and url.hostname:
            return url.hostname, url.port or 80
    except ValueError:
        pass
    return None


def _supported(setting):
    return _proxy_address(setting) is not None


def _tunnel(upstream, family, address, port, host):
    """Open a tunnel to a checked address through the router's own proxy.

    Returns:
        (socket, bytes the proxy sent past its answer, which belong to the destination).
    """
    proxy_address = _proxy_address(upstream)
    if proxy_address is None:
        raise _Refusal(502, "the router's proxy setting is not an http:// proxy (https:// and SOCKS proxies are "
                            "not supported)")
    url = urlsplit(upstream if "://" in upstream else "http://" + upstream)
    authority = f"[{address}]:{port}" if family == socket.AF_INET6 else f"{address}:{port}"
    credentials = ""
    if url.username is not None:
        token = base64.b64encode(f"{unquote(url.username)}:{unquote(url.password or '')}".encode()).decode()
        credentials = f"Proxy-Authorization: Basic {token}\r\n"
    server = socket.create_connection(proxy_address, timeout=CONNECT_SECONDS)
    try:
        server.sendall(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n{credentials}\r\n".encode())
        head, rest = _read_head(server, time.monotonic() + CONNECT_SECONDS, "answer of the router's proxy")
        status = head.split(b"\r\n", 1)[0].split(b" ")
        code = status[1] if len(status) > 1 else b""
        if not (len(code) == 3 and code.startswith(b"2")):
            raise _Refusal(502, f"the router's proxy refused a tunnel to {host}:{port} "
                                f"({code.decode('latin-1') or 'no status'})")
    except BaseException:
        server.close()
        raise
    server.settimeout(IDLE_SECONDS)
    return server, rest


def relay(first, second, idle=None):
    """Copy bytes both ways, one thread per direction, until both sides have finished.

    A side that stops reading holds up only its own direction. With idle, the connection
    closes once no byte has moved either way for idle seconds (twice that at most).

    Raises:
        RuntimeError: The second thread cannot start.
    """
    moved = [time.monotonic()]

    def copy(source, target):
        try:
            while True:
                try:
                    data = source.recv(65536)
                except socket.timeout:
                    if time.monotonic() - moved[0] < idle:
                        continue
                    raise
                if not data:
                    target.shutdown(socket.SHUT_WR)
                    return
                moved[0] = time.monotonic()
                target.sendall(data)
                moved[0] = time.monotonic()
        except OSError:  # a peer went away or the connection idled out: both directions end
            for sock in (first, second):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    for sock in (first, second):
        sock.settimeout(idle)
    back = threading.Thread(target=copy, args=(second, first), daemon=True)
    back.start()
    copy(first, second)
    back.join()


def forward(path):
    """Relay connections to 127.0.0.1 in this (the sandbox's) network namespace to the proxy at path.

    Returns:
        The port the relay listens on.
    """
    listener = socket.create_server(("127.0.0.1", 0))

    def admit(client):
        try:
            threading.Thread(target=_pass_on, args=(client, path), daemon=True).start()
        except RuntimeError:  # no thread can start now
            client.close()
    threading.Thread(target=_accept, args=(listener, admit), name="toolsmith-relay", daemon=True).start()
    return listener.getsockname()[1]


def _pass_on(client, path):
    with client, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as proxy:
        try:
            proxy.connect(path)
            relay(client, proxy)
        except (OSError, RuntimeError):
            pass
