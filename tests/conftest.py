"""Shared test setup: tests run offline; evaluator tests need the eval extra."""
import importlib
import socket

import pytest

# Modules only the eval extra provides: evaluation_kit imports them, and the
# evaluation tests write xlsx prediction files.
try:
    for module in ("openpyxl", "pandas", "portalocker", "sty", "validators", "xlsxwriter"):
        importlib.import_module(module)
except ImportError:
    collect_ignore_glob = ["evaluation/*"]

CREDENTIALS = ("OPENAI_API_KEY", "OPENAI_API_BASE", "AZURE_OPENAI_API_KEY", "SERPER_API_KEY",
               "MOONSHOT_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Drop provider credentials and proxy settings; refuse non-loopback connections.

    Without this, code that builds a client from the environment would reach a
    real model, judge or search API whenever the shell exports a key. Proxies
    are disabled because a loopback proxy would forward requests off the machine.
    """
    for name in CREDENTIALS + PROXIES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    connect = socket.socket.connect

    def loopback_only(sock, address):
        host = address[0] if isinstance(address, tuple) else None
        if sock.family != socket.AF_UNIX and host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"tests are offline; refused connection to {address!r}")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
