"""Shared test setup: tests run offline; evaluator tests need the eval extra."""
import socket

import pytest

try:
    import pandas  # noqa: F401
except ImportError:  # pragma: no cover - exercised in minimal installs
    collect_ignore_glob = ["evaluation/*"]

CREDENTIALS = ("OPENAI_API_KEY", "OPENAI_API_BASE", "AZURE_OPENAI_API_KEY", "SERPER_API_KEY",
               "MOONSHOT_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Drop provider credentials and refuse non-loopback connections.

    Without this, code that builds a client from the environment would reach a
    real model, judge or search API whenever the shell exports a key.
    """
    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    connect = socket.socket.connect

    def loopback_only(sock, address):
        host = address[0] if isinstance(address, tuple) else None
        if sock.family != socket.AF_UNIX and host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"tests are offline; refused connection to {address!r}")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
