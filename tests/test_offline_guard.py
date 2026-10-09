"""The autouse fixture in conftest.py refuses connections to non-loopback hosts."""
import pytest
import requests


def test_non_loopback_connection_is_refused():
    with pytest.raises(requests.exceptions.ConnectionError, match="tests are offline"):
        requests.get("http://192.0.2.1:9/", timeout=1)
