from mini_vlm_toolkit.utils import api_utils


def test_retry_with_exponential_backoff_retries_transient_429():
    state = {"calls": 0}

    def flaky():
        state["calls"] += 1
        if state["calls"] < 3:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    wrapped = api_utils.retry_with_exponential_backoff(
        flaky,
        max_retries=3,
        jitter=False,
        initial_delay=0.01,
        max_delay=0.02,
    )

    original_sleep = api_utils.time.sleep
    api_utils.time.sleep = lambda _x: None
    try:
        assert wrapped() == "ok"
    finally:
        api_utils.time.sleep = original_sleep

    assert state["calls"] == 3


def test_retry_with_exponential_backoff_does_not_retry_non_transient_400():
    state = {"calls": 0}

    def bad_request():
        state["calls"] += 1
        raise RuntimeError("400 Bad Request")

    wrapped = api_utils.retry_with_exponential_backoff(
        bad_request,
        max_retries=3,
        jitter=False,
        initial_delay=0.01,
        max_delay=0.02,
    )

    original_sleep = api_utils.time.sleep
    api_utils.time.sleep = lambda _x: None
    try:
        try:
            wrapped()
            raise AssertionError("Expected RuntimeError")
        except RuntimeError as exc:
            assert "400" in str(exc)
    finally:
        api_utils.time.sleep = original_sleep

    assert state["calls"] == 1


if __name__ == "__main__":
    test_retry_with_exponential_backoff_retries_transient_429()
    test_retry_with_exponential_backoff_does_not_retry_non_transient_400()
    print("PASS: test_retry_with_exponential_backoff_retries_transient_429")
    print("PASS: test_retry_with_exponential_backoff_does_not_retry_non_transient_400")
