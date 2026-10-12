"""SerperRequestError carries the provider body for operators without changing agent-visible text."""
import unittest
from unittest.mock import patch

import requests

from openvlharness.services.tools import serper_common as sc


def _http_error(status, body):
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    return requests.HTTPError(f"{status} Client Error", response=response)


class ProviderBodyTests(unittest.TestCase):
    def setUp(self):
        sc._ACCOUNT_LEVEL_WARNED = False

    def _call(self, exc):
        with patch("time.sleep"):
            with self.assertRaises(sc.SerperRequestError) as ctx:
                sc._call_serper_with_retry(endpoint_label="search", do_request=lambda: (_ for _ in ()).throw(exc))
        return ctx.exception

    def test_account_level_body_is_carried_but_kept_out_of_the_message(self):
        err = self._call(_http_error(400, '{"message":"Not enough credits","statusCode":400}'))
        self.assertEqual(err.status_code, 400)
        self.assertIn("Not enough credits", err.provider_body)
        self.assertNotIn("credits", str(err))
        self.assertTrue(err.is_account_level)
        self.assertEqual(err.attempts, 1)

    def test_ordinary_client_error_is_not_account_level(self):
        err = self._call(_http_error(400, '{"message":"query must not be empty"}'))
        self.assertIn("query must not be empty", err.provider_body)
        self.assertFalse(err.is_account_level)

    def test_account_level_failure_is_logged_once_per_process(self):
        with self.assertLogs(sc.logger, level="ERROR") as logs:
            self._call(_http_error(400, "Not enough credits"))
            self._call(_http_error(400, "Not enough credits"))
        self.assertEqual(sum("ACCOUNT-LEVEL" in line for line in logs.output), 1)

    def test_transient_error_still_retries_and_keeps_last_body(self):
        err = self._call(_http_error(503, "upstream unavailable"))
        self.assertTrue(err.transient)
        self.assertEqual(err.attempts, sc._SERPER_MAX_RETRIES)
        self.assertEqual(err.provider_body, "upstream unavailable")

    def test_missing_response_body_is_empty_string(self):
        self.assertEqual(sc._http_error_body(requests.HTTPError("no response")), "")


if __name__ == "__main__":
    unittest.main()
