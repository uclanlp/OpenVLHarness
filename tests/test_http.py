"""Retry classification uses HTTP metadata and transport exception types."""
import unittest
from unittest.mock import Mock, patch

import requests

from openvlharness.services.http import retry_with_exponential_backoff


class RetryClassificationTests(unittest.TestCase):
    def test_requests_http_errors_use_response_status_even_when_response_is_falsey(self):
        for status, attempts in ((503, 3), (400, 1), (401, 1), (429, 3)):
            with self.subTest(status=status):
                response = requests.Response()
                response.status_code = status
                operation = Mock(side_effect=requests.HTTPError(
                    "http://192.168.1.2 internal connection 503", response=response))
                with patch("openvlharness.services.http.time.sleep") as sleep:
                    with self.assertRaises(requests.HTTPError):
                        retry_with_exponential_backoff(operation, max_retries=2)()
                    self.assertEqual(operation.call_count, attempts)
                    self.assertEqual(sleep.call_count, attempts - 1)

    def test_transport_types_retry_but_unstructured_text_does_not(self):
        for error, attempts in ((requests.Timeout("opaque"), 2),
                                (requests.ConnectionError("opaque"), 2),
                                (RuntimeError("503 internal connection timeout"), 1)):
            with self.subTest(error=type(error).__name__):
                operation = Mock(side_effect=error)
                with patch("openvlharness.services.http.time.sleep"):
                    with self.assertRaises(type(error)):
                        retry_with_exponential_backoff(operation, max_retries=1)()
                self.assertEqual(operation.call_count, attempts)
