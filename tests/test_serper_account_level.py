"""An account-level Serper failure (credits, key, payment) must not look like a bad query.

Serper answers such failures with a 4xx whose body names the account. They are
told apart from ordinary request errors, rate limits and relayed site errors,
counted under ``serper_account`` in the degradation counters, and rendered to
the agent as "search unavailable, do not retry" instead of retry hints.
"""
import unittest
from unittest.mock import patch

import requests

from openvlharness.services import degradation
from openvlharness.services.tools import serper_common as sc
import openvlharness.services.tools.serper_image_search.tool as ist
import openvlharness.services.tools.serper_text_search.tool as tst
import openvlharness.services.tools.serper_webpage_visit.tool as wvt


def _http_error(status, body):
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    return requests.HTTPError(f"{status} Client Error", response=response)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        sc._ACCOUNT_LEVEL_WARNED = False
        degradation.reset()

    def _call(self, exc):
        with patch("time.sleep"):
            with self.assertRaises(sc.SerperRequestError) as ctx:
                sc._call_serper_with_retry(endpoint_label="search",
                                           do_request=lambda: (_ for _ in ()).throw(exc))
        return ctx.exception

    def test_credit_key_and_payment_bodies_on_4xx_are_account_level(self):
        for status, body in ((400, '{"message":"Not enough credits","statusCode":400}'),
                             (401, '{"message":"Unauthorized"}'),
                             (403, '{"message":"Invalid API key"}'),
                             (402, '{"message":"Payment required"}')):
            with self.subTest(status=status):
                self.assertTrue(sc.is_account_level_failure(status, body))
                self.assertTrue(self._call(_http_error(status, body)).account_level)

    def test_rate_limits_relayed_site_errors_and_bad_requests_are_not(self):
        cases = ((429, '{"message":"Rate quota exceeded, slow down"}'),
                 (403, '{"message":"Forbidden"}'),
                 (400, '{"message":"query must not be empty"}'),
                 (500, '{"message":"payment required"}'),
                 (403, 'Failed to fetch https://site.example: 403 Forbidden'))
        for status, body in cases:
            with self.subTest(status=status, body=body):
                self.assertFalse(sc.is_account_level_failure(status, body))
                self.assertFalse(self._call(_http_error(status, body)).account_level)
        self.assertNotIn("serper_account", degradation.snapshot()["counts"])

    def test_every_account_level_call_is_counted_and_the_first_is_logged(self):
        with self.assertLogs(sc.logger, level="ERROR") as logs:
            for _ in range(3):
                self._call(_http_error(400, "Not enough credits"))
        self.assertEqual(sum("ACCOUNT-LEVEL" in line for line in logs.output), 1)
        counts = degradation.snapshot()["counts"]
        self.assertEqual(counts["serper_account"], 3)
        self.assertEqual(counts["serper_account:http_400"], 3)
        self.assertEqual(degradation.snapshot()["last"]["serper_account"]["tool"], "search")


class AgentTextTests(unittest.TestCase):
    ERR = sc.SerperRequestError("Serper search failed after 1 attempt(s): HTTP 400 from search",
                                status_code=400, transient=False, endpoint="search",
                                provider_body="Not enough credits", account_level=True)

    def test_text_search_says_unavailable_and_offers_no_rewrites(self):
        out = tst.Text_Search_Tool._format_serper_error("who painted this", "general", self.ERR)
        self.assertIn("SEARCH UNAVAILABLE", out)
        self.assertIn("do NOT retry", out)
        for retry_hint in ("Query rewrites", "search_type=<one of", "Retry the SAME"):
            self.assertNotIn(retry_hint, out)
        self.assertNotIn("Not enough credits", out)

    def test_image_search_says_unavailable_instead_of_crop_hints(self):
        out = ist.Image_Search_Tool._format_lens_error("input_image_1", self.ERR)
        self.assertIn("REVERSE IMAGE SEARCH UNAVAILABLE", out)
        self.assertNotIn("Zoom_In", out)
        self.assertNotIn("Not enough credits", out)

    def test_ordinary_errors_keep_their_recovery_menu(self):
        err = sc.SerperRequestError("Serper search failed after 1 attempt(s): HTTP 400 from search",
                                    status_code=400, transient=False, endpoint="search")
        self.assertIn("Query rewrites", tst.Text_Search_Tool._format_serper_error("q", "general", err))
        self.assertIn("Suggested next actions", ist.Image_Search_Tool._format_lens_error("input_image_1", err))

    def test_webpage_visit_stops_without_retrying(self):
        tool = wvt.Webpage_Visit_Tool.__new__(wvt.Webpage_Visit_Tool)
        calls = []

        def scrape(url):
            calls.append(url)
            raise self.ERR
        tool._fetch_page_serper = scrape
        with patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "search account"):
                tool._fetch_page("https://example.test/page")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
