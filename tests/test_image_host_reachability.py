"""A dead image host must surface as an error, not as images without matches.

Serper Lens answers HTTP 200 with no results both when an image has no matches
and when it cannot fetch the image. The router therefore fetches each published
image through its public URL itself: once per base URL before the first search,
and again whenever Lens returns nothing.
"""
import hashlib
import http.server
import os
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from openvlharness.services import degradation
from openvlharness.services.tools import image_upload_utils as up
import openvlharness.services.tools.serper_image_search.tool as ist

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 64
HASH = hashlib.sha256(JPEG).hexdigest()


def _serve(directory):
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=directory, **k)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.RequestHandlerClass.log_message = lambda *a, **k: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _closed_port_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


class PublicUrlProblemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server, self.base = _serve(self.tmp.name)

    def tearDown(self):
        self.server.shutdown()
        self.tmp.cleanup()

    def test_served_image_has_no_problem(self):
        with open(os.path.join(self.tmp.name, f"{HASH}.jpg"), "wb") as f:
            f.write(JPEG)
        self.assertIsNone(up.public_url_problem(f"{self.base}/{HASH}.jpg", HASH))

    def test_missing_file_wrong_body_and_dead_host_are_reported(self):
        self.assertEqual(up.public_url_problem(f"{self.base}/{HASH}.jpg", HASH), "HTTP 404")
        with open(os.path.join(self.tmp.name, f"{HASH}.jpg"), "wb") as f:
            f.write(b"<html>tunnel error page</html>")
        self.assertIn("not the uploaded image", up.public_url_problem(f"{self.base}/{HASH}.jpg", HASH))
        self.assertIn("ConnectionError", up.public_url_problem(f"{_closed_port_url()}/{HASH}.jpg", HASH, timeout=2))

    def test_content_hash_is_read_from_the_url(self):
        self.assertEqual(up.content_hash_from_url(f"https://h.example/{HASH}.jpg?X-Amz-Signature=1"), HASH)
        self.assertIsNone(up.content_hash_from_url("https://example.com/uploaded.jpg"))


class LocalUploadVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server, self.base = _serve(self.tmp.name)
        self.env = patch.dict(os.environ, {"IMAGE_UPLOAD_PROVIDER": "local",
                                           "IMAGE_UPLOAD_LOCAL_DIR": self.tmp.name,
                                           "IMAGE_UPLOAD_PUBLIC_BASE_URL": self.base})
        self.env.start()
        self.private = patch.object(up, "_is_private_host", lambda host: False)  # the test host is loopback
        self.private.start()
        up._verified_base_urls.clear()
        degradation.reset()

    def tearDown(self):
        self.private.stop()
        self.env.stop()
        self.server.shutdown()
        self.tmp.cleanup()

    def test_first_upload_is_fetched_back_once_per_base_url(self):
        calls = []
        real = up.public_url_problem
        with patch.object(up, "public_url_problem", lambda *a, **k: calls.append(a) or real(*a, **k)):
            url = up.upload_image_bytes(JPEG, "image/jpeg", ".jpg")
            self.assertEqual(url, f"{self.base}/{HASH}.jpg")
            up.upload_image_bytes(JPEG + b"y", "image/jpeg", ".jpg")
        self.assertEqual(len(calls), 1)
        self.assertEqual(degradation.snapshot()["counts"], {})

    def test_dead_host_raises_counts_and_recovers(self):
        with patch.dict(os.environ, {"IMAGE_UPLOAD_PUBLIC_BASE_URL": _closed_port_url()}):
            with patch.object(up, "_VERIFY_TIMEOUT_SEC", 2.0):
                with self.assertRaises(up.ImageHostUnreachableError):
                    up.upload_image_bytes(JPEG, "image/jpeg", ".jpg")
        self.assertEqual(degradation.snapshot()["counts"]["image_host:unreachable"], 1)
        self.assertEqual(up._verified_base_urls, set())
        # The image was written; once the host answers, the next upload verifies and succeeds.
        self.assertEqual(up.upload_image_bytes(JPEG, "image/jpeg", ".jpg"), f"{self.base}/{HASH}.jpg")
        self.assertEqual(up._verified_base_urls, {self.base})

    def test_private_base_url_is_rejected(self):
        self.private.stop()
        try:
            for base in ("http://localhost:8111", "http://127.0.0.1:8111", "http://10.0.0.5:8111"):
                with patch.dict(os.environ, {"IMAGE_UPLOAD_PUBLIC_BASE_URL": base}):
                    with self.assertRaisesRegex(RuntimeError, "private address"):
                        up.upload_image_bytes(JPEG, "image/jpeg", ".jpg")
        finally:
            self.private.start()


class _EmptyLens:
    def __init__(self):
        self.calls = 0

    def lens(self, url):
        self.calls += 1
        return {"organic": []}


class ImageSearchToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server, self.base = _serve(self.tmp.name)
        degradation.reset()
        up._verified_base_urls.clear()
        ist.LENS_EMPTY_MAX_ATTEMPTS = 2
        ist.LENS_EMPTY_BACKOFF_BASE_SEC = 0.0
        ist.LENS_EMPTY_BACKOFF_MAX_SEC = 0.0

    def tearDown(self):
        self.server.shutdown()
        self.tmp.cleanup()

    def _tool(self, url):
        tool = ist.Image_Search_Tool(llm_engine=None)
        tool._llm_summarize = False
        tool._multimodal_summarize = False
        tool._resolve_image_url = lambda b64: url
        client = _EmptyLens()
        tool._get_client = lambda: client
        return tool, client

    @staticmethod
    def _text(out):
        return out["text"] if isinstance(out, dict) else out

    def test_empty_lens_with_an_unreachable_image_is_an_error_not_no_matches(self):
        tool, client = self._tool(f"{self.base}/{HASH}.jpg")  # nothing is served at that name
        up._verified_base_urls.add(self.base)
        out = self._text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
        self.assertIn("QUERY IMAGE NOT REACHABLE", out)
        self.assertNotIn("NO VISUAL MATCHES", out)
        self.assertIn("HTTP 404", out)
        self.assertEqual(client.calls, 2)
        self.assertEqual(degradation.snapshot()["counts"]["image_host:unreachable"], 1)
        self.assertEqual(up._verified_base_urls, set(), "the base URL must be re-verified on the next upload")

    def test_empty_lens_with_a_served_image_is_a_genuine_no_match(self):
        with open(os.path.join(self.tmp.name, f"{HASH}.jpg"), "wb") as f:
            f.write(JPEG)
        tool, client = self._tool(f"{self.base}/{HASH}.jpg")
        out = self._text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
        self.assertIn("NO VISUAL MATCHES", out)
        self.assertEqual(degradation.snapshot()["counts"], {})

    def test_unreachable_host_at_upload_time_is_reported_to_the_agent(self):
        tool, client = self._tool(None)
        tool._resolve_image_url = lambda b64: (_ for _ in ()).throw(up.ImageHostUnreachableError("HTTP 502"))
        out = self._text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
        self.assertIn("QUERY IMAGE NOT REACHABLE", out)
        self.assertIn("HTTP 502", out)
        self.assertEqual(client.calls, 0)

    def test_urls_without_a_content_hash_are_not_checked(self):
        tool, client = self._tool("https://example.com/uploaded.jpg")
        out = self._text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
        self.assertIn("NO VISUAL MATCHES", out)


if __name__ == "__main__":
    unittest.main()
