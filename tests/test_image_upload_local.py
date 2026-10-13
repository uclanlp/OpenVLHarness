"""The local image provider stores query images on disk and the image host serves them."""
import asyncio
import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi", reason="the image host needs the server extra")
import httpx  # noqa: E402

from openvlharness.services import image_host  # noqa: E402
from openvlharness.services.tools import image_upload_utils as up  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 64


class LocalProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"IMAGE_UPLOAD_PROVIDER": "local",
                                           "IMAGE_UPLOAD_LOCAL_DIR": self.tmp.name,
                                           "IMAGE_UPLOAD_PUBLIC_BASE_URL": "https://img.example.test/"})
        self.env.start()
        up._upload_cache.clear()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_writes_file_and_returns_public_url(self):
        url = up.upload_image_bytes(JPEG, "image/jpeg", ".jpg")
        name = hashlib.sha256(JPEG).hexdigest() + ".jpg"
        self.assertEqual(url, f"https://img.example.test/{name}")
        with open(os.path.join(self.tmp.name, name), "rb") as f:
            self.assertEqual(f.read(), JPEG)
        self.assertEqual(up.upload_image_bytes(JPEG, "image/jpeg", ".jpg"), url)
        self.assertEqual(len(os.listdir(self.tmp.name)), 1)
        with patch.dict(os.environ, {"IMAGE_UPLOAD_PUBLIC_BASE_URL": "https://moved.example.test"}):
            self.assertEqual(up.upload_image_bytes(JPEG, "image/jpeg", ".jpg"), f"https://moved.example.test/{name}")

    def test_missing_public_base_url_is_an_error(self):
        with patch.dict(os.environ, {"IMAGE_UPLOAD_PUBLIC_BASE_URL": ""}):
            with self.assertRaises(RuntimeError):
                up._upload_local(JPEG, "image/jpeg", ".jpg", "0" * 64)

    def test_image_host_serves_stored_images_only(self):
        url = up.upload_image_bytes(JPEG, "image/jpeg", ".jpg")
        name = url.rsplit("/", 1)[1]
        app = image_host.create_app(self.tmp.name)

        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://host.invalid") as client:
                ok = await client.get(f"/{name}")
                self.assertEqual(ok.status_code, 200)
                self.assertEqual(ok.content, JPEG)
                self.assertTrue(ok.headers["content-type"].startswith("image/jpeg"))
                for bad in ("../etc/passwd", "notahash.jpg", "0" * 64 + ".jpg"):
                    self.assertEqual((await client.get(f"/{bad}")).status_code, 404)
                self.assertEqual((await client.get("/health")).json()["status"], "ok")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
