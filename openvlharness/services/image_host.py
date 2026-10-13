"""Serve locally stored query images so that Serper Lens can fetch them.

With ``IMAGE_UPLOAD_PROVIDER=local`` the search tools write query images to
IMAGE_UPLOAD_LOCAL_DIR instead of a cloud bucket. This read-only server
exposes that directory; publish it at a public address (for example through a
tunnel) and set IMAGE_UPLOAD_PUBLIC_BASE_URL to that address.

    python -m openvlharness.services.image_host --port 8111
"""
import argparse
import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .tools.image_upload_utils import local_upload_dir

# Content-hash file names only: no path separators, no dot segments.
_NAME = re.compile(r"^[0-9a-f]{64}\.[A-Za-z0-9]{1,8}$")


def create_app(directory: str) -> FastAPI:
    """Build a FastAPI app that serves content-hash-named files from ``directory``."""
    app = FastAPI(title="OpenVLHarness image host")

    @app.get("/health")
    async def health():
        """Report readiness."""
        return {"status": "ok"}

    root = os.path.realpath(directory)

    @app.get("/{name}")
    async def image(name: str):
        """Return one stored image; anything but a regular file in the directory is not found."""
        if not _NAME.match(name):
            raise HTTPException(status_code=404)
        path = os.path.join(root, name)
        # Symbolic links are refused so that a link planted in the directory
        # cannot publish a file from elsewhere on the machine.
        if os.path.islink(path) or not os.path.isfile(path):
            raise HTTPException(status_code=404)
        if os.path.dirname(os.path.realpath(path)) != root:
            raise HTTPException(status_code=404)
        return FileResponse(path)

    return app


def main():
    """Serve the upload directory with Uvicorn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=None, help="directory to serve; defaults to IMAGE_UPLOAD_LOCAL_DIR")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8111)
    args = parser.parse_args()
    directory = args.dir or local_upload_dir()
    os.makedirs(directory, mode=0o700, exist_ok=True)
    import uvicorn
    uvicorn.run(create_app(directory), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
