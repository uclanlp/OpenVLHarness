"""Image transport encoding shared by memory and model input formatting."""
import base64
import mimetypes
import os

def local_image_to_data_url(image_path: str) -> str:
    """Encode a local file as a data URL without changing its bytes.

    Args:
        image_path: Path to the image file. Its suffix determines the MIME type;
            unrecognized suffixes use application/octet-stream.

    Returns:
        A MIME-prefixed base64 data URL.

    Raises:
        FileNotFoundError: The path does not identify a file.
        OSError: Reading the file fails.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    # check if file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

import io
from PIL import Image

def resize_image_by_factor(img, factor=1):
    """Return a resized Pillow image with width and height multiplied by factor."""
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img

def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    """Encode a Pillow image while applying configured size limits.

    Args:
        img: Pillow image. A positive target_size may shrink it in place.
        target_size: Maximum thumbnail edge, or a nonpositive value to skip it.
        fmt: Pillow output format; defaults to JPEG.

    Returns:
        Base64 text without a data-URL prefix. Palette/alpha images are converted
        to RGB. VLMEVAL_MIN_IMAGE_EDGE controls upscaling and
        VLMEVAL_MAX_IMAGE_SIZE limits encoded text length by iterative resizing.
    """
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    max_size = os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e9)
    min_edge = os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 1e2)
    max_size = int(max_size)
    min_edge = int(min_edge)

    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Half Pixels Per Resize, approximately
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    if factor < 1:
        new_w, new_h = image_new.size
        print(
            f'Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, '
            f'resize to {factor:.2f} of original size: ({new_w}, {new_h})'
        )

    return ret
