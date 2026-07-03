"""Input acquisition utilities for facial image processing workflows."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_source(image_source: str | Path) -> Path:
    """Validate that the input file exists and uses a supported image format.

    Parameters
    ----------
    image_source:
        Filesystem path pointing to an image selected by the user.

    Returns
    -------
    Path
        Resolved path to the image file.
    """
    image_path = Path(image_source).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image format: {image_path.suffix}. Supported: {supported}")

    return image_path


def load_image(image_source: str | Path) -> dict:
    """Load an image with OpenCV and collect basic metadata using Pillow.

    Parameters
    ----------
    image_source:
        Filesystem path to a JPG or PNG image.

    Returns
    -------
    dict
        Image payload containing BGR and RGB pixel matrices with metadata.
    """
    image_path = validate_image_source(image_source)
    # Use imdecode on raw bytes so Windows Unicode paths work reliably.
    image_buffer = np.fromfile(image_path, dtype=np.uint8)
    bgr_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Failed to decode image: {image_path}")

    with Image.open(image_path) as pil_image:
        width, height = pil_image.size
        image_mode = pil_image.mode

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return {
        "path": str(image_path),
        "file_name": image_path.name,
        "format": image_path.suffix.lower().lstrip("."),
        "width": width,
        "height": height,
        "shape": bgr_image.shape,
        "pil_mode": image_mode,
        "pixels": bgr_image,
        "rgb_pixels": rgb_image,
        "color_space": "BGR",
        "dtype": str(bgr_image.dtype),
    }


def request_image_input(image_source: str | Path) -> dict:
    """Load a user-provided image and return an input stage payload.

    Parameters
    ----------
    image_source:
        Path to a user-selected facial image.

    Returns
    -------
    dict
        Input payload ready for preprocessing.
    """
    image = load_image(image_source)
    return {
        "source": str(image_source),
        "image": image,
    }
