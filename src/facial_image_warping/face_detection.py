"""Face localization and cropping utilities."""

from pathlib import Path
import ctypes

import cv2
import numpy as np


FACES_OUTPUT_DIR = Path("outputs/faces")


def _to_windows_short_path(path: Path) -> str:
    """Convert a path to its Windows short form for OpenCV file APIs.

    Some OpenCV file-loading paths on Windows still fail when Unicode
    characters are present. The short 8.3 path often avoids that issue.
    If conversion fails, the original path string is returned.
    """
    path_str = str(path)
    buffer_size = 4096
    buffer = ctypes.create_unicode_buffer(buffer_size)
    result = ctypes.windll.kernel32.GetShortPathNameW(path_str, buffer, buffer_size)
    if result == 0 or result > buffer_size:
        return path_str
    return buffer.value


def load_haar_cascade() -> cv2.CascadeClassifier:
    """Load the default OpenCV frontal-face Haar cascade classifier."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(_to_windows_short_path(cascade_path))
    if classifier.empty():
        raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
    return classifier


def ensure_bgr_image(image: dict) -> dict:
    """Convert grayscale or RGB image payloads into BGR for face detection."""
    pixels = image["pixels"]
    color_space = image.get("color_space", "BGR")

    if color_space == "BGR":
        bgr_pixels = pixels
    elif color_space == "RGB":
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    elif color_space == "GRAYSCALE":
        gray_pixels = pixels
        if gray_pixels.dtype.kind == "f":
            gray_pixels = np.clip(gray_pixels * 255.0, 0, 255).astype(np.uint8)
        bgr_pixels = cv2.cvtColor(gray_pixels, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported color space for face detection: {color_space}")

    if bgr_pixels.dtype.kind == "f":
        bgr_pixels = np.clip(bgr_pixels * 255.0, 0, 255).astype(np.uint8)

    return {
        **image,
        "pixels": bgr_pixels,
        "shape": bgr_pixels.shape,
        "color_space": "BGR",
        "dtype": str(bgr_pixels.dtype),
    }


def convert_to_detection_grayscale(image: dict) -> np.ndarray:
    """Prepare an 8-bit grayscale matrix for Haar-based face detection."""
    bgr_image = ensure_bgr_image(image)
    return cv2.cvtColor(bgr_image["pixels"], cv2.COLOR_BGR2GRAY)


def select_largest_face(detections: np.ndarray) -> tuple[int, int, int, int]:
    """Select the largest detected face region by area."""
    if len(detections) == 0:
        raise ValueError("No face detections were provided.")
    x, y, w, h = max(detections, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(w), int(h)


def draw_face_bounding_box(image: dict, bounding_box: tuple[int, int, int, int]) -> dict:
    """Draw a face bounding box preview on top of the input image."""
    x, y, w, h = bounding_box
    preview = ensure_bgr_image(image)["pixels"].copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return {
        **image,
        "pixels": preview,
        "shape": preview.shape,
        "color_space": "BGR",
        "dtype": str(preview.dtype),
    }


def crop_face_region(image: dict, bounding_box: tuple[int, int, int, int]) -> dict:
    """Crop the face region of interest using the detected bounding box."""
    x, y, w, h = bounding_box
    bgr_image = ensure_bgr_image(image)["pixels"]
    face_crop = bgr_image[y : y + h, x : x + w].copy()
    if face_crop.size == 0:
        raise ValueError("Detected face bounding box produced an empty crop.")
    return {
        "path": image.get("path"),
        "file_name": image.get("file_name", "face.png"),
        "format": image.get("format", "png"),
        "width": face_crop.shape[1],
        "height": face_crop.shape[0],
        "shape": face_crop.shape,
        "pixels": face_crop,
        "color_space": "BGR",
        "dtype": str(face_crop.dtype),
        "crop_applied": True,
    }


def resize_face_crop(face_image: dict, target_size: tuple[int, int] = (512, 512)) -> dict:
    """Normalize the cropped face to a standard resolution."""
    resized = cv2.resize(face_image["pixels"], target_size, interpolation=cv2.INTER_AREA)
    return {
        **face_image,
        "pixels": resized,
        "width": target_size[0],
        "height": target_size[1],
        "shape": resized.shape,
        "target_size": target_size,
        "normalized_face": True,
    }


def save_face_preview(image: dict, output_path: str | Path) -> Path:
    """Save a BGR face preview image to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoded, buffer = cv2.imencode(output_file.suffix or ".png", image["pixels"])
    if not encoded:
        raise ValueError(f"Failed to encode face preview for {output_file}")
    try:
        buffer.tofile(output_file)
    except OSError as exc:
        raise ValueError(f"Failed to save face preview to {output_file}") from exc
    if not output_file.exists():
        raise ValueError(f"Failed to save face preview to {output_file}")
    return output_file


def detect_face_region(
    image: dict,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    target_size: tuple[int, int] = (512, 512),
    save_outputs: bool = True,
) -> dict:
    """Detect, crop, normalize, and save the primary face region from an image.

    Parameters
    ----------
    image:
        Preprocessed or raw image payload.
    scale_factor:
        Pyramid scale factor used by Haar cascade detection.
    min_neighbors:
        Minimum neighbor count required to keep a detection.
    target_size:
        Output size for the normalized face crop.
    save_outputs:
        Whether to write face preview artifacts to ``outputs/faces``.

    Returns
    -------
    dict
        Face crop, bounding box, preview paths, and detection metadata.
    """
    grayscale = convert_to_detection_grayscale(image)
    classifier = load_haar_cascade()
    detections = classifier.detectMultiScale(
        grayscale,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
    )

    if len(detections) == 0:
        raise ValueError(
            "No face detected in the provided image. Use a clear frontal face image with visible eyes, nose, and mouth."
        )

    bounding_box = select_largest_face(detections)
    boxed_preview = draw_face_bounding_box(image, bounding_box)
    cropped_face = crop_face_region(image, bounding_box)
    normalized_face = resize_face_crop(cropped_face, target_size=target_size)

    stem = Path(image.get("file_name", "image.png")).stem
    boxed_preview_path = FACES_OUTPUT_DIR / f"{stem}_detected_face.png"
    cropped_face_path = FACES_OUTPUT_DIR / f"{stem}_face_crop.png"

    if save_outputs:
        save_face_preview(boxed_preview, boxed_preview_path)
        save_face_preview(normalized_face, cropped_face_path)

    x, y, w, h = bounding_box
    return {
        "face_image": normalized_face,
        "bounding_box": bounding_box,
        "face_coordinates": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        },
        "confidence_score": None,
        "detector": "opencv_haar_cascade",
        "preview_path": str(boxed_preview_path),
        "cropped_face_path": str(cropped_face_path),
        "detection_count": int(len(detections)),
    }
