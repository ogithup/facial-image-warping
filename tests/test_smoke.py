"""Tests for Sprint 1 preprocessing and Sprint 2 face detection."""

from pathlib import Path

import numpy as np
from PIL import Image

from app import run_face_detection_pipeline, run_preprocessing_pipeline
from facial_image_warping.face_detection import crop_face_region, detect_face_region, resize_face_crop
from facial_image_warping.input_module import load_image
from facial_image_warping.preprocessing import convert_bgr_to_rgb, convert_to_grayscale, resize_to_standard


def _create_sample_image(tmp_path: Path, file_name: str = "sample.png") -> Path:
    """Create a simple synthetic RGB image for deterministic tests."""
    sample = np.zeros((24, 32, 3), dtype=np.uint8)
    sample[..., 0] = 255
    sample[8:16, 8:16, 1] = 128
    image_path = tmp_path / file_name
    Image.fromarray(sample, mode="RGB").save(image_path)
    return image_path


def test_load_image_supports_png(tmp_path: Path) -> None:
    """Image loader should accept supported formats and return pixel data."""
    image_path = _create_sample_image(tmp_path)
    image = load_image(image_path)

    assert image["format"] == "png"
    assert image["color_space"] == "BGR"
    assert image["pixels"].shape == (24, 32, 3)


def test_resize_to_standard_returns_requested_shape(tmp_path: Path) -> None:
    """Resizing should produce the configured output resolution."""
    image_path = _create_sample_image(tmp_path)
    loaded = load_image(image_path)
    rgb_image = convert_bgr_to_rgb(loaded)
    resized = resize_to_standard(rgb_image, target_size=(512, 512))

    assert resized["pixels"].shape == (512, 512, 3)
    assert resized["width"] == 512
    assert resized["height"] == 512


def test_convert_to_grayscale_returns_single_channel(tmp_path: Path) -> None:
    """Grayscale conversion should return a two-dimensional image matrix."""
    image_path = _create_sample_image(tmp_path)
    loaded = load_image(image_path)
    rgb_image = convert_bgr_to_rgb(loaded)
    grayscale = convert_to_grayscale(rgb_image)

    assert grayscale["color_space"] == "GRAYSCALE"
    assert grayscale["pixels"].ndim == 2


def test_preprocessing_pipeline_creates_output_artifacts(tmp_path: Path, monkeypatch) -> None:
    """The Sprint 1 pipeline should save its processed image and histogram."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "face.png")

    result = run_preprocessing_pipeline(str(image_path))

    assert result["status"] == "preprocessed"
    assert Path(result["histogram_path"]).exists()
    assert Path(result["processed_image_path"]).exists()


class _FakeCascadeClassifier:
    """Small fake detector used to isolate face-detection tests."""

    def __init__(self, detections: np.ndarray) -> None:
        self._detections = detections

    def detectMultiScale(self, grayscale, scaleFactor=1.1, minNeighbors=5):  # noqa: N802
        return self._detections


def test_detect_face_region_returns_bbox_and_saves_outputs(tmp_path: Path, monkeypatch) -> None:
    """Face detection should crop, normalize, and save preview artifacts."""
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "portrait.png")
    image = load_image(image_path)

    fake_detections = np.array([[4, 5, 12, 10]])
    monkeypatch.setattr(
        "facial_image_warping.face_detection.load_haar_cascade",
        lambda: _FakeCascadeClassifier(fake_detections),
    )

    result = detect_face_region(image, save_outputs=True)

    assert result["detector"] == "opencv_haar_cascade"
    assert result["bounding_box"] == (4, 5, 12, 10)
    assert result["face_image"]["pixels"].shape == (512, 512, 3)
    assert Path(result["preview_path"]).exists()
    assert Path(result["cropped_face_path"]).exists()


def test_detect_face_region_raises_clear_error_when_missing(monkeypatch, tmp_path: Path) -> None:
    """No-face cases should raise a clear and explicit error message."""
    image_path = _create_sample_image(tmp_path, "empty.png")
    image = load_image(image_path)

    monkeypatch.setattr(
        "facial_image_warping.face_detection.load_haar_cascade",
        lambda: _FakeCascadeClassifier(np.empty((0, 4), dtype=np.int32)),
    )

    try:
        detect_face_region(image, save_outputs=False)
    except ValueError as exc:
        assert "No face detected" in str(exc)
    else:
        raise AssertionError("Expected face detection to fail when no face is present.")


def test_crop_and_resize_face_region() -> None:
    """Face ROI extraction should preserve the requested region and normalize size."""
    pixels = np.zeros((20, 20, 3), dtype=np.uint8)
    pixels[2:10, 3:11] = 255
    image = {
        "pixels": pixels,
        "color_space": "BGR",
        "file_name": "synthetic.png",
        "format": "png",
    }

    cropped = crop_face_region(image, (3, 2, 8, 8))
    resized = resize_face_crop(cropped, target_size=(64, 64))

    assert cropped["pixels"].shape == (8, 8, 3)
    assert resized["pixels"].shape == (64, 64, 3)


def test_face_detection_pipeline_uses_preprocessing_and_detection(tmp_path: Path, monkeypatch) -> None:
    """The Sprint 2 pipeline should run preprocessing before detection."""
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "face.png")

    monkeypatch.setattr(
        "facial_image_warping.face_detection.load_haar_cascade",
        lambda: _FakeCascadeClassifier(np.array([[1, 1, 10, 10]])),
    )

    result = run_face_detection_pipeline(str(image_path))

    assert result["face_image"]["pixels"].shape == (512, 512, 3)
    assert result["face_coordinates"]["width"] == 10
