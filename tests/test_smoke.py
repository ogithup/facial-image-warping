"""Tests for Sprint 1-5 image processing pipeline stages."""

from pathlib import Path
import json

import numpy as np
from PIL import Image

from app import (
    run_expression_warp_pipeline,
    run_face_detection_pipeline,
    run_frequency_analysis_pipeline,
    run_landmark_pipeline,
    run_preprocessing_pipeline,
)
from facial_image_warping.face_detection import crop_face_region, detect_face_region, resize_face_crop
from facial_image_warping.fourier_analysis import (
    analyze_frequency_content,
    calculate_frequency_energy,
    compute_fft,
    compute_magnitude_spectrum,
    export_frequency_analysis_to_csv,
    visualize_spectrum,
)
from facial_image_warping.geometric_warping import (
    apply_delaunay_triangulation,
    apply_expression_warp,
    create_target_landmarks,
    warp_triangle,
)
from facial_image_warping.landmark_detection import (
    detect_landmarks,
    export_landmarks_to_csv,
    export_landmarks_to_json,
    toggle_landmark_visualization,
)
from facial_image_warping.input_module import load_image
from facial_image_warping.preprocessing import convert_bgr_to_rgb, convert_to_grayscale, resize_to_standard


def _create_sample_image(tmp_path: Path, file_name: str = "sample.png") -> Path:
    sample = np.zeros((24, 32, 3), dtype=np.uint8)
    sample[..., 0] = 255
    sample[8:16, 8:16, 1] = 128
    image_path = tmp_path / file_name
    Image.fromarray(sample, mode="RGB").save(image_path)
    return image_path


def test_load_image_supports_png(tmp_path: Path) -> None:
    image_path = _create_sample_image(tmp_path)
    image = load_image(image_path)
    assert image["format"] == "png"
    assert image["color_space"] == "BGR"
    assert image["pixels"].shape == (24, 32, 3)


def test_resize_to_standard_returns_requested_shape(tmp_path: Path) -> None:
    image_path = _create_sample_image(tmp_path)
    loaded = load_image(image_path)
    rgb_image = convert_bgr_to_rgb(loaded)
    resized = resize_to_standard(rgb_image, target_size=(512, 512))
    assert resized["pixels"].shape == (512, 512, 3)
    assert resized["width"] == 512
    assert resized["height"] == 512


def test_convert_to_grayscale_returns_single_channel(tmp_path: Path) -> None:
    image_path = _create_sample_image(tmp_path)
    loaded = load_image(image_path)
    rgb_image = convert_bgr_to_rgb(loaded)
    grayscale = convert_to_grayscale(rgb_image)
    assert grayscale["color_space"] == "GRAYSCALE"
    assert grayscale["pixels"].ndim == 2


def test_preprocessing_pipeline_creates_output_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "face.png")
    result = run_preprocessing_pipeline(str(image_path))
    assert result["status"] == "preprocessed"
    assert Path(result["histogram_path"]).exists()
    assert Path(result["processed_image_path"]).exists()


class _FakeCascadeClassifier:
    def __init__(self, detections: np.ndarray) -> None:
        self._detections = detections

    def detectMultiScale(self, grayscale, scaleFactor=1.1, minNeighbors=5):  # noqa: N802
        return self._detections


def test_detect_face_region_returns_bbox_and_saves_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "portrait.png")
    image = load_image(image_path)
    fake_detections = np.array([[4, 5, 12, 10]])
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(fake_detections))
    result = detect_face_region(image, save_outputs=True)
    assert result["detector"] == "opencv_haar_cascade"
    assert result["bounding_box"] == (4, 5, 12, 10)
    assert result["face_image"]["pixels"].shape == (512, 512, 3)
    assert Path(result["preview_path"]).exists()
    assert Path(result["cropped_face_path"]).exists()


def test_detect_face_region_raises_clear_error_when_missing(monkeypatch, tmp_path: Path) -> None:
    image_path = _create_sample_image(tmp_path, "empty.png")
    image = load_image(image_path)
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.empty((0, 4), dtype=np.int32)))
    try:
        detect_face_region(image, save_outputs=False)
    except ValueError as exc:
        assert "No face detected" in str(exc)
    else:
        raise AssertionError("Expected face detection to fail when no face is present.")


def test_crop_and_resize_face_region() -> None:
    pixels = np.zeros((20, 20, 3), dtype=np.uint8)
    pixels[2:10, 3:11] = 255
    image = {"pixels": pixels, "color_space": "BGR", "file_name": "synthetic.png", "format": "png"}
    cropped = crop_face_region(image, (3, 2, 8, 8))
    resized = resize_face_crop(cropped, target_size=(64, 64))
    assert cropped["pixels"].shape == (8, 8, 3)
    assert resized["pixels"].shape == (64, 64, 3)


def test_face_detection_pipeline_uses_preprocessing_and_detection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "face.png")
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.array([[1, 1, 10, 10]])))
    result = run_face_detection_pipeline(str(image_path))
    assert result["face_image"]["pixels"].shape == (512, 512, 3)
    assert result["face_coordinates"]["width"] == 10


class _FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeFaceLandmarks:
    def __init__(self, count: int = 468) -> None:
        self.landmark = [_FakeLandmark((index % 20) / 20.0, (index % 15) / 15.0, index / count) for index in range(count)]


class _FakeFaceMeshResults:
    def __init__(self, count: int = 468) -> None:
        self.multi_face_landmarks = [_FakeFaceLandmarks(count)]


def _make_face_image() -> dict:
    pixels = np.zeros((128, 128, 3), dtype=np.uint8)
    pixels[20:100, 20:100] = 180
    return {"pixels": pixels, "color_space": "BGR", "file_name": "face.png", "format": "png"}


def _make_dense_landmarks() -> list[dict]:
    landmarks = []
    for index in range(468):
        landmarks.append({"index": index, "x": 10 + (index % 18) * 6, "y": 10 + ((index // 18) % 26) * 4, "z": 0.0, "normalized_x": 0.1, "normalized_y": 0.1})
    return landmarks


def test_detect_landmarks_exports_visualization_and_coordinate_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("facial_image_warping.landmark_detection.detect_face_mesh", lambda image, **kwargs: _FakeFaceMeshResults())
    result = detect_landmarks(_make_face_image(), selected_regions=["eyes", "lips"])
    assert result["model"] == "mediapipe_face_mesh"
    assert result["landmark_count"] == 468
    assert Path(result["visualization_path"]).exists()
    assert Path(result["json_path"]).exists()
    assert Path(result["csv_path"]).exists()


def test_export_landmarks_json_and_csv(tmp_path: Path) -> None:
    landmarks = [
        {"index": 0, "x": 10, "y": 20, "z": 0.1, "normalized_x": 0.1, "normalized_y": 0.2},
        {"index": 1, "x": 30, "y": 40, "z": 0.2, "normalized_x": 0.3, "normalized_y": 0.4},
    ]
    json_path = export_landmarks_to_json(landmarks, tmp_path / "landmarks.json")
    csv_path = export_landmarks_to_csv(landmarks, tmp_path / "landmarks.csv")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["index"] == 0
    assert "normalized_x" in csv_path.read_text(encoding="utf-8")


def test_toggle_landmark_visualization_supports_selected_regions() -> None:
    image = _make_face_image()
    landmarks = [{"index": index, "x": 10 + (index % 10), "y": 10 + (index % 10), "z": 0.0, "normalized_x": 0.1, "normalized_y": 0.1} for index in range(468)]
    visualization = toggle_landmark_visualization(image, landmarks, show_full_mesh=False, selected_regions=["eyes", "nose"])
    assert visualization["color_space"] == "RGB"
    assert visualization["pixels"].shape == (128, 128, 3)


def test_run_landmark_pipeline_uses_face_detection_and_landmark_steps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "landmark_face.png")
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.array([[1, 1, 20, 20]])))
    monkeypatch.setattr("facial_image_warping.landmark_detection.detect_face_mesh", lambda image, **kwargs: _FakeFaceMeshResults())
    result = run_landmark_pipeline(str(image_path), show_full_mesh=True, selected_regions=["lips"])
    assert result["landmark_count"] == 468
    assert result["selected_regions"] == ["lips"]


def test_create_target_landmarks_changes_expression_points() -> None:
    landmarks = _make_dense_landmarks()
    targets = create_target_landmarks(landmarks, transformation="smile_enhancement", intensity=1.0)
    assert targets[61]["x"] < landmarks[61]["x"]
    assert targets[291]["x"] > landmarks[291]["x"]


def test_apply_delaunay_triangulation_returns_triangle_indices() -> None:
    landmarks = _make_dense_landmarks()[:80]
    triangles = apply_delaunay_triangulation((128, 128, 3), landmarks)
    assert triangles
    assert all(len(triangle) == 3 for triangle in triangles)


def test_warp_triangle_modifies_destination_region() -> None:
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    source[5:20, 5:20] = (255, 255, 255)
    destination = np.zeros((32, 32, 3), dtype=np.float32)
    warp_triangle(source, destination, [(5, 5), (20, 5), (5, 20)], [(8, 8), (24, 7), (7, 24)])
    assert destination.sum() > 0


def test_apply_expression_warp_creates_output_images(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    face_image = _make_face_image()
    landmarks = _make_dense_landmarks()
    result = apply_expression_warp(face_image, landmarks, transformation="eyebrow_raising", intensity=0.7, save_outputs=True)
    assert result["operation"] == "eyebrow_raising"
    assert Path(result["warped_image_path"]).exists()
    assert Path(result["comparison_image_path"]).exists()


def test_expression_warp_pipeline_runs_with_mocked_face_and_landmarks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "warp_face.png")
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.array([[1, 1, 20, 20]])))
    monkeypatch.setattr("facial_image_warping.landmark_detection.detect_face_mesh", lambda image, **kwargs: _FakeFaceMeshResults())
    result = run_expression_warp_pipeline(str(image_path), transformation="lip_widening", intensity=0.6)
    assert result["operation"] == "lip_widening"
    assert result["image"]["pixels"].shape[2] == 3


def test_compute_fft_returns_centered_frequency_grid() -> None:
    fft_result = compute_fft(_make_face_image())
    assert fft_result.shape == (128, 128)
    assert np.iscomplexobj(fft_result)


def test_compute_magnitude_spectrum_returns_non_negative_values() -> None:
    spectrum = compute_magnitude_spectrum(compute_fft(_make_face_image()))
    assert spectrum.shape == (128, 128)
    assert np.all(spectrum >= 0)


def test_calculate_frequency_energy_returns_consistent_metrics() -> None:
    metrics = calculate_frequency_energy(compute_fft(_make_face_image()))
    assert metrics["total_energy"] >= metrics["low_frequency_energy"]
    assert metrics["total_energy"] >= metrics["high_frequency_energy"]
    assert metrics["low_frequency_radius"] > 0


def test_visualize_spectrum_and_csv_export_create_files(tmp_path: Path) -> None:
    image = _make_face_image()
    fft_result = compute_fft(image)
    spectrum = compute_magnitude_spectrum(fft_result)
    metrics = calculate_frequency_energy(fft_result)
    spectrum_path = visualize_spectrum(image, spectrum, tmp_path / "spectrum.png")
    csv_path = export_frequency_analysis_to_csv(metrics, tmp_path / "metrics.csv")
    assert spectrum_path.exists()
    assert csv_path.exists()


def test_analyze_frequency_content_creates_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = analyze_frequency_content(_make_face_image(), save_outputs=True)
    assert Path(result["spectrum_path"]).exists()
    assert Path(result["csv_path"]).exists()
    assert result["high_low_ratio"] >= 0


def test_frequency_analysis_pipeline_runs_on_sample_image(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "frequency_face.png")
    result = run_frequency_analysis_pipeline(str(image_path))
    assert Path(result["spectrum_path"]).exists()
    assert Path(result["csv_path"]).exists()
