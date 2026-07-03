"""Tests for Sprint 1-10 image processing pipeline stages."""

from pathlib import Path
import json

import numpy as np
from PIL import Image

from app import (
    run_aging_pipeline,
    run_analysis_pipeline,
    run_deaging_pipeline,
    run_reference_expression_transfer_pipeline,
    run_expression_warp_pipeline,
    run_face_detection_pipeline,
    run_frequency_analysis_pipeline,
    run_landmark_pipeline,
    run_preprocessing_pipeline,
    run_realtime_frame_pipeline,
    prepare_reference_expression_payload,
)
from facial_image_warping.aging_filter import apply_aging_filter, apply_deaging_filter
from facial_image_warping.expression_transfer import (
    apply_reference_expression_transfer,
    create_expression_transfer_targets,
)
from facial_image_warping.evaluation import (
    compute_mse,
    compute_psnr,
    compute_ssim,
    create_image_difference_visualization,
    evaluate_transformation,
    export_evaluation_to_csv,
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


def test_apply_aging_filter_creates_output_and_explanation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = apply_aging_filter(_make_face_image(), intensity=0.8)
    assert result["mode"] == "aging"
    assert Path(result["output_path"]).exists()
    assert result["filter_explanation"]


def test_apply_deaging_filter_creates_output_and_explanation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = apply_deaging_filter(_make_face_image(), intensity=0.6)
    assert result["mode"] == "deaging"
    assert Path(result["output_path"]).exists()
    assert result["filter_explanation"]


def test_aging_and_deaging_pipelines_run_on_sample_image(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "aging_face.png")
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.array([[1, 1, 10, 10]])))
    aging_result = run_aging_pipeline(str(image_path), intensity=0.7)
    deaging_result = run_deaging_pipeline(str(image_path), intensity=0.4)
    assert Path(aging_result["output_path"]).exists()
    assert Path(deaging_result["output_path"]).exists()


def test_compute_mse_returns_expected_value() -> None:
    original = {"pixels": np.zeros((4, 4, 3), dtype=np.uint8), "color_space": "BGR", "file_name": "orig.png"}
    transformed = {"pixels": np.full((4, 4, 3), 10, dtype=np.uint8), "color_space": "BGR", "file_name": "trans.png"}
    assert compute_mse(original, transformed) == 100.0


def test_compute_psnr_returns_infinite_for_identical_images() -> None:
    original = {"pixels": np.zeros((4, 4, 3), dtype=np.uint8), "color_space": "BGR", "file_name": "orig.png"}
    transformed = {"pixels": np.zeros((4, 4, 3), dtype=np.uint8), "color_space": "BGR", "file_name": "trans.png"}
    assert compute_psnr(original, transformed) == float("inf")


def test_compute_ssim_returns_one_for_identical_images() -> None:
    original = {"pixels": np.zeros((12, 12, 3), dtype=np.uint8), "color_space": "BGR", "file_name": "orig.png"}
    transformed = {"pixels": np.zeros((12, 12, 3), dtype=np.uint8), "color_space": "BGR", "file_name": "trans.png"}
    assert compute_ssim(original, transformed) == 1.0


def test_evaluation_artifacts_are_exported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    original = _make_face_image()
    transformed = {**_make_face_image(), "pixels": np.clip(_make_face_image()["pixels"] + 20, 0, 255).astype(np.uint8)}
    results = evaluate_transformation(original, transformed, save_outputs=True)
    difference_path = create_image_difference_visualization(original, transformed, tmp_path / "difference.png")
    csv_path = export_evaluation_to_csv(results, tmp_path / "evaluation.csv")
    assert Path(results["difference_path"]).exists()
    assert Path(results["csv_path"]).exists()
    assert difference_path.exists()
    assert csv_path.exists()



def test_create_expression_transfer_targets_blends_selected_regions() -> None:
    source = _make_dense_landmarks()
    reference = [dict(item) for item in source]
    reference[61]["x"] += 20
    reference[61]["y"] -= 10
    targets = create_expression_transfer_targets(source, reference, blend_factor=0.5, regions=["lips"])
    assert targets[61]["x"] == source[61]["x"] + 10
    assert targets[61]["y"] == source[61]["y"] - 5


def test_apply_reference_expression_transfer_creates_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    face_image = _make_face_image()
    source_landmarks = _make_dense_landmarks()
    reference_landmarks = [dict(item) for item in source_landmarks]
    for index in [61, 291, 70, 300]:
        reference_landmarks[index]["x"] += 8
        reference_landmarks[index]["y"] -= 4
    result = apply_reference_expression_transfer(
        face_image,
        source_landmarks,
        reference_landmarks,
        blend_factor=0.7,
        regions=["lips", "eyebrows"],
        save_outputs=True,
    )
    assert result["operation"] == "reference_expression_transfer"
    assert Path(result["warped_image_path"]).exists()
    assert Path(result["comparison_image_path"]).exists()


def test_prepare_reference_expression_payload_returns_face_and_landmarks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    reference_path = _create_sample_image(tmp_path, "reference_payload.png")
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.array([[1, 1, 10, 10]])))
    monkeypatch.setattr("facial_image_warping.landmark_detection.detect_face_mesh", lambda image, **kwargs: _FakeFaceMeshResults())
    payload = prepare_reference_expression_payload(str(reference_path), show_landmarks=True, selected_regions=["eyes", "lips"])
    assert payload["face_detection"]["face_image"]["pixels"].shape == (512, 512, 3)
    assert payload["landmarks"]["landmark_count"] == 468


def test_run_realtime_frame_pipeline_returns_live_preview(monkeypatch) -> None:
    frame = np.zeros((180, 220, 3), dtype=np.uint8)
    frame[40:150, 60:170] = 160
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.array([[60, 40, 110, 110]])))
    monkeypatch.setattr("facial_image_warping.landmark_detection.detect_face_mesh", lambda image, **kwargs: _FakeFaceMeshResults())
    result = run_realtime_frame_pipeline(
        frame,
        transformation="aging",
        intensity=0.5,
        show_landmarks=True,
        selected_regions=["eyes", "nose", "lips"],
    )
    assert result["original_frame"].shape[0] == frame.shape[0]
    assert result["transformed_frame"].shape[1] == frame.shape[1]
    assert result["composite_frame"].shape[1] == frame.shape[1] * 2


def test_reference_expression_transfer_pipeline_runs_with_mocked_dependencies(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_path = _create_sample_image(tmp_path, "source_face.png")
    reference_path = _create_sample_image(tmp_path, "reference_face.png")
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.array([[1, 1, 10, 10]])))
    monkeypatch.setattr("facial_image_warping.landmark_detection.detect_face_mesh", lambda image, **kwargs: _FakeFaceMeshResults())
    result = run_reference_expression_transfer_pipeline(
        str(source_path),
        str(reference_path),
        blend_factor=0.65,
        show_landmarks=True,
        selected_regions=["eyes", "eyebrows", "lips"],
    )
    assert result["transformation"]["operation"] == "reference_expression_transfer"
    assert result["metrics"]["mse"] >= 0
    assert result["source_landmarks"] is not None
    assert result["reference_landmarks"] is not None

def test_run_analysis_pipeline_returns_metrics_and_frequency(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = _create_sample_image(tmp_path, "analysis_face.png")
    monkeypatch.setattr("facial_image_warping.face_detection.load_haar_cascade", lambda: _FakeCascadeClassifier(np.array([[1, 1, 10, 10]])))
    monkeypatch.setattr("facial_image_warping.landmark_detection.detect_face_mesh", lambda image, **kwargs: _FakeFaceMeshResults())
    result = run_analysis_pipeline(str(image_path), transformation="smile_enhancement", intensity=0.5, show_landmarks=True)
    assert result["metrics"]["mse"] >= 0
    assert "original_high_low_ratio" in result["metrics"]
    assert result["landmarks"] is not None
