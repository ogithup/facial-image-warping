"""Application entry point for the facial image processing pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from facial_image_warping.input_module import request_image_input
from facial_image_warping.preprocessing import preprocess_image
from facial_image_warping.face_detection import detect_face_region
from facial_image_warping.landmark_detection import detect_landmarks, toggle_landmark_visualization
from facial_image_warping.geometric_warping import apply_expression_warp
from facial_image_warping.aging_filter import apply_aging_filter, apply_deaging_filter
from facial_image_warping.expression_transfer import (
    apply_reference_expression_transfer,
    extract_expression_coefficients,
    TRANSFER_METHODS,
)
from facial_image_warping.fourier_analysis import analyze_frequency_content
from facial_image_warping.evaluation import evaluate_transformation
from facial_image_warping.visualization import build_result_summary


EXPRESSION_TRANSFORMATIONS = {
    "smile_enhancement",
    "eyebrow_raising",
    "lip_widening",
    "face_slimming",
}
GUI_TRANSFORMATIONS = EXPRESSION_TRANSFORMATIONS | {"aging", "de-aging", "reference_expression_transfer"}
DEFAULT_LANDMARK_REGIONS = ["eyes", "lips", "nose"]
DEFAULT_TRANSFER_REGIONS = ["eyes", "eyebrows", "lips"]
DEFAULT_TRANSFER_METHOD = "expression_coefficients"
DEFAULT_REALTIME_PROFILE = {
    "profile_name": "balanced",
    "max_width": 640,
    "analysis_size": 320,
    "landmark_max_dimension": 384,
    "detection_interval": 2,
    "tracking_padding": 0.18,
    "frame_skip_interval": 2,
    "smoothing_alpha": 0.65,
    "reference_update_interval": 2,
}


def make_image_dict_from_bgr_frame(frame: np.ndarray, file_name: str = "frame.png") -> dict:
    """Wrap a raw BGR frame in the project's image dictionary format."""
    return {
        "file_name": file_name,
        "format": Path(file_name).suffix.lower().lstrip(".") or "png",
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "shape": frame.shape,
        "pixels": frame.copy(),
        "color_space": "BGR",
        "dtype": str(frame.dtype),
    }


def _merge_realtime_profile(profile: dict | None) -> dict:
    resolved = dict(DEFAULT_REALTIME_PROFILE)
    if profile:
        resolved.update({key: value for key, value in profile.items() if value is not None})
    return resolved


def _bbox_is_valid(frame_shape: tuple[int, ...], bbox: tuple[int, int, int, int] | None) -> bool:
    if bbox is None:
        return False
    height, width = frame_shape[:2]
    x, y, box_width, box_height = bbox
    return (
        box_width > 0
        and box_height > 0
        and x >= 0
        and y >= 0
        and x + box_width <= width
        and y + box_height <= height
    )


def _expand_bbox(frame_shape: tuple[int, ...], bbox: tuple[int, int, int, int], padding_ratio: float) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x, y, box_width, box_height = bbox
    pad_x = int(round(box_width * padding_ratio))
    pad_y = int(round(box_height * padding_ratio))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(width, x + box_width + pad_x)
    bottom = min(height, y + box_height + pad_y)
    return left, top, max(1, right - left), max(1, bottom - top)


def _crop_frame_with_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    x, y, width, height = bbox
    face_crop = frame[y : y + height, x : x + width].copy()
    return make_image_dict_from_bgr_frame(face_crop, file_name="webcam_face_crop.png")


def _detect_face_with_strategy(
    frame: np.ndarray,
    frame_image: dict,
    profile: dict,
    previous_bbox: tuple[int, int, int, int] | None,
    frame_index: int,
) -> tuple[dict, dict]:
    detection_interval = max(1, int(profile["detection_interval"]))
    should_run_full_detection = (
        previous_bbox is None
        or frame_index <= 1
        or (frame_index - 1) % detection_interval == 0
        or not _bbox_is_valid(frame.shape, previous_bbox)
    )
    if should_run_full_detection:
        face_result = detect_face_region(frame_image, save_outputs=False)
        return face_result, {"mode": "detector", "reused_bbox": False}

    tracking_bbox = _expand_bbox(frame.shape, previous_bbox, float(profile["tracking_padding"]))
    tracked_face = _crop_frame_with_bbox(frame, tracking_bbox)
    analysis_size = int(profile["analysis_size"])
    tracked_analysis = _prepare_analysis_face(tracked_face, target_size=(analysis_size, analysis_size))
    return (
        {
            "face_image": tracked_face,
            "analysis_face_image": tracked_analysis,
            "bounding_box": tracking_bbox,
            "face_coordinates": {
                "x": tracking_bbox[0],
                "y": tracking_bbox[1],
                "width": tracking_bbox[2],
                "height": tracking_bbox[3],
            },
            "confidence_score": None,
            "detector": "tracked_bbox_reuse",
            "preview_path": None,
            "cropped_face_path": None,
            "detection_count": 1,
        },
        {"mode": "tracking", "reused_bbox": True},
    )


def _resize_for_bbox(face_pixels: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Resize a transformed face ROI back to the original bounding-box size."""
    _, _, width, height = bbox
    return cv2.resize(face_pixels, (width, height), interpolation=cv2.INTER_LINEAR)


def _composite_face_into_frame(frame: np.ndarray, bbox: tuple[int, int, int, int], face_pixels: np.ndarray) -> np.ndarray:
    """Paste a transformed face ROI back into the live frame."""
    x, y, width, height = bbox
    composite = frame.copy()
    composite[y : y + height, x : x + width] = _resize_for_bbox(face_pixels, bbox)
    cv2.rectangle(composite, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return composite


def _prepare_analysis_face(face_image: dict, target_size: tuple[int, int] = (512, 512)) -> dict:
    """Build a normalized copy for metrics and Fourier analysis without altering preview ROI size."""
    normalized_target = (int(target_size[0]), int(target_size[1]))
    pixels = cv2.resize(face_image["pixels"], normalized_target, interpolation=cv2.INTER_AREA)
    return {
        **face_image,
        "pixels": pixels,
        "width": normalized_target[0],
        "height": normalized_target[1],
        "shape": pixels.shape,
        "target_size": normalized_target,
        "normalized_face": True,
    }


def _prepare_landmark_inference_face(face_image: dict, max_dimension: int = 512) -> tuple[dict, float]:
    """Upscale small face crops for more stable landmark inference without changing aspect ratio."""
    height, width = face_image["pixels"].shape[:2]
    current_max = max(height, width)
    if current_max >= max_dimension:
        return face_image, 1.0

    scale = max_dimension / max(current_max, 1)
    resized = cv2.resize(
        face_image["pixels"],
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return (
        {
            **face_image,
            "pixels": resized,
            "width": resized.shape[1],
            "height": resized.shape[0],
            "shape": resized.shape,
            "landmark_inference_scale": scale,
        },
        scale,
    )


def _detect_stable_landmarks_for_face(
    face_image: dict,
    selected_regions: list[str],
    show_landmarks: bool = False,
) -> dict:
    """Run the stabilized landmark path on a still face crop."""
    landmarks_result = _prepare_landmarks_if_needed(
        face_image,
        transformation="reference_expression_transfer",
        show_landmarks=show_landmarks,
        selected_regions=selected_regions,
    )
    if landmarks_result is None:
        raise ValueError("Landmark preparation failed.")
    return landmarks_result


def _rescale_landmarks_to_face(landmarks_result: dict, scale: float, target_face: dict) -> dict:
    """Map landmark coordinates from the inference image back to the original face crop."""
    if scale == 1.0:
        return landmarks_result

    scaled_landmarks: list[dict] = []
    target_height, target_width = target_face["pixels"].shape[:2]
    for landmark in landmarks_result["landmarks"]:
        scaled_landmarks.append(
            {
                **landmark,
                "x": min(max(int(round(landmark["x"] / scale)), 0), target_width - 1),
                "y": min(max(int(round(landmark["y"] / scale)), 0), target_height - 1),
            }
        )

    return {
        **landmarks_result,
        "landmarks": scaled_landmarks,
        "pixel_coordinates": [(landmark["x"], landmark["y"]) for landmark in scaled_landmarks],
        "inference_scale": scale,
        "face_image": landmarks_result["face_image"],
    }


def _render_landmark_overlay(
    face_image: dict,
    landmarks: list[dict],
    show_full_mesh: bool,
    selected_regions: list[str],
) -> np.ndarray:
    """Render landmarks directly from known coordinates instead of re-running MediaPipe."""
    visualization = toggle_landmark_visualization(
        face_image,
        landmarks,
        show_full_mesh=show_full_mesh,
        selected_regions=selected_regions,
    )
    return cv2.cvtColor(visualization["pixels"], cv2.COLOR_RGB2BGR)


def _should_render_full_mesh(selected_regions: list[str]) -> bool:
    """Prefer region-focused overlays; fall back to full mesh only when nothing is selected."""
    return len(selected_regions) == 0


def _smooth_landmarks(
    current_landmarks: list[dict],
    previous_landmarks: list[dict] | None,
    smoothing_alpha: float = 0.65,
) -> list[dict]:
    """Apply EMA smoothing to reduce webcam landmark jitter."""
    if not previous_landmarks or len(previous_landmarks) != len(current_landmarks):
        return current_landmarks

    alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
    smoothed: list[dict] = []
    for current, previous in zip(current_landmarks, previous_landmarks):
        smoothed.append(
            {
                **current,
                "x": int(round(previous["x"] * (1.0 - alpha) + current["x"] * alpha)),
                "y": int(round(previous["y"] * (1.0 - alpha) + current["y"] * alpha)),
            }
        )
    return smoothed


def _prepare_landmarks_if_needed(
    face_image: dict,
    transformation: str,
    show_landmarks: bool,
    selected_regions: list[str],
    landmark_max_dimension: int = 512,
) -> dict | None:
    needs_landmarks = show_landmarks or transformation in EXPRESSION_TRANSFORMATIONS or transformation == "reference_expression_transfer"
    if not needs_landmarks:
        return None
    inference_face, scale = _prepare_landmark_inference_face(face_image, max_dimension=landmark_max_dimension)
    landmarks_result = detect_landmarks(
        inference_face,
        show_full_mesh=show_landmarks and _should_render_full_mesh(selected_regions),
        selected_regions=selected_regions,
        save_outputs=False,
    )
    return _rescale_landmarks_to_face(landmarks_result, scale, face_image)


def prepare_reference_expression_payload(
    reference_image_source: str,
    show_landmarks: bool = False,
    selected_regions: list[str] | None = None,
) -> dict:
    """Precompute reference face ROI and landmarks for repeated transfer use."""
    reference_request = request_image_input(reference_image_source)
    reference_preprocessed = preprocess_image(reference_request["image"], save_outputs=False)
    reference_face = detect_face_region(reference_request["image"], save_outputs=False)
    selected_regions = selected_regions or DEFAULT_TRANSFER_REGIONS
    reference_landmarks = _prepare_landmarks_if_needed(
        reference_face["face_image"],
        transformation="reference_expression_transfer",
        show_landmarks=show_landmarks,
        selected_regions=selected_regions,
    )
    if reference_landmarks is None:
        raise ValueError("Reference landmark preparation failed.")
    return {
        "input": reference_request,
        "preprocessed": reference_preprocessed,
        "face_detection": reference_face,
        "landmarks": reference_landmarks,
        "selected_regions": selected_regions,
    }


def _apply_selected_transformation(
    face_image: dict,
    transformation: str,
    intensity: float,
    landmarks_result: dict | None,
    reference_payload: dict | None = None,
    selected_regions: list[str] | None = None,
    transfer_method: str = DEFAULT_TRANSFER_METHOD,
) -> dict:
    if transformation in EXPRESSION_TRANSFORMATIONS:
        if landmarks_result is None:
            raise ValueError("Landmark detection is required for geometric expression warping.")
        return apply_expression_warp(
            face_image,
            landmarks_result["landmarks"],
            transformation=transformation,
            intensity=intensity,
            save_outputs=False,
        )
    if transformation == "aging":
        return apply_aging_filter(face_image, intensity=intensity)
    if transformation == "de-aging":
        return apply_deaging_filter(face_image, intensity=intensity)
    if transformation == "reference_expression_transfer":
        if landmarks_result is None or reference_payload is None:
            raise ValueError("Reference expression transfer requires both source and reference landmarks.")
        return apply_reference_expression_transfer(
            face_image,
            landmarks_result["landmarks"],
            reference_payload["landmarks"]["landmarks"],
            blend_factor=intensity,
            regions=selected_regions,
            method=transfer_method,
            reference_face_image=reference_payload["face_detection"]["face_image"],
            save_outputs=False,
        )
    supported = ", ".join(sorted(GUI_TRANSFORMATIONS))
    raise ValueError(f"Unsupported transformation '{transformation}'. Supported: {supported}")


def run_realtime_frame_pipeline(
    frame: np.ndarray,
    transformation: str = "aging",
    intensity: float = 0.5,
    show_landmarks: bool = False,
    selected_regions: list[str] | None = None,
    reference_payload: dict | None = None,
    previous_landmarks: list[dict] | None = None,
    previous_bbox: tuple[int, int, int, int] | None = None,
    frame_index: int = 1,
    smoothing_alpha: float = 0.65,
    transfer_method: str = DEFAULT_TRANSFER_METHOD,
    realtime_profile: dict | None = None,
) -> dict:
    """Process a live webcam frame and return original/transformed previews."""
    profile = _merge_realtime_profile(realtime_profile)
    selected_regions = selected_regions or (
        DEFAULT_TRANSFER_REGIONS if transformation == "reference_expression_transfer" else DEFAULT_LANDMARK_REGIONS
    )
    timings: dict[str, float] = {}
    stage_start = time.perf_counter()
    frame_image = make_image_dict_from_bgr_frame(frame, file_name="webcam_frame.png")
    timings["frame_prepare_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    face_result, detection_debug = _detect_face_with_strategy(
        frame,
        frame_image,
        profile=profile,
        previous_bbox=previous_bbox,
        frame_index=frame_index,
    )
    timings["face_detect_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    landmarks_result = _prepare_landmarks_if_needed(
        face_result["face_image"],
        transformation=transformation,
        show_landmarks=show_landmarks,
        selected_regions=selected_regions,
        landmark_max_dimension=int(profile["landmark_max_dimension"]),
    )
    if landmarks_result is not None:
        smoothed_landmarks = _smooth_landmarks(
            landmarks_result["landmarks"],
            previous_landmarks,
            smoothing_alpha=smoothing_alpha,
        )
        landmarks_result = {
            **landmarks_result,
            "landmarks": smoothed_landmarks,
            "pixel_coordinates": [(landmark["x"], landmark["y"]) for landmark in smoothed_landmarks],
        }
    timings["landmark_detect_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    transformed = _apply_selected_transformation(
        face_result["face_image"],
        transformation=transformation,
        intensity=intensity,
        landmarks_result=landmarks_result,
        reference_payload=reference_payload,
        selected_regions=selected_regions,
        transfer_method=transfer_method,
    )
    timings["transform_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    original_frame = frame.copy()
    x, y, width, height = face_result["bounding_box"]
    cv2.rectangle(original_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    transformed_frame = _composite_face_into_frame(frame, face_result["bounding_box"], transformed["image"]["pixels"])

    if show_landmarks and landmarks_result is not None:
        original_frame = _composite_face_into_frame(
            frame,
            face_result["bounding_box"],
            _render_landmark_overlay(
                face_result["face_image"],
                landmarks_result["landmarks"],
                show_full_mesh=_should_render_full_mesh(selected_regions),
                selected_regions=selected_regions,
            ),
        )
        transformed_overlay_landmarks = transformed.get("target_landmarks")
        if transformed_overlay_landmarks is not None:
            transformed_landmarks = {
                "landmarks": transformed_overlay_landmarks,
                "visualization": None,
            }
            transformed_frame = _composite_face_into_frame(
                frame,
                face_result["bounding_box"],
                _render_landmark_overlay(
                    transformed["image"],
                    transformed_overlay_landmarks,
                    show_full_mesh=_should_render_full_mesh(selected_regions),
                    selected_regions=selected_regions,
                ),
            )
        else:
            transformed_landmarks = detect_landmarks(
                _prepare_landmark_inference_face(transformed["image"])[0],
                show_full_mesh=_should_render_full_mesh(selected_regions),
                selected_regions=selected_regions,
                save_outputs=False,
            )
            transformed_frame = _composite_face_into_frame(
                frame,
                face_result["bounding_box"],
                cv2.cvtColor(transformed_landmarks["visualization"]["pixels"], cv2.COLOR_RGB2BGR),
            )
    else:
        transformed_landmarks = None
    timings["overlay_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    composite_frame = cv2.hconcat([original_frame, transformed_frame])
    cv2.putText(
        composite_frame,
        f"{transformation} intensity={float(np.clip(intensity, 0.0, 1.0)):.2f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    timings["compose_ms"] = (time.perf_counter() - stage_start) * 1000.0
    timings["pipeline_total_ms"] = sum(
        timings[key]
        for key in ("frame_prepare_ms", "face_detect_ms", "landmark_detect_ms", "transform_ms", "overlay_ms", "compose_ms")
    )

    return {
        "frame_image": frame_image,
        "face_detection": face_result,
        "landmarks": landmarks_result,
        "transformed_landmarks": transformed_landmarks,
        "transformation": transformed,
        "original_frame": original_frame,
        "transformed_frame": transformed_frame,
        "composite_frame": composite_frame,
        "timings": timings,
        "debug": {
            "detection_mode": detection_debug["mode"],
            "reused_bbox": detection_debug["reused_bbox"],
            "profile": profile,
        },
    }


def run_preprocessing_pipeline(image_source: str) -> dict:
    image_request = request_image_input(image_source)
    return preprocess_image(image_request["image"])


def run_face_detection_pipeline(image_source: str) -> dict:
    image_request = request_image_input(image_source)
    preprocess_image(image_request["image"])
    return detect_face_region(image_request["image"])


def run_landmark_pipeline(
    image_source: str,
    show_full_mesh: bool = True,
    selected_regions: list[str] | None = None,
) -> dict:
    image_request = request_image_input(image_source)
    preprocess_image(image_request["image"])
    face_crop = detect_face_region(image_request["image"])
    return detect_landmarks(
        face_crop["face_image"],
        show_full_mesh=show_full_mesh,
        selected_regions=selected_regions,
    )


def run_expression_warp_pipeline(
    image_source: str,
    transformation: str = "smile_enhancement",
    intensity: float = 0.5,
) -> dict:
    image_request = request_image_input(image_source)
    preprocess_image(image_request["image"])
    face_crop = detect_face_region(image_request["image"])
    landmarks = detect_landmarks(face_crop["face_image"], show_full_mesh=False, selected_regions=["eyes", "lips"], save_outputs=False)
    return apply_expression_warp(
        face_crop["face_image"],
        landmarks["landmarks"],
        transformation=transformation,
        intensity=intensity,
    )


def run_frequency_analysis_pipeline(image_source: str) -> dict:
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    return analyze_frequency_content(preprocessed["rgb_image"])


def run_aging_pipeline(image_source: str, intensity: float = 0.5) -> dict:
    image_request = request_image_input(image_source)
    preprocess_image(image_request["image"])
    face_crop = detect_face_region(image_request["image"])
    return apply_aging_filter(face_crop["face_image"], intensity=intensity)


def run_deaging_pipeline(image_source: str, intensity: float = 0.5) -> dict:
    image_request = request_image_input(image_source)
    preprocess_image(image_request["image"])
    face_crop = detect_face_region(image_request["image"])
    return apply_deaging_filter(face_crop["face_image"], intensity=intensity)


def run_reference_expression_transfer_pipeline(
    image_source: str,
    reference_image_source: str,
    blend_factor: float = 0.7,
    show_landmarks: bool = False,
    selected_regions: list[str] | None = None,
    transfer_method: str = DEFAULT_TRANSFER_METHOD,
) -> dict:
    source_request = request_image_input(image_source)
    source_preprocessed = preprocess_image(source_request["image"])
    source_face = detect_face_region(source_request["image"])

    selected_regions = selected_regions or DEFAULT_TRANSFER_REGIONS
    reference_payload = prepare_reference_expression_payload(
        reference_image_source,
        show_landmarks=show_landmarks,
        selected_regions=selected_regions,
    )
    source_landmarks = _detect_stable_landmarks_for_face(
        source_face["face_image"],
        selected_regions=selected_regions,
        show_landmarks=show_landmarks,
    )

    transformed = apply_reference_expression_transfer(
        source_face["face_image"],
        source_landmarks["landmarks"],
        reference_payload["landmarks"]["landmarks"],
        blend_factor=blend_factor,
        regions=selected_regions,
        method=transfer_method,
        reference_face_image=reference_payload["face_detection"]["face_image"],
    )
    transformed_landmarks = _detect_stable_landmarks_for_face(
        transformed["image"],
        selected_regions=selected_regions,
        show_landmarks=False,
    )
    source_analysis_face = source_face["analysis_face_image"]
    transformed_analysis_face = _prepare_analysis_face(
        transformed["image"],
        target_size=source_analysis_face.get("target_size", (512, 512)),
    )
    original_frequency = analyze_frequency_content(source_analysis_face, save_outputs=False)
    transformed_frequency = analyze_frequency_content(transformed_analysis_face, save_outputs=False)
    metrics = evaluate_transformation(source_analysis_face, transformed_analysis_face, save_outputs=True)
    metrics["original_high_low_ratio"] = original_frequency["high_low_ratio"]
    metrics["transformed_high_low_ratio"] = transformed_frequency["high_low_ratio"]
    reference_coefficients = extract_expression_coefficients(reference_payload["landmarks"]["landmarks"])
    transformed_coefficients = extract_expression_coefficients(transformed_landmarks["landmarks"])
    source_coefficients = extract_expression_coefficients(source_landmarks["landmarks"])
    expression_keys = [
        "left_eye_open",
        "right_eye_open",
        "mouth_width",
        "mouth_open",
        "brow_raise_left",
        "brow_raise_right",
        "mouth_corner_left",
        "mouth_corner_right",
    ]
    expression_distance = float(
        np.mean([abs(transformed_coefficients[key] - reference_coefficients[key]) for key in expression_keys])
    )
    source_expression_distance = float(
        np.mean([abs(source_coefficients[key] - reference_coefficients[key]) for key in expression_keys])
    )
    expression_match_score = max(0.0, 100.0 * (1.0 - min(expression_distance / 0.12, 1.0)))
    identity_preservation_score = max(0.0, min(metrics["ssim"], 1.0) * 100.0)
    transfer_quality_score = 0.65 * expression_match_score + 0.35 * identity_preservation_score
    metrics["expression_distance_to_reference"] = expression_distance
    metrics["source_expression_distance_to_reference"] = source_expression_distance
    metrics["expression_match_score"] = expression_match_score
    metrics["identity_preservation_score"] = identity_preservation_score
    metrics["transfer_quality_score"] = transfer_quality_score

    return {
        "source_input": source_request,
        "reference_input": reference_payload["input"],
        "source_face_detection": source_face,
        "reference_face_detection": reference_payload["face_detection"],
        "source_landmarks": source_landmarks,
        "reference_landmarks": reference_payload["landmarks"],
        "transformed_landmarks": transformed_landmarks,
        "transformation": transformed,
        "original_frequency": original_frequency,
        "transformed_frequency": transformed_frequency,
        "metrics": metrics,
        "transfer_method": transfer_method,
    }


def compare_reference_expression_transfer_methods(
    image_source: str,
    reference_image_source: str,
    blend_factor: float = 0.7,
    show_landmarks: bool = False,
    selected_regions: list[str] | None = None,
    methods: list[str] | None = None,
) -> dict:
    """Run multiple transfer methods on the same inputs and rank them by transfer quality."""
    selected_regions = selected_regions or DEFAULT_TRANSFER_REGIONS
    methods = methods or sorted(TRANSFER_METHODS)
    results: dict[str, dict] = {}
    for method in methods:
        results[method] = run_reference_expression_transfer_pipeline(
            image_source,
            reference_image_source,
            blend_factor=blend_factor,
            show_landmarks=show_landmarks,
            selected_regions=selected_regions,
            transfer_method=method,
        )
    ranked_methods = sorted(
        methods,
        key=lambda item: results[item]["metrics"]["transfer_quality_score"],
        reverse=True,
    )
    return {
        "results": results,
        "ranked_methods": ranked_methods,
        "best_method": ranked_methods[0],
    }


def run_analysis_pipeline(
    image_source: str,
    transformation: str = "smile_enhancement",
    intensity: float = 0.5,
    show_landmarks: bool = False,
    selected_regions: list[str] | None = None,
) -> dict:
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    face_crop = detect_face_region(image_request["image"])

    selected_regions = selected_regions or DEFAULT_LANDMARK_REGIONS
    landmarks_result = _prepare_landmarks_if_needed(
        face_crop["face_image"],
        transformation=transformation,
        show_landmarks=show_landmarks,
        selected_regions=selected_regions,
    )

    transformed = _apply_selected_transformation(
        face_crop["face_image"],
        transformation=transformation,
        intensity=intensity,
        landmarks_result=landmarks_result,
        reference_payload=None,
        selected_regions=selected_regions,
    )
    analysis_face = face_crop["analysis_face_image"]
    transformed_analysis_face = _prepare_analysis_face(
        transformed["image"],
        target_size=analysis_face.get("target_size", (512, 512)),
    )
    original_frequency = analyze_frequency_content(analysis_face, save_outputs=False)
    transformed_frequency = analyze_frequency_content(transformed_analysis_face, save_outputs=False)
    metrics = evaluate_transformation(analysis_face, transformed_analysis_face, save_outputs=True)
    metrics["original_high_low_ratio"] = original_frequency["high_low_ratio"]
    metrics["transformed_high_low_ratio"] = transformed_frequency["high_low_ratio"]

    return {
        "input": image_request,
        "preprocessed": preprocessed,
        "face_detection": face_crop,
        "landmarks": landmarks_result,
        "transformation": transformed,
        "original_frequency": original_frequency,
        "transformed_frequency": transformed_frequency,
        "metrics": metrics,
    }


def run_pipeline(image_source: str) -> dict:
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    face_crop = detect_face_region(image_request["image"])
    landmarks = detect_landmarks(face_crop["face_image"])
    warped = apply_expression_warp(face_crop["face_image"], landmarks["landmarks"])
    aged = apply_aging_filter(warped["image"])
    deaged = apply_deaging_filter(warped["image"])
    analysis_face = face_crop["analysis_face_image"]
    warped_analysis_face = _prepare_analysis_face(warped["image"], target_size=analysis_face.get("target_size", (512, 512)))
    original_frequency = analyze_frequency_content(analysis_face)
    transformed_frequency = analyze_frequency_content(warped_analysis_face)
    metrics = evaluate_transformation(analysis_face, warped_analysis_face)

    return build_result_summary(
        original_image=image_request["image"],
        preprocessed_image=preprocessed["image"],
        face_image=face_crop["face_image"],
        landmarks=landmarks["landmarks"],
        warped_image=warped["image"],
        aged_image=aged["image"],
        deaged_image=deaged["image"],
        original_frequency=original_frequency,
        transformed_frequency=transformed_frequency,
        metrics=metrics,
    )


if __name__ == "__main__":
    print(
        "Use run_preprocessing_pipeline(...), run_face_detection_pipeline(...), run_landmark_pipeline(...), "
        "run_expression_warp_pipeline(...), run_frequency_analysis_pipeline(...), run_aging_pipeline(...), "
        "run_deaging_pipeline(...), run_reference_expression_transfer_pipeline(...), run_realtime_frame_pipeline(...), "
        "prepare_reference_expression_payload(...), or run_analysis_pipeline(...)."
    )
