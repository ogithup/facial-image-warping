"""Application entry point for the facial image processing pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from facial_image_warping.input_module import request_image_input
from facial_image_warping.preprocessing import preprocess_image
from facial_image_warping.face_detection import detect_face_region
from facial_image_warping.landmark_detection import detect_landmarks
from facial_image_warping.geometric_warping import apply_expression_warp
from facial_image_warping.aging_filter import apply_aging_filter, apply_deaging_filter
from facial_image_warping.expression_transfer import apply_reference_expression_transfer
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
DEFAULT_TRANSFER_REGIONS = ["eyes", "eyebrows", "nose", "lips"]


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


def _prepare_landmarks_if_needed(
    face_image: dict,
    transformation: str,
    show_landmarks: bool,
    selected_regions: list[str],
) -> dict | None:
    needs_landmarks = show_landmarks or transformation in EXPRESSION_TRANSFORMATIONS or transformation == "reference_expression_transfer"
    if not needs_landmarks:
        return None
    return detect_landmarks(
        face_image,
        show_full_mesh=show_landmarks,
        selected_regions=selected_regions,
        save_outputs=False,
    )


def prepare_reference_expression_payload(
    reference_image_source: str,
    show_landmarks: bool = False,
    selected_regions: list[str] | None = None,
) -> dict:
    """Precompute reference face ROI and landmarks for repeated transfer use."""
    reference_request = request_image_input(reference_image_source)
    reference_preprocessed = preprocess_image(reference_request["image"], save_outputs=False)
    reference_face = detect_face_region(reference_preprocessed["rgb_image"], save_outputs=False)
    selected_regions = selected_regions or DEFAULT_TRANSFER_REGIONS
    reference_landmarks = detect_landmarks(
        reference_face["face_image"],
        show_full_mesh=show_landmarks,
        selected_regions=selected_regions,
        save_outputs=False,
    )
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
) -> dict:
    """Process a live webcam frame and return original/transformed previews."""
    selected_regions = selected_regions or (
        DEFAULT_TRANSFER_REGIONS if transformation == "reference_expression_transfer" else DEFAULT_LANDMARK_REGIONS
    )
    frame_image = make_image_dict_from_bgr_frame(frame, file_name="webcam_frame.png")
    face_result = detect_face_region(frame_image, save_outputs=False)
    landmarks_result = _prepare_landmarks_if_needed(
        face_result["face_image"],
        transformation=transformation,
        show_landmarks=show_landmarks,
        selected_regions=selected_regions,
    )
    transformed = _apply_selected_transformation(
        face_result["face_image"],
        transformation=transformation,
        intensity=intensity,
        landmarks_result=landmarks_result,
        reference_payload=reference_payload,
        selected_regions=selected_regions,
    )

    original_frame = frame.copy()
    x, y, width, height = face_result["bounding_box"]
    cv2.rectangle(original_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    transformed_frame = _composite_face_into_frame(frame, face_result["bounding_box"], transformed["image"]["pixels"])

    if show_landmarks and landmarks_result is not None:
        original_frame = _composite_face_into_frame(frame, face_result["bounding_box"], cv2.cvtColor(landmarks_result["visualization"]["pixels"], cv2.COLOR_RGB2BGR))
        transformed_landmarks = detect_landmarks(
            transformed["image"],
            show_full_mesh=True,
            selected_regions=selected_regions,
            save_outputs=False,
        )
        transformed_frame = _composite_face_into_frame(frame, face_result["bounding_box"], cv2.cvtColor(transformed_landmarks["visualization"]["pixels"], cv2.COLOR_RGB2BGR))
    else:
        transformed_landmarks = None

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

    return {
        "frame_image": frame_image,
        "face_detection": face_result,
        "landmarks": landmarks_result,
        "transformed_landmarks": transformed_landmarks,
        "transformation": transformed,
        "original_frame": original_frame,
        "transformed_frame": transformed_frame,
        "composite_frame": composite_frame,
    }


def run_preprocessing_pipeline(image_source: str) -> dict:
    image_request = request_image_input(image_source)
    return preprocess_image(image_request["image"])


def run_face_detection_pipeline(image_source: str) -> dict:
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    return detect_face_region(preprocessed["rgb_image"])


def run_landmark_pipeline(
    image_source: str,
    show_full_mesh: bool = True,
    selected_regions: list[str] | None = None,
) -> dict:
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    face_crop = detect_face_region(preprocessed["rgb_image"])
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
    preprocessed = preprocess_image(image_request["image"])
    face_crop = detect_face_region(preprocessed["rgb_image"])
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
    preprocessed = preprocess_image(image_request["image"])
    face_crop = detect_face_region(preprocessed["rgb_image"])
    return apply_aging_filter(face_crop["face_image"], intensity=intensity)


def run_deaging_pipeline(image_source: str, intensity: float = 0.5) -> dict:
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    face_crop = detect_face_region(preprocessed["rgb_image"])
    return apply_deaging_filter(face_crop["face_image"], intensity=intensity)


def run_reference_expression_transfer_pipeline(
    image_source: str,
    reference_image_source: str,
    blend_factor: float = 0.7,
    show_landmarks: bool = False,
    selected_regions: list[str] | None = None,
) -> dict:
    source_request = request_image_input(image_source)
    source_preprocessed = preprocess_image(source_request["image"])
    source_face = detect_face_region(source_preprocessed["rgb_image"])

    selected_regions = selected_regions or DEFAULT_TRANSFER_REGIONS
    reference_payload = prepare_reference_expression_payload(
        reference_image_source,
        show_landmarks=show_landmarks,
        selected_regions=selected_regions,
    )
    source_landmarks = detect_landmarks(
        source_face["face_image"],
        show_full_mesh=show_landmarks,
        selected_regions=selected_regions,
        save_outputs=show_landmarks,
    )

    transformed = apply_reference_expression_transfer(
        source_face["face_image"],
        source_landmarks["landmarks"],
        reference_payload["landmarks"]["landmarks"],
        blend_factor=blend_factor,
        regions=selected_regions,
    )
    original_frequency = analyze_frequency_content(source_face["face_image"], save_outputs=False)
    transformed_frequency = analyze_frequency_content(transformed["image"], save_outputs=False)
    metrics = evaluate_transformation(source_face["face_image"], transformed["image"], save_outputs=True)
    metrics["original_high_low_ratio"] = original_frequency["high_low_ratio"]
    metrics["transformed_high_low_ratio"] = transformed_frequency["high_low_ratio"]

    return {
        "source_input": source_request,
        "reference_input": reference_payload["input"],
        "source_face_detection": source_face,
        "reference_face_detection": reference_payload["face_detection"],
        "source_landmarks": source_landmarks,
        "reference_landmarks": reference_payload["landmarks"],
        "transformation": transformed,
        "original_frequency": original_frequency,
        "transformed_frequency": transformed_frequency,
        "metrics": metrics,
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
    face_crop = detect_face_region(preprocessed["rgb_image"])

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
    original_frequency = analyze_frequency_content(face_crop["face_image"], save_outputs=False)
    transformed_frequency = analyze_frequency_content(transformed["image"], save_outputs=False)
    metrics = evaluate_transformation(face_crop["face_image"], transformed["image"], save_outputs=True)
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
    face_crop = detect_face_region(preprocessed["image"])
    landmarks = detect_landmarks(face_crop["face_image"])
    warped = apply_expression_warp(face_crop["face_image"], landmarks["landmarks"])
    aged = apply_aging_filter(warped["image"])
    deaged = apply_deaging_filter(warped["image"])
    original_frequency = analyze_frequency_content(face_crop["face_image"])
    transformed_frequency = analyze_frequency_content(warped["image"])
    metrics = evaluate_transformation(face_crop["face_image"], warped["image"])

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
