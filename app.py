"""Application entry point for the facial image processing pipeline."""

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from facial_image_warping.input_module import request_image_input
from facial_image_warping.preprocessing import preprocess_image
from facial_image_warping.face_detection import detect_face_region
from facial_image_warping.landmark_detection import detect_landmarks
from facial_image_warping.geometric_warping import apply_expression_warp
from facial_image_warping.aging_filter import apply_aging_filter, apply_deaging_filter
from facial_image_warping.fourier_analysis import analyze_frequency_content
from facial_image_warping.evaluation import evaluate_transformation
from facial_image_warping.visualization import build_result_summary


def run_preprocessing_pipeline(image_source: str) -> dict:
    """Run only the Sprint 1 image input and preprocessing stages."""
    image_request = request_image_input(image_source)
    return preprocess_image(image_request["image"])


def run_face_detection_pipeline(image_source: str) -> dict:
    """Run Sprint 1 preprocessing followed by Sprint 2 face detection."""
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    return detect_face_region(preprocessed["rgb_image"])


def run_landmark_pipeline(
    image_source: str,
    show_full_mesh: bool = True,
    selected_regions: list[str] | None = None,
) -> dict:
    """Run preprocessing, face detection, and MediaPipe landmark extraction."""
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
    """Run preprocessing, face detection, landmarks, and geometric warping."""
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    face_crop = detect_face_region(preprocessed["rgb_image"])
    landmarks = detect_landmarks(face_crop["face_image"], show_full_mesh=False, selected_regions=["eyes", "lips"])
    return apply_expression_warp(
        face_crop["face_image"],
        landmarks["landmarks"],
        transformation=transformation,
        intensity=intensity,
    )


def run_frequency_analysis_pipeline(image_source: str) -> dict:
    """Run preprocessing followed by Fourier-domain analysis."""
    image_request = request_image_input(image_source)
    preprocessed = preprocess_image(image_request["image"])
    return analyze_frequency_content(preprocessed["rgb_image"])


def run_pipeline(image_source: str) -> dict:
    """Run the planned end-to-end DSP workflow on a single input image."""
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
    print("Use run_preprocessing_pipeline(...), run_face_detection_pipeline(...), run_landmark_pipeline(...), run_expression_warp_pipeline(...), or run_frequency_analysis_pipeline(...).")
