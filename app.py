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


def run_pipeline(image_source: str) -> dict:
    """Run the planned end-to-end DSP workflow on a single input image.

    Parameters
    ----------
    image_source:
        Path or identifier for the user-supplied facial image.

    Returns
    -------
    dict
        A structured dictionary containing placeholders for intermediate and
        final outputs.
    """
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
    example_result = run_pipeline("sample_face.png")
    print("Pipeline scaffold executed successfully.")
    print(f"Available result keys: {sorted(example_result.keys())}")
