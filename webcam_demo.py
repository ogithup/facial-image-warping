"""OpenCV-based real-time webcam demo for facial transformations."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from facial_image_warping.aging_filter import apply_aging_filter, apply_deaging_filter
from facial_image_warping.expression_transfer import apply_reference_expression_transfer
from facial_image_warping.face_detection import detect_face_region
from facial_image_warping.geometric_warping import apply_expression_warp
from facial_image_warping.input_module import load_image
from facial_image_warping.landmark_detection import detect_landmarks


EXPRESSION_TRANSFORMATIONS = {"smile_enhancement", "eyebrow_raising", "lip_widening", "face_slimming"}


def _frame_to_image_dict(frame: np.ndarray, file_name: str = "webcam_frame.png") -> dict:
    return {
        "file_name": file_name,
        "format": "png",
        "width": frame.shape[1],
        "height": frame.shape[0],
        "shape": frame.shape,
        "pixels": frame,
        "color_space": "BGR",
        "dtype": str(frame.dtype),
    }


def _render_result(frame: np.ndarray, transformation: str, intensity: float, reference_landmarks: list[dict] | None) -> np.ndarray:
    face_result = detect_face_region(_frame_to_image_dict(frame), save_outputs=False)
    face_image = face_result["face_image"]

    if transformation in EXPRESSION_TRANSFORMATIONS:
        landmarks = detect_landmarks(face_image, show_full_mesh=False, selected_regions=["eyes", "lips", "nose"], save_outputs=False)
        transformed = apply_expression_warp(face_image, landmarks["landmarks"], transformation=transformation, intensity=intensity, save_outputs=False)
    elif transformation == "aging":
        transformed = apply_aging_filter(face_image, intensity=intensity)
    elif transformation == "de-aging":
        transformed = apply_deaging_filter(face_image, intensity=intensity)
    elif transformation == "reference_expression_transfer":
        if reference_landmarks is None:
            raise ValueError("Reference landmarks are required for reference expression transfer.")
        landmarks = detect_landmarks(face_image, show_full_mesh=False, selected_regions=["eyes", "lips", "nose"], save_outputs=False)
        transformed = apply_reference_expression_transfer(
            face_image,
            landmarks["landmarks"],
            reference_landmarks,
            blend_factor=intensity,
            save_outputs=False,
        )
    else:
        raise ValueError(f"Unsupported webcam transformation: {transformation}")

    original = face_image["pixels"]
    result = transformed["image"]["pixels"]
    canvas = cv2.hconcat([original, result])
    cv2.putText(canvas, f"{transformation} intensity={intensity:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time webcam facial transformation demo")
    parser.add_argument("--transformation", default="aging", choices=["aging", "de-aging", "smile_enhancement", "eyebrow_raising", "lip_widening", "face_slimming", "reference_expression_transfer"])
    parser.add_argument("--intensity", type=float, default=0.5)
    parser.add_argument("--reference-image", type=str, default=None)
    args = parser.parse_args()

    reference_landmarks = None
    if args.transformation == "reference_expression_transfer":
        if not args.reference_image:
            raise ValueError("--reference-image is required for reference_expression_transfer mode.")
        reference_image = load_image(Path(args.reference_image))
        reference_face = detect_face_region(reference_image, save_outputs=False)
        reference_landmarks = detect_landmarks(reference_face["face_image"], show_full_mesh=False, selected_regions=["eyes", "lips", "nose"], save_outputs=False)["landmarks"]

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Failed to open the default webcam.")

    print("Press Q to quit the real-time webcam demo.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            try:
                rendered = _render_result(frame, args.transformation, args.intensity, reference_landmarks)
                cv2.imshow("Facial Image Warping Webcam Demo", rendered)
            except Exception as exc:
                fallback = frame.copy()
                cv2.putText(fallback, str(exc)[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.imshow("Facial Image Warping Webcam Demo", fallback)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
