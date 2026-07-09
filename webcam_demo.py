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


def _resize_frame_for_preview(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / max(width, 1)
    return cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _probe_camera_indices(max_devices: int = 8) -> list[int]:
    available: list[int] = []
    for index in range(max_devices):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                available.append(index)
        capture.release()
    return available


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
    parser.add_argument("--device-index", type=int, default=0, help="Windows camera index. Use DroidCam's index here.")
    parser.add_argument("--probe-devices", action="store_true", help="List camera indices that OpenCV can open, then exit.")
    parser.add_argument("--raw-preview", action="store_true", help="Show raw camera frames only, without face processing.")
    parser.add_argument("--max-width", type=int, default=640, help="Downscale preview width for more stable local capture.")
    args = parser.parse_args()

    if args.probe_devices:
        available = _probe_camera_indices()
        print("Available camera indices:", ", ".join(str(index) for index in available) if available else "none")
        return

    reference_landmarks = None
    if args.transformation == "reference_expression_transfer" and not args.raw_preview:
        if not args.reference_image:
            raise ValueError("--reference-image is required for reference_expression_transfer mode.")
        reference_image = load_image(Path(args.reference_image))
        reference_face = detect_face_region(reference_image, save_outputs=False)
        reference_landmarks = detect_landmarks(reference_face["face_image"], show_full_mesh=False, selected_regions=["eyes", "lips", "nose"], save_outputs=False)["landmarks"]

    capture = cv2.VideoCapture(args.device_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open camera index {args.device_index}.")

    print(f"Press Q to quit the real-time webcam demo. Device index: {args.device_index}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            try:
                preview_frame = _resize_frame_for_preview(frame, max_width=max(160, args.max_width))
                if args.raw_preview:
                    rendered = preview_frame.copy()
                    cv2.putText(rendered, "Raw camera debug", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                else:
                    rendered = _render_result(preview_frame, args.transformation, args.intensity, reference_landmarks)
                cv2.imshow("Facial Image Warping Webcam Demo", rendered)
            except Exception as exc:
                fallback = _resize_frame_for_preview(frame, max_width=max(160, args.max_width))
                cv2.putText(fallback, str(exc)[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.imshow("Facial Image Warping Webcam Demo", fallback)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
