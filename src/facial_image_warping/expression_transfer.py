"""Reference-driven facial expression transfer utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facial_image_warping.geometric_warping import apply_delaunay_triangulation, warp_triangle
from facial_image_warping.landmark_detection import FACE_REGIONS


TRANSFER_OUTPUT_DIR = Path("outputs/transfer")
DEFAULT_TRANSFER_REGIONS = ["eyebrows", "eyes", "lips", "nose"]


def _ensure_bgr_uint8(image: dict) -> dict:
    """Convert supported image payloads to uint8 BGR for warping."""
    pixels = image["pixels"]
    if pixels.dtype.kind == "f":
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)

    color_space = image.get("color_space", "BGR")
    if color_space == "BGR":
        bgr_pixels = pixels
    elif color_space == "RGB":
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    elif color_space == "GRAYSCALE":
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported color space for expression transfer: {color_space}")

    return {
        **image,
        "pixels": bgr_pixels,
        "shape": bgr_pixels.shape,
        "color_space": "BGR",
        "dtype": str(bgr_pixels.dtype),
    }


def _save_bgr_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save a BGR image with Unicode-safe file writing."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoded, buffer = cv2.imencode(output_file.suffix or ".png", image)
    if not encoded:
        raise ValueError(f"Failed to encode image for {output_file}")
    buffer.tofile(output_file)
    return output_file


def create_expression_transfer_targets(
    source_landmarks: list[dict],
    reference_landmarks: list[dict],
    blend_factor: float = 0.7,
    regions: list[str] | None = None,
) -> list[dict]:
    """Create target landmarks by blending expressive source regions toward a reference face."""
    if len(source_landmarks) != len(reference_landmarks):
        raise ValueError(
            "Source and reference landmark sets must have identical sizes for expression transfer. "
            f"Got {len(source_landmarks)} and {len(reference_landmarks)}."
        )

    blend_factor = float(np.clip(blend_factor, 0.0, 1.0))
    regions = regions or DEFAULT_TRANSFER_REGIONS
    targets = [dict(landmark) for landmark in source_landmarks]

    for region in regions:
        if region not in FACE_REGIONS:
            raise ValueError(f"Unknown transfer region requested: {region}")
        for index in FACE_REGIONS[region]:
            source = source_landmarks[index]
            reference = reference_landmarks[index]
            targets[index]["x"] = int(round(source["x"] + (reference["x"] - source["x"]) * blend_factor))
            targets[index]["y"] = int(round(source["y"] + (reference["y"] - source["y"]) * blend_factor))

    return targets


def apply_reference_expression_transfer(
    source_face_image: dict,
    source_landmarks: list[dict],
    reference_landmarks: list[dict],
    blend_factor: float = 0.7,
    regions: list[str] | None = None,
    save_outputs: bool = True,
) -> dict:
    """Warp a source face toward the expression geometry of a reference face."""
    prepared_face = _ensure_bgr_uint8(source_face_image)
    source_pixels = prepared_face["pixels"]
    target_landmarks = create_expression_transfer_targets(
        source_landmarks,
        reference_landmarks,
        blend_factor=blend_factor,
        regions=regions,
    )
    triangles = apply_delaunay_triangulation(source_pixels.shape, source_landmarks)

    warped_pixels = source_pixels.astype(np.float32).copy()
    source_points = [(int(landmark["x"]), int(landmark["y"])) for landmark in source_landmarks]
    target_points = [(int(landmark["x"]), int(landmark["y"])) for landmark in target_landmarks]

    for triangle in triangles:
        src_triangle = [source_points[index] for index in triangle]
        dst_triangle = [target_points[index] for index in triangle]
        warp_triangle(source_pixels, warped_pixels, src_triangle, dst_triangle)

    warped_pixels = np.clip(warped_pixels, 0, 255).astype(np.uint8)
    comparison = cv2.hconcat([source_pixels, warped_pixels])

    stem = Path(prepared_face.get("file_name", "face.png")).stem
    transferred_path = TRANSFER_OUTPUT_DIR / f"{stem}_reference_expression.png"
    comparison_path = TRANSFER_OUTPUT_DIR / f"{stem}_reference_expression_comparison.png"
    if save_outputs:
        _save_bgr_image(warped_pixels, transferred_path)
        _save_bgr_image(comparison, comparison_path)

    return {
        "image": {
            **prepared_face,
            "pixels": warped_pixels,
            "shape": warped_pixels.shape,
            "color_space": "BGR",
            "dtype": str(warped_pixels.dtype),
            "reference_expression_transfer": True,
        },
        "operation": "reference_expression_transfer",
        "blend_factor": blend_factor,
        "target_landmarks": target_landmarks,
        "regions": regions or DEFAULT_TRANSFER_REGIONS,
        "triangles": triangles,
        "warped_image_path": str(transferred_path),
        "comparison_image_path": str(comparison_path),
        "explanation": [
            "MediaPipe facial landmarks are extracted from both source and reference faces.",
            "Expression-sensitive regions are blended toward the reference geometry.",
            "Delaunay triangle warping transfers the reference expression while keeping source identity cues.",
        ],
    }
