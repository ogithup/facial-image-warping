"""Facial landmark extraction interfaces powered by MediaPipe Face Mesh."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


LANDMARKS_OUTPUT_DIR = Path("outputs/landmarks")
REGION_COLORS = {
    "eyes": (0, 255, 0),
    "eyebrows": (255, 0, 0),
    "nose": (0, 0, 255),
    "lips": (255, 0, 255),
    "jawline": (255, 255, 0),
    "cheeks": (0, 165, 255),
}
FACE_REGIONS = {
    "eyes": sorted(
        {
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
        }
    ),
    "eyebrows": sorted({46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 276, 283, 282, 295, 285, 300, 293, 334, 296, 336}),
    "nose": sorted({1, 2, 4, 5, 6, 19, 45, 48, 49, 98, 168, 195, 197, 198, 209, 218, 275, 278, 279, 344, 440}),
    "lips": sorted(
        {
            0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
            146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
        }
    ),
    "jawline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "cheeks": sorted({50, 101, 118, 119, 120, 121, 123, 126, 142, 147, 187, 203, 205, 206, 207, 280, 330, 347, 348, 349, 350, 352, 371, 411, 425, 427, 429}),
}


def _import_mediapipe():
    """Import MediaPipe lazily so the module remains importable without it."""
    import mediapipe as mp

    return mp


def ensure_rgb_uint8(image: dict) -> dict:
    """Convert image payloads into uint8 RGB for MediaPipe inference."""
    pixels = image["pixels"]
    if pixels.dtype.kind == "f":
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)

    color_space = image.get("color_space", "BGR")
    if color_space == "BGR":
        rgb_pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    elif color_space == "RGB":
        rgb_pixels = pixels
    elif color_space == "GRAYSCALE":
        rgb_pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported color space for landmark detection: {color_space}")

    return {
        **image,
        "pixels": rgb_pixels,
        "shape": rgb_pixels.shape,
        "color_space": "RGB",
        "dtype": str(rgb_pixels.dtype),
    }


def detect_face_mesh(
    image: dict,
    static_image_mode: bool = True,
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
):
    """Run MediaPipe Face Mesh and return its raw results object."""
    mp = _import_mediapipe()
    rgb_image = ensure_rgb_uint8(image)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
    ) as face_mesh:
        return face_mesh.process(rgb_image["pixels"])


def normalized_landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    """Convert MediaPipe's normalized coordinates into image pixel coordinates.

    MediaPipe returns ``x`` and ``y`` in normalized image space, meaning each
    value is relative to width and height in the ``[0, 1]`` range. To draw or
    crop using OpenCV, those normalized coordinates must be mapped back into
    absolute pixel positions.
    """
    x = min(max(int(round(landmark.x * (width - 1))), 0), width - 1)
    y = min(max(int(round(landmark.y * (height - 1))), 0), height - 1)
    return x, y


def extract_landmark_coordinates(face_landmarks, image_shape: tuple[int, ...]) -> list[dict]:
    """Extract normalized and pixel-space landmark coordinates."""
    height, width = image_shape[:2]
    coordinates = []
    for index, landmark in enumerate(face_landmarks.landmark):
        pixel_x, pixel_y = normalized_landmark_to_pixel(landmark, width, height)
        coordinates.append(
            {
                "index": index,
                "x": pixel_x,
                "y": pixel_y,
                "z": float(landmark.z),
                "normalized_x": float(landmark.x),
                "normalized_y": float(landmark.y),
            }
        )
    return coordinates


def draw_full_landmarks(image: dict, landmarks: list[dict], radius: int = 1) -> dict:
    """Draw all detected landmark points on the image."""
    annotated = ensure_rgb_uint8(image)["pixels"].copy()
    for landmark in landmarks:
        cv2.circle(annotated, (landmark["x"], landmark["y"]), radius, (0, 255, 255), -1)
    return {
        **image,
        "pixels": annotated,
        "shape": annotated.shape,
        "color_space": "RGB",
        "dtype": str(annotated.dtype),
    }


def draw_selected_regions(image: dict, landmarks: list[dict], regions: list[str]) -> dict:
    """Draw only the selected facial landmark regions."""
    annotated = ensure_rgb_uint8(image)["pixels"].copy()
    landmark_by_index = {landmark["index"]: landmark for landmark in landmarks}

    for region in regions:
        if region not in FACE_REGIONS:
            raise ValueError(f"Unknown landmark region requested: {region}")
        color = REGION_COLORS.get(region, (255, 255, 255))
        for landmark_index in FACE_REGIONS[region]:
            landmark = landmark_by_index.get(landmark_index)
            if landmark is None:
                continue
            cv2.circle(annotated, (landmark["x"], landmark["y"]), 2, color, -1)

    return {
        **image,
        "pixels": annotated,
        "shape": annotated.shape,
        "color_space": "RGB",
        "dtype": str(annotated.dtype),
    }


def toggle_landmark_visualization(
    image: dict,
    landmarks: list[dict],
    show_full_mesh: bool = True,
    selected_regions: list[str] | None = None,
) -> dict:
    """Toggle between full-mesh visualization and selected region overlays."""
    selected_regions = selected_regions or []
    base = ensure_rgb_uint8(image)

    if show_full_mesh:
        base = draw_full_landmarks(base, landmarks)

    if selected_regions:
        base = draw_selected_regions(base, landmarks, selected_regions)

    return base


def export_landmarks_to_json(landmarks: list[dict], output_path: str | Path) -> Path:
    """Export landmark coordinates to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(landmarks, file, indent=2)
    return output_file


def export_landmarks_to_csv(landmarks: list[dict], output_path: str | Path) -> Path:
    """Export landmark coordinates to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "x", "y", "z", "normalized_x", "normalized_y"])
        writer.writeheader()
        writer.writerows(landmarks)
    return output_file


def save_landmark_image(image: dict, output_path: str | Path) -> Path:
    """Save an RGB visualization image to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    bgr_pixels = cv2.cvtColor(image["pixels"], cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_file), bgr_pixels):
        raise ValueError(f"Failed to save landmark visualization to {output_file}")
    return output_file


def detect_landmarks(
    face_image: dict,
    show_full_mesh: bool = True,
    selected_regions: list[str] | None = None,
    save_outputs: bool = True,
) -> dict:
    """Detect 468 face landmarks and export optional visualizations.

    Parameters
    ----------
    face_image:
        Cropped facial image generated by the face detection stage.
    show_full_mesh:
        Whether to draw all landmarks on the output visualization.
    selected_regions:
        Optional subset of regions such as ``eyes`` or ``lips`` to highlight.
    save_outputs:
        Whether to write image and coordinate artifacts to disk.
    """
    rgb_face = ensure_rgb_uint8(face_image)
    results = detect_face_mesh(rgb_face)
    if not results.multi_face_landmarks:
        raise ValueError(
            "No facial landmarks detected. Use a frontal face crop with visible facial features and sufficient lighting."
        )

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = extract_landmark_coordinates(face_landmarks, rgb_face["pixels"].shape)
    visualization = toggle_landmark_visualization(
        rgb_face,
        landmarks,
        show_full_mesh=show_full_mesh,
        selected_regions=selected_regions,
    )

    stem = Path(face_image.get("file_name", "face.png")).stem
    image_path = LANDMARKS_OUTPUT_DIR / f"{stem}_landmarks.png"
    json_path = LANDMARKS_OUTPUT_DIR / f"{stem}_landmarks.json"
    csv_path = LANDMARKS_OUTPUT_DIR / f"{stem}_landmarks.csv"

    if save_outputs:
        save_landmark_image(visualization, image_path)
        export_landmarks_to_json(landmarks, json_path)
        export_landmarks_to_csv(landmarks, csv_path)

    return {
        "landmarks": landmarks,
        "pixel_coordinates": [(landmark["x"], landmark["y"]) for landmark in landmarks],
        "landmark_count": len(landmarks),
        "model": "mediapipe_face_mesh",
        "face_image": rgb_face,
        "visualization": visualization,
        "visualization_path": str(image_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "selected_regions": selected_regions or [],
        "show_full_mesh": show_full_mesh,
    }


def export_landmarks(landmarks: list[dict]) -> dict:
    """Prepare landmark coordinates for downstream geometric transformations."""
    return {
        "count": len(landmarks),
        "coordinates": landmarks,
    }
