"""Geometric facial manipulation using landmark-driven triangle warping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


WARPING_OUTPUT_DIR = Path("outputs/warping")
EXPRESSION_REGIONS = {
    "smile_enhancement": {
        "upper_lip": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409],
        "lower_lip": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        "mouth_corners": [61, 291],
    },
    "eyebrow_raising": {
        "left_brow": [70, 63, 105, 66, 107],
        "right_brow": [300, 293, 334, 296, 336],
    },
    "lip_widening": {
        "mouth_corners": [61, 291],
        "upper_lip": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415],
        "lower_lip": [95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    },
    "face_slimming": {
        "left_cheek": [234, 93, 132, 58, 172, 136],
        "right_cheek": [454, 323, 361, 288, 397, 365],
        "jawline": [149, 176, 148, 152, 377, 400, 378, 379],
    },
}


def _ensure_bgr_uint8(face_image: dict) -> dict:
    """Convert supported image payloads to uint8 BGR for OpenCV warping."""
    pixels = face_image["pixels"]
    if pixels.dtype.kind == "f":
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)

    color_space = face_image.get("color_space", "BGR")
    if color_space == "BGR":
        bgr_pixels = pixels
    elif color_space == "RGB":
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    elif color_space == "GRAYSCALE":
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported color space for warping: {color_space}")

    return {
        **face_image,
        "pixels": bgr_pixels,
        "shape": bgr_pixels.shape,
        "color_space": "BGR",
        "dtype": str(bgr_pixels.dtype),
    }


def _landmarks_to_points(landmarks: list[dict]) -> list[tuple[int, int]]:
    """Convert landmark dictionaries into OpenCV-style integer point tuples."""
    return [(int(landmark["x"]), int(landmark["y"])) for landmark in landmarks]


def create_target_landmarks(
    landmarks: list[dict],
    transformation: str,
    intensity: float = 0.5,
) -> list[dict]:
    """Create target landmark positions for expression editing."""
    if transformation not in EXPRESSION_REGIONS:
        supported = ", ".join(sorted(EXPRESSION_REGIONS))
        raise ValueError(f"Unsupported transformation '{transformation}'. Supported: {supported}")

    intensity = float(np.clip(intensity, 0.0, 1.0))
    targets = [dict(landmark) for landmark in landmarks]

    def move(index: int, dx: float = 0.0, dy: float = 0.0) -> None:
        targets[index]["x"] = int(round(targets[index]["x"] + dx))
        targets[index]["y"] = int(round(targets[index]["y"] + dy))

    if transformation == "smile_enhancement":
        corners = EXPRESSION_REGIONS[transformation]["mouth_corners"]
        upper_lip = EXPRESSION_REGIONS[transformation]["upper_lip"]
        lower_lip = EXPRESSION_REGIONS[transformation]["lower_lip"]
        move(corners[0], dx=-14 * intensity, dy=-10 * intensity)
        move(corners[1], dx=14 * intensity, dy=-10 * intensity)
        for index in upper_lip:
            move(index, dy=-4 * intensity)
        for index in lower_lip:
            move(index, dy=2 * intensity)
    elif transformation == "eyebrow_raising":
        for index in EXPRESSION_REGIONS[transformation]["left_brow"]:
            move(index, dy=-12 * intensity)
        for index in EXPRESSION_REGIONS[transformation]["right_brow"]:
            move(index, dy=-12 * intensity)
    elif transformation == "lip_widening":
        left_corner, right_corner = EXPRESSION_REGIONS[transformation]["mouth_corners"]
        move(left_corner, dx=-18 * intensity)
        move(right_corner, dx=18 * intensity)
        for index in EXPRESSION_REGIONS[transformation]["upper_lip"]:
            delta = -6 * intensity if targets[index]["x"] < landmarks[right_corner]["x"] else 6 * intensity
            move(index, dx=delta * 0.25)
        for index in EXPRESSION_REGIONS[transformation]["lower_lip"]:
            delta = -6 * intensity if targets[index]["x"] < landmarks[right_corner]["x"] else 6 * intensity
            move(index, dx=delta * 0.25)
    elif transformation == "face_slimming":
        for index in EXPRESSION_REGIONS[transformation]["left_cheek"]:
            move(index, dx=10 * intensity)
        for index in EXPRESSION_REGIONS[transformation]["right_cheek"]:
            move(index, dx=-10 * intensity)
        for index in EXPRESSION_REGIONS[transformation]["jawline"]:
            direction = -1 if landmarks[index]["x"] > landmarks[152]["x"] else 1
            move(index, dx=direction * 6 * intensity)

    return targets


def apply_delaunay_triangulation(
    image_shape: tuple[int, ...],
    landmarks: list[dict],
) -> list[tuple[int, int, int]]:
    """Generate triangle connectivity from landmarks using Delaunay triangulation."""
    height, width = image_shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    points = _landmarks_to_points(landmarks)
    point_to_index = {point: index for index, point in enumerate(points)}

    for point in points:
        x = min(max(point[0], 0), width - 1)
        y = min(max(point[1], 0), height - 1)
        subdiv.insert((float(x), float(y)))

    triangles: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for triangle in subdiv.getTriangleList():
        vertices = [
            (int(round(triangle[0])), int(round(triangle[1]))),
            (int(round(triangle[2])), int(round(triangle[3]))),
            (int(round(triangle[4])), int(round(triangle[5]))),
        ]
        if any(vertex not in point_to_index for vertex in vertices):
            continue
        indices = tuple(sorted(point_to_index[vertex] for vertex in vertices))
        if len(set(indices)) < 3 or indices in seen:
            continue
        seen.add(indices)
        triangles.append(indices)

    return triangles


def warp_triangle(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    source_triangle: list[tuple[int, int]],
    target_triangle: list[tuple[int, int]],
) -> None:
    """Warp one triangle from source to destination using affine transformation."""
    src_rect = cv2.boundingRect(np.float32([source_triangle]))
    dst_rect = cv2.boundingRect(np.float32([target_triangle]))

    src_rect_points = [(point[0] - src_rect[0], point[1] - src_rect[1]) for point in source_triangle]
    dst_rect_points = [(point[0] - dst_rect[0], point[1] - dst_rect[1]) for point in target_triangle]

    src_crop = source_image[src_rect[1] : src_rect[1] + src_rect[3], src_rect[0] : src_rect[0] + src_rect[2]]
    warp_matrix = cv2.getAffineTransform(np.float32(src_rect_points), np.float32(dst_rect_points))
    warped_patch = cv2.warpAffine(
        src_crop,
        warp_matrix,
        (dst_rect[2], dst_rect[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_rect_points), (1.0, 1.0, 1.0), 16, 0)

    destination_slice = destination_image[
        dst_rect[1] : dst_rect[1] + dst_rect[3],
        dst_rect[0] : dst_rect[0] + dst_rect[2],
    ]
    destination_slice *= 1.0 - mask
    destination_slice += warped_patch.astype(np.float32) * mask
    destination_image[
        dst_rect[1] : dst_rect[1] + dst_rect[3],
        dst_rect[0] : dst_rect[0] + dst_rect[2],
    ] = destination_slice


def _save_bgr_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save a BGR image with Unicode-safe file writing."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoded, buffer = cv2.imencode(output_file.suffix or ".png", image)
    if not encoded:
        raise ValueError(f"Failed to encode image for {output_file}")
    buffer.tofile(output_file)
    return output_file


def apply_expression_warp(
    face_image: dict,
    landmarks: list[dict],
    transformation: str = "smile_enhancement",
    intensity: float = 0.5,
    save_outputs: bool = True,
) -> dict:
    """Apply a landmark-driven geometric expression warp to a face image."""
    prepared_face = _ensure_bgr_uint8(face_image)
    source_pixels = prepared_face["pixels"]
    target_landmarks = create_target_landmarks(landmarks, transformation=transformation, intensity=intensity)
    triangles = apply_delaunay_triangulation(source_pixels.shape, landmarks)

    warped_pixels = source_pixels.astype(np.float32).copy()
    source_points = _landmarks_to_points(landmarks)
    target_points = _landmarks_to_points(target_landmarks)

    for triangle in triangles:
        src_triangle = [source_points[index] for index in triangle]
        dst_triangle = [target_points[index] for index in triangle]
        warp_triangle(source_pixels, warped_pixels, src_triangle, dst_triangle)

    warped_pixels = np.clip(warped_pixels, 0, 255).astype(np.uint8)
    comparison = cv2.hconcat([source_pixels, warped_pixels])

    stem = Path(prepared_face.get("file_name", "face.png")).stem
    warped_path = WARPING_OUTPUT_DIR / f"{stem}_{transformation}.png"
    comparison_path = WARPING_OUTPUT_DIR / f"{stem}_{transformation}_comparison.png"
    if save_outputs:
        _save_bgr_image(warped_pixels, warped_path)
        _save_bgr_image(comparison, comparison_path)

    return {
        "image": {
            **prepared_face,
            "pixels": warped_pixels,
            "shape": warped_pixels.shape,
            "warp_applied": True,
            "color_space": "BGR",
            "dtype": str(warped_pixels.dtype),
        },
        "operation": transformation,
        "landmark_count": len(landmarks),
        "target_landmarks": target_landmarks,
        "triangles": triangles,
        "intensity": float(np.clip(intensity, 0.0, 1.0)),
        "warped_image_path": str(warped_path),
        "comparison_image_path": str(comparison_path),
    }


def generate_delaunay_mesh(landmarks: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Backward-compatible mesh helper retained for earlier scaffold references."""
    indexed = [{"index": index, "x": int(point[0]), "y": int(point[1])} for index, point in enumerate(landmarks)]
    if not indexed:
        return []
    width = max(point["x"] for point in indexed) + 1
    height = max(point["y"] for point in indexed) + 1
    return apply_delaunay_triangulation((height, width, 3), indexed)
