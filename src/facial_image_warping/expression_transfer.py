"""Reference-driven facial expression transfer utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facial_image_warping.geometric_warping import apply_delaunay_triangulation, warp_triangle
from facial_image_warping.landmark_detection import FACE_REGIONS


TRANSFER_OUTPUT_DIR = Path("outputs/transfer")
DEFAULT_TRANSFER_REGIONS = ["eyebrows", "eyes", "lips"]
STABLE_ANCHOR_INDICES = [10, 54, 67, 103, 109, 127, 152, 162, 234, 251, 297, 323, 356, 389, 454]
INNER_MOUTH_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
TRANSFER_METHODS = {
    "safe_classical",
    "tps",
    "expression_coefficients",
}


def _estimate_similarity_matrix(source_landmarks: list[dict], reference_landmarks: list[dict]) -> np.ndarray:
    """Estimate a global similarity transform that aligns reference geometry to the source face."""
    source_points = np.float32([(landmark["x"], landmark["y"]) for landmark in source_landmarks])
    reference_points = np.float32([(landmark["x"], landmark["y"]) for landmark in reference_landmarks])
    matrix, _ = cv2.estimateAffinePartial2D(reference_points, source_points, method=cv2.LMEDS)
    if matrix is None:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return matrix.astype(np.float32)


def _transform_landmarks(landmarks: list[dict], matrix: np.ndarray) -> list[dict]:
    """Apply a 2D affine transform to landmark coordinates."""
    transformed: list[dict] = []
    for landmark in landmarks:
        x, y = landmark["x"], landmark["y"]
        mapped_x = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
        mapped_y = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
        transformed.append(
            {
                **landmark,
                "x": int(round(mapped_x)),
                "y": int(round(mapped_y)),
            }
        )
    return transformed


def _compute_max_displacement(source_landmarks: list[dict]) -> float:
    """Bound local transfer strength to reduce foldovers and stretched triangles."""
    x_values = [landmark["x"] for landmark in source_landmarks]
    y_values = [landmark["y"] for landmark in source_landmarks]
    face_span = max(max(x_values) - min(x_values), max(y_values) - min(y_values), 1)
    return float(face_span) * 0.12


def _align_reference_landmarks_to_source(
    source_landmarks: list[dict],
    reference_landmarks: list[dict],
) -> list[dict]:
    """Align reference landmarks into the source face coordinate frame."""
    return _transform_landmarks(reference_landmarks, _estimate_similarity_matrix(source_landmarks, reference_landmarks))


def _resize_face_image_and_landmarks(
    face_image: dict,
    landmarks: list[dict],
    target_shape: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    """Resize a face image and its landmark coordinates onto a target canvas size."""
    target_height, target_width = target_shape[:2]
    source_height, source_width = face_image["pixels"].shape[:2]
    if source_height == target_height and source_width == target_width:
        return face_image, landmarks

    scale_x = target_width / max(source_width, 1)
    scale_y = target_height / max(source_height, 1)
    resized_pixels = cv2.resize(
        face_image["pixels"],
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_image = {
        **face_image,
        "pixels": resized_pixels,
        "width": target_width,
        "height": target_height,
        "shape": resized_pixels.shape,
    }
    resized_landmarks: list[dict] = []
    for landmark in landmarks:
        resized_landmarks.append(
            {
                **landmark,
                "x": int(round(landmark["x"] * scale_x)),
                "y": int(round(landmark["y"] * scale_y)),
            }
        )
    return resized_image, resized_landmarks


def _build_region_blend_mask(
    image_shape: tuple[int, ...],
    source_landmarks: list[dict],
    target_landmarks: list[dict],
    regions: list[str],
) -> np.ndarray:
    """Build a soft union mask covering both source and target expression regions."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    source_by_index = {landmark["index"]: landmark for landmark in source_landmarks}
    target_by_index = {landmark["index"]: landmark for landmark in target_landmarks}
    for region in regions:
        region_indices = FACE_REGIONS.get(region, [])
        for landmark_lookup in (source_by_index, target_by_index):
            region_points = [
                (int(landmark_lookup[index]["x"]), int(landmark_lookup[index]["y"]))
                for index in region_indices
                if index in landmark_lookup
            ]
            if len(region_points) < 3:
                continue
            hull = cv2.convexHull(np.array(region_points, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull, 255)

    kernel = np.ones((21, 21), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask


def _build_polygon_mask(
    image_shape: tuple[int, ...],
    landmarks: list[dict],
    indices: list[int],
    dilation: int = 5,
    blur: int = 15,
) -> np.ndarray:
    """Build a soft mask from one landmark polygon."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    landmark_by_index = {landmark["index"]: landmark for landmark in landmarks}
    polygon_points = [
        (int(landmark_by_index[index]["x"]), int(landmark_by_index[index]["y"]))
        for index in indices
        if index in landmark_by_index
    ]
    if len(polygon_points) < 3:
        return mask
    polygon = np.array(polygon_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 255)
    if dilation > 0:
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if blur > 0:
        blur = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask


def _blend_warped_face(
    source_pixels: np.ndarray,
    warped_pixels: np.ndarray,
    source_landmarks: list[dict],
    target_landmarks: list[dict],
    regions: list[str],
) -> np.ndarray:
    """Blend a warped face back onto the source with feathered regional replacement."""
    blend_mask = _build_region_blend_mask(
        source_pixels.shape,
        source_landmarks,
        target_landmarks,
        regions,
    )
    if np.count_nonzero(blend_mask) == 0:
        return warped_pixels
    alpha = (blend_mask.astype(np.float32) / 255.0)[..., None]
    blended = warped_pixels.astype(np.float32) * alpha + source_pixels.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _blend_mouth_interior_from_reference(
    base_pixels: np.ndarray,
    reference_face_image: dict | None,
    source_landmarks: list[dict],
    reference_landmarks: list[dict],
    target_landmarks: list[dict],
    blend_factor: float,
) -> np.ndarray:
    """Transfer visible mouth interior like teeth and mouth cavity when the reference mouth is open."""
    if reference_face_image is None:
        return base_pixels

    source_coeffs = extract_expression_coefficients(source_landmarks)
    reference_coeffs = extract_expression_coefficients(reference_landmarks)
    target_coeffs = extract_expression_coefficients(target_landmarks)
    if reference_coeffs["mouth_open"] < 0.045 or target_coeffs["mouth_open"] <= source_coeffs["mouth_open"]:
        return base_pixels

    prepared_reference = _ensure_bgr_uint8(reference_face_image)
    resized_reference_image, resized_reference_landmarks = _resize_face_image_and_landmarks(
        prepared_reference,
        reference_landmarks,
        base_pixels.shape,
    )
    warped_reference_face = _warp_image_with_tps(
        resized_reference_image["pixels"],
        resized_reference_landmarks,
        target_landmarks,
        regions=["lips"],
    )

    mouth_mask = _build_polygon_mask(
        base_pixels.shape,
        target_landmarks,
        INNER_MOUTH_INDICES,
        dilation=7,
        blur=21,
    )
    if np.count_nonzero(mouth_mask) == 0:
        return base_pixels

    openness_gain = min(max((reference_coeffs["mouth_open"] - source_coeffs["mouth_open"]) / 0.08, 0.0), 1.0)
    alpha = (mouth_mask.astype(np.float32) / 255.0)[..., None] * min(0.9, 0.35 + 0.55 * blend_factor * openness_gain)
    blended = warped_reference_face.astype(np.float32) * alpha + base_pixels.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _tps_kernel(distances_squared: np.ndarray) -> np.ndarray:
    """Evaluate the Thin Plate Spline radial basis kernel."""
    safe = np.maximum(distances_squared, 1e-8)
    values = safe * np.log(safe)
    values[distances_squared <= 1e-8] = 0.0
    return values


def _deduplicate_control_points(points: list[tuple[float, float]], values: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate TPS control points while preserving order."""
    deduped_points: list[tuple[float, float]] = []
    deduped_values: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for point, value in zip(points, values):
        key = (int(round(point[0] * 1000)), int(round(point[1] * 1000)))
        if key in seen:
            continue
        seen.add(key)
        deduped_points.append(point)
        deduped_values.append(value)
    return np.asarray(deduped_points, dtype=np.float32), np.asarray(deduped_values, dtype=np.float32)


def _build_tps_control_points(
    source_landmarks: list[dict],
    target_landmarks: list[dict],
    regions: list[str],
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble regional TPS control points plus stable anchors and frame anchors."""
    moved_indices = sorted({index for region in regions for index in FACE_REGIONS.get(region, [])})
    source_points = [(float(source_landmarks[index]["x"]), float(source_landmarks[index]["y"])) for index in moved_indices]
    target_points = [(float(target_landmarks[index]["x"]), float(target_landmarks[index]["y"])) for index in moved_indices]

    for index in STABLE_ANCHOR_INDICES:
        if index >= len(source_landmarks) or index in moved_indices:
            continue
        point = (float(source_landmarks[index]["x"]), float(source_landmarks[index]["y"]))
        source_points.append(point)
        target_points.append(point)

    height, width = image_shape[:2]
    frame_anchor_points = [
        (0.0, 0.0),
        (width - 1.0, 0.0),
        (0.0, height - 1.0),
        (width - 1.0, height - 1.0),
        (width / 2.0, 0.0),
        (width / 2.0, height - 1.0),
        (0.0, height / 2.0),
        (width - 1.0, height / 2.0),
    ]
    for point in frame_anchor_points:
        source_points.append(point)
        target_points.append(point)

    return _deduplicate_control_points(source_points, target_points)


def _fit_tps_model(control_points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Thin Plate Spline mapping from control points to target values."""
    count = control_points.shape[0]
    pairwise_diff = control_points[:, None, :] - control_points[None, :, :]
    distances_squared = np.sum(pairwise_diff * pairwise_diff, axis=2)
    kernel = _tps_kernel(distances_squared)
    affine = np.concatenate([np.ones((count, 1), dtype=np.float32), control_points], axis=1)

    top = np.concatenate([kernel, affine], axis=1)
    bottom = np.concatenate([affine.T, np.zeros((3, 3), dtype=np.float32)], axis=1)
    system = np.concatenate([top, bottom], axis=0)
    targets = np.concatenate([values, np.zeros((3, 2), dtype=np.float32)], axis=0)

    parameters = np.linalg.pinv(system) @ targets
    return parameters[:count], parameters[count:]


def _evaluate_tps_mapping(
    query_points: np.ndarray,
    control_points: np.ndarray,
    weights: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted TPS mapping at arbitrary query points."""
    pairwise_diff = query_points[:, None, :] - control_points[None, :, :]
    distances_squared = np.sum(pairwise_diff * pairwise_diff, axis=2)
    kernel = _tps_kernel(distances_squared)
    query_affine = np.concatenate([np.ones((query_points.shape[0], 1), dtype=np.float32), query_points], axis=1)
    return kernel @ weights + query_affine @ affine


def _warp_image_with_tps(
    source_pixels: np.ndarray,
    source_landmarks: list[dict],
    target_landmarks: list[dict],
    regions: list[str],
) -> np.ndarray:
    """Warp the source face with inverse TPS remapping for smoother expression transfer."""
    source_points, target_points = _build_tps_control_points(
        source_landmarks,
        target_landmarks,
        regions=regions,
        image_shape=source_pixels.shape,
    )
    weights, affine = _fit_tps_model(target_points, source_points)

    height, width = source_pixels.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    query_points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    mapped_points = _evaluate_tps_mapping(query_points, target_points, weights, affine)

    map_x = np.clip(mapped_points[:, 0], 0, width - 1).reshape(height, width).astype(np.float32)
    map_y = np.clip(mapped_points[:, 1], 0, height - 1).reshape(height, width).astype(np.float32)
    return cv2.remap(
        source_pixels,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _warp_image_with_triangles(
    source_pixels: np.ndarray,
    source_landmarks: list[dict],
    target_landmarks: list[dict],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Warp an image with classic Delaunay triangle warping."""
    triangles = apply_delaunay_triangulation(source_pixels.shape, source_landmarks)
    warped_pixels = source_pixels.astype(np.float32).copy()
    source_points = [(int(landmark["x"]), int(landmark["y"])) for landmark in source_landmarks]
    target_points = [(int(landmark["x"]), int(landmark["y"])) for landmark in target_landmarks]
    for triangle in triangles:
        src_triangle = [source_points[index] for index in triangle]
        dst_triangle = [target_points[index] for index in triangle]
        warp_triangle(source_pixels, warped_pixels, src_triangle, dst_triangle)
    return np.clip(warped_pixels, 0, 255).astype(np.uint8), triangles


def _landmark_point(landmarks: list[dict], index: int) -> np.ndarray:
    """Read one landmark as a float vector."""
    return np.array([float(landmarks[index]["x"]), float(landmarks[index]["y"])], dtype=np.float32)


def _mean_point(landmarks: list[dict], indices: list[int]) -> np.ndarray:
    """Average a landmark region into a single 2D point."""
    points = np.array([_landmark_point(landmarks, index) for index in indices], dtype=np.float32)
    return np.mean(points, axis=0)


def _point_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance helper."""
    return float(np.linalg.norm(a - b))


def _safe_divide(numerator: float, denominator: float) -> float:
    """Avoid zero-division in coefficient normalization."""
    return float(numerator / denominator) if abs(denominator) > 1e-6 else 0.0


def extract_expression_coefficients(landmarks: list[dict]) -> dict:
    """Extract compact expression coefficients from facial landmarks."""
    left_eye_center = _mean_point(landmarks, [33, 133, 159, 145])
    right_eye_center = _mean_point(landmarks, [362, 263, 386, 374])
    eye_distance = max(_point_distance(left_eye_center, right_eye_center), 1.0)

    left_eye_open = _safe_divide(_point_distance(_landmark_point(landmarks, 159), _landmark_point(landmarks, 145)), eye_distance)
    right_eye_open = _safe_divide(_point_distance(_landmark_point(landmarks, 386), _landmark_point(landmarks, 374)), eye_distance)
    mouth_width = _safe_divide(_point_distance(_landmark_point(landmarks, 61), _landmark_point(landmarks, 291)), eye_distance)
    mouth_open = _safe_divide(_point_distance(_landmark_point(landmarks, 13), _landmark_point(landmarks, 14)), eye_distance)

    brow_left_y = _mean_point(landmarks, [70, 63, 105, 66, 107])[1]
    brow_right_y = _mean_point(landmarks, [300, 293, 334, 296, 336])[1]
    left_eye_y = left_eye_center[1]
    right_eye_y = right_eye_center[1]

    brow_raise_left = _safe_divide(left_eye_y - brow_left_y, eye_distance)
    brow_raise_right = _safe_divide(right_eye_y - brow_right_y, eye_distance)
    mouth_corner_left = _safe_divide(_landmark_point(landmarks, 61)[1] - _landmark_point(landmarks, 13)[1], eye_distance)
    mouth_corner_right = _safe_divide(_landmark_point(landmarks, 291)[1] - _landmark_point(landmarks, 13)[1], eye_distance)

    return {
        "eye_distance": eye_distance,
        "left_eye_open": left_eye_open,
        "right_eye_open": right_eye_open,
        "mouth_width": mouth_width,
        "mouth_open": mouth_open,
        "brow_raise_left": brow_raise_left,
        "brow_raise_right": brow_raise_right,
        "mouth_corner_left": mouth_corner_left,
        "mouth_corner_right": mouth_corner_right,
    }


def create_expression_coefficient_targets(
    source_landmarks: list[dict],
    reference_landmarks: list[dict],
    blend_factor: float = 0.7,
) -> list[dict]:
    """Approximate reenactment by transferring normalized expression coefficients onto the source."""
    source_coeffs = extract_expression_coefficients(source_landmarks)
    reference_coeffs = extract_expression_coefficients(reference_landmarks)
    eye_distance = source_coeffs["eye_distance"]
    targets = [dict(landmark) for landmark in source_landmarks]

    def shift(indices: list[int], dx: float = 0.0, dy: float = 0.0) -> None:
        for index in indices:
            targets[index]["x"] = int(round(targets[index]["x"] + dx))
            targets[index]["y"] = int(round(targets[index]["y"] + dy))

    brow_left_delta = (reference_coeffs["brow_raise_left"] - source_coeffs["brow_raise_left"]) * eye_distance * blend_factor
    brow_right_delta = (reference_coeffs["brow_raise_right"] - source_coeffs["brow_raise_right"]) * eye_distance * blend_factor
    mouth_open_delta = (reference_coeffs["mouth_open"] - source_coeffs["mouth_open"]) * eye_distance * blend_factor
    mouth_width_delta = (reference_coeffs["mouth_width"] - source_coeffs["mouth_width"]) * eye_distance * blend_factor
    left_corner_delta = (reference_coeffs["mouth_corner_left"] - source_coeffs["mouth_corner_left"]) * eye_distance * blend_factor
    right_corner_delta = (reference_coeffs["mouth_corner_right"] - source_coeffs["mouth_corner_right"]) * eye_distance * blend_factor

    shift([70, 63, 105, 66, 107], dy=-brow_left_delta)
    shift([300, 293, 334, 296, 336], dy=-brow_right_delta)

    left_lip = [61, 78, 80, 81, 82, 84, 87, 91, 95, 146, 178, 181, 185]
    right_lip = [291, 308, 310, 311, 312, 314, 317, 321, 324, 375, 402, 405, 409]
    upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 13]
    lower_lip = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 14]

    shift(left_lip, dx=-mouth_width_delta * 0.4)
    shift(right_lip, dx=mouth_width_delta * 0.4)
    shift(upper_lip, dy=-mouth_open_delta * 0.55)
    shift(lower_lip, dy=mouth_open_delta * 0.55)
    shift([61], dy=left_corner_delta * 0.9)
    shift([291], dy=right_corner_delta * 0.9)

    return targets


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
    aligned_reference_landmarks = _align_reference_landmarks_to_source(source_landmarks, reference_landmarks)
    max_displacement = _compute_max_displacement(source_landmarks)

    for region in regions:
        if region not in FACE_REGIONS:
            raise ValueError(f"Unknown transfer region requested: {region}")
        for index in FACE_REGIONS[region]:
            source = source_landmarks[index]
            reference = aligned_reference_landmarks[index]
            delta_x = (reference["x"] - source["x"]) * blend_factor
            delta_y = (reference["y"] - source["y"]) * blend_factor
            displacement = float(np.hypot(delta_x, delta_y))
            if displacement > max_displacement:
                scale = max_displacement / displacement
                delta_x *= scale
                delta_y *= scale
            targets[index]["x"] = int(round(source["x"] + delta_x))
            targets[index]["y"] = int(round(source["y"] + delta_y))

    return targets


def apply_reference_expression_transfer(
    source_face_image: dict,
    source_landmarks: list[dict],
    reference_landmarks: list[dict],
    blend_factor: float = 0.7,
    regions: list[str] | None = None,
    save_outputs: bool = True,
    method: str = "tps",
    reference_face_image: dict | None = None,
) -> dict:
    """Warp a source face toward the expression geometry of a reference face."""
    prepared_face = _ensure_bgr_uint8(source_face_image)
    source_pixels = prepared_face["pixels"]
    regions = regions or DEFAULT_TRANSFER_REGIONS
    if method not in TRANSFER_METHODS:
        supported = ", ".join(sorted(TRANSFER_METHODS))
        raise ValueError(f"Unsupported transfer method '{method}'. Supported: {supported}")

    if method == "expression_coefficients":
        target_landmarks = create_expression_coefficient_targets(
            source_landmarks,
            reference_landmarks,
            blend_factor=blend_factor,
        )
    else:
        target_landmarks = create_expression_transfer_targets(
            source_landmarks,
            reference_landmarks,
            blend_factor=blend_factor,
            regions=regions,
        )

    if method == "safe_classical":
        warped_pixels, triangles = _warp_image_with_triangles(source_pixels, source_landmarks, target_landmarks)
        warped_pixels = _blend_warped_face(
            source_pixels,
            warped_pixels,
            source_landmarks,
            target_landmarks,
            regions,
        )
    else:
        warped_pixels = _warp_image_with_tps(
            source_pixels,
            source_landmarks,
            target_landmarks,
            regions=regions,
        )
        warped_pixels = _blend_warped_face(
            source_pixels,
            warped_pixels,
            source_landmarks,
            target_landmarks,
            regions,
        )
        triangles = None
    warped_pixels = _blend_mouth_interior_from_reference(
        warped_pixels,
        reference_face_image=reference_face_image,
        source_landmarks=source_landmarks,
        reference_landmarks=reference_landmarks,
        target_landmarks=target_landmarks,
        blend_factor=blend_factor,
    )
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
        "regions": regions,
        "triangles": triangles,
        "warping_method": method,
        "warped_image_path": str(transferred_path),
        "comparison_image_path": str(comparison_path),
        "explanation": [
            "MediaPipe facial landmarks are extracted from both source and reference faces.",
            "Expression-sensitive regions are aligned and blended toward the reference geometry.",
            "Open-mouth references can transfer inner-mouth texture like teeth through a dedicated mouth patch blend.",
            f"Transfer method: {method}.",
        ],
    }
