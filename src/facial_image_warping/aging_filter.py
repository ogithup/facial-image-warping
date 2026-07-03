"""Frequency-inspired aging and de-aging simulation filters."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


AGING_OUTPUT_DIR = Path("outputs/aging")


def _ensure_bgr_uint8(image: dict) -> dict:
    """Convert supported image payloads to uint8 BGR for filtering."""
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
        raise ValueError(f"Unsupported color space for aging filter: {color_space}")

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


def _high_pass_texture(image: np.ndarray, sigma: float) -> np.ndarray:
    """Extract high-frequency texture by subtracting a Gaussian blur."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.subtract(image, blurred)


def _edge_mask(image: np.ndarray) -> np.ndarray:
    """Create an edge mask to preserve important facial contours."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 60, 160)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    normalized = edges.astype(np.float32) / 255.0
    return cv2.merge([normalized, normalized, normalized])


def _local_contrast_boost(image: np.ndarray, clip_limit: float) -> np.ndarray:
    """Boost local contrast using CLAHE in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    boosted_l = clahe.apply(l_channel)
    boosted = cv2.merge([boosted_l, a_channel, b_channel])
    return cv2.cvtColor(boosted, cv2.COLOR_LAB2BGR)


def _preserve_facial_features(original: np.ndarray, filtered: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """Blend filtered result with original edges to preserve eyes, lips, and jaw."""
    mask = _edge_mask(original)
    original_float = original.astype(np.float32)
    filtered_float = filtered.astype(np.float32)
    blended = filtered_float * (1.0 - mask * strength) + original_float * (mask * strength)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_aging_filter(image: dict, intensity: float = 0.5) -> dict:
    """Simulate aging using high-frequency enhancement and contrast shaping."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    prepared = _ensure_bgr_uint8(image)
    base = prepared["pixels"]

    texture = _high_pass_texture(base, sigma=2.5 + intensity)
    wrinkle_boost = cv2.addWeighted(base, 1.0, texture, 0.8 * intensity, 0)
    contrast_boosted = _local_contrast_boost(wrinkle_boost, clip_limit=1.5 + intensity)
    sharpened = cv2.addWeighted(contrast_boosted, 1.0 + 0.25 * intensity, cv2.GaussianBlur(contrast_boosted, (0, 0), 1.2), -0.25 * intensity, 0)
    aged = _preserve_facial_features(base, sharpened, strength=0.85)

    stem = Path(prepared.get("file_name", "face.png")).stem
    output_path = AGING_OUTPUT_DIR / f"{stem}_aged.png"
    _save_bgr_image(aged, output_path)

    return {
        "image": {**prepared, "pixels": aged, "shape": aged.shape, "color_space": "BGR", "dtype": str(aged.dtype), "aging_intensity": intensity},
        "mode": "aging",
        "intensity": intensity,
        "output_path": str(output_path),
        "filter_explanation": [
            "High-pass texture extraction increased wrinkle-like micro-contrast.",
            "CLAHE slightly boosted local contrast for older skin appearance.",
            "Edge-preserving blend kept strong facial contours from over-degrading.",
        ],
    }


def apply_deaging_filter(image: dict, intensity: float = 0.5) -> dict:
    """Simulate de-aging using low-pass smoothing with edge preservation."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    prepared = _ensure_bgr_uint8(image)
    base = prepared["pixels"]

    sigma = 1.5 + 2.5 * intensity
    smoothed = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bilateral = cv2.bilateralFilter(smoothed, d=9, sigmaColor=35 + int(40 * intensity), sigmaSpace=35 + int(40 * intensity))
    softened = cv2.addWeighted(base, 0.3, bilateral, 0.7, 0)
    deaged = _preserve_facial_features(base, softened, strength=0.95)

    stem = Path(prepared.get("file_name", "face.png")).stem
    output_path = AGING_OUTPUT_DIR / f"{stem}_deaged.png"
    _save_bgr_image(deaged, output_path)

    return {
        "image": {**prepared, "pixels": deaged, "shape": deaged.shape, "color_space": "BGR", "dtype": str(deaged.dtype), "deaging_intensity": intensity},
        "mode": "deaging",
        "intensity": intensity,
        "output_path": str(output_path),
        "filter_explanation": [
            "Low-pass smoothing reduced high-frequency skin texture.",
            "Bilateral filtering softened skin while respecting local edges.",
            "Edge-preserving blend protected eyes, lips, and jawline from over-blur.",
        ],
    }
