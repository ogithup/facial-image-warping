"""Preprocessing stage for normalization and analysis readiness."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


OUTPUTS_DIR = Path("outputs")


def convert_bgr_to_rgb(image: dict) -> dict:
    """Convert BGR pixel data to RGB format."""
    rgb_pixels = cv2.cvtColor(image["pixels"], cv2.COLOR_BGR2RGB)
    return {
        **image,
        "pixels": rgb_pixels,
        "rgb_pixels": rgb_pixels,
        "color_space": "RGB",
    }


def convert_rgb_to_bgr(image: dict) -> dict:
    """Convert RGB pixel data back to BGR format."""
    bgr_pixels = cv2.cvtColor(image["pixels"], cv2.COLOR_RGB2BGR)
    return {
        **image,
        "pixels": bgr_pixels,
        "color_space": "BGR",
    }


def convert_to_grayscale(image: dict) -> dict:
    """Convert the current image payload into a single-channel grayscale image."""
    if image["color_space"] == "BGR":
        grayscale = cv2.cvtColor(image["pixels"], cv2.COLOR_BGR2GRAY)
    elif image["color_space"] == "RGB":
        grayscale = cv2.cvtColor(image["pixels"], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported color space for grayscale conversion: {image['color_space']}")

    return {
        **image,
        "pixels": grayscale,
        "shape": grayscale.shape,
        "color_space": "GRAYSCALE",
    }


def resize_to_standard(image: dict, target_size: tuple[int, int] = (512, 512)) -> dict:
    """Resize the current image payload to a standard resolution."""
    resized = cv2.resize(image["pixels"], target_size, interpolation=cv2.INTER_AREA)
    return {
        **image,
        "pixels": resized,
        "width": target_size[0],
        "height": target_size[1],
        "shape": resized.shape,
        "target_size": target_size,
    }


def normalize_pixel_values(image: dict) -> dict:
    """Normalize pixel values from 0-255 into 0-1 float space."""
    normalized = image["pixels"].astype(np.float32) / 255.0
    return {
        **image,
        "pixels": normalized,
        "dtype": str(normalized.dtype),
        "normalized": True,
        "pixel_range": (float(normalized.min()), float(normalized.max())),
    }


def compute_histogram(image: dict) -> dict:
    """Compute histogram data for grayscale or color images."""
    pixels = image["pixels"]
    histogram: dict[str, np.ndarray] = {}

    if pixels.ndim == 2:
        histogram["gray"] = cv2.calcHist([pixels.astype(np.float32)], [0], None, [256], [0, 1 if pixels.dtype.kind == "f" else 256])
    else:
        channel_names = ("blue", "green", "red") if image["color_space"] == "BGR" else ("red", "green", "blue")
        upper_bound = 1 if pixels.dtype.kind == "f" else 256
        for index, channel_name in enumerate(channel_names):
            histogram[channel_name] = cv2.calcHist([pixels.astype(np.float32)], [index], None, [256], [0, upper_bound])

    return histogram


def display_image_histogram(image: dict, output_path: str | Path | None = None) -> Path:
    """Render and save the image histogram using Matplotlib."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    histogram_path = Path(output_path) if output_path else OUTPUTS_DIR / "histogram.png"
    histogram = compute_histogram(image)

    plt.figure(figsize=(8, 4))
    for label, values in histogram.items():
        color = "gray" if label == "gray" else label
        plt.plot(values, color=color, label=label)
    plt.title("Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()
    return histogram_path


def save_processed_image(image: dict, output_path: str | Path) -> Path:
    """Save the current image payload into the outputs directory."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pixels = image["pixels"]

    if pixels.dtype.kind == "f":
        save_pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
    else:
        save_pixels = pixels

    if image["color_space"] == "RGB" and save_pixels.ndim == 3:
        save_pixels = cv2.cvtColor(save_pixels, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(str(output_file), save_pixels):
        raise ValueError(f"Failed to save processed image to {output_file}")

    return output_file


def preprocess_image(
    image: dict,
    target_size: tuple[int, int] = (512, 512),
    save_outputs: bool = True,
) -> dict:
    """Run Sprint 1 preprocessing steps and save artifacts to the outputs folder."""
    rgb_image = convert_bgr_to_rgb(image)
    resized_rgb = resize_to_standard(rgb_image, target_size=target_size)
    grayscale = convert_to_grayscale(resized_rgb)
    normalized = normalize_pixel_values(grayscale)

    histogram_path = display_image_histogram(normalized, OUTPUTS_DIR / "histogram.png")
    processed_image_path = OUTPUTS_DIR / "processed" / f"preprocessed_{Path(image['file_name']).stem}.png"
    if save_outputs:
        save_processed_image(normalized, processed_image_path)

    return {
        "image": normalized,
        "rgb_image": resized_rgb,
        "grayscale_image": grayscale,
        "histogram_path": str(histogram_path),
        "processed_image_path": str(processed_image_path),
        "status": "preprocessed",
    }
