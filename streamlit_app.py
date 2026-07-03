"""Streamlit GUI for facial image warping, aging, and evaluation."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import csv
import tempfile

import cv2
import numpy as np
import streamlit as st

from app import run_analysis_pipeline


TRANSFORMATION_OPTIONS = {
    "Smile Enhancement": "smile_enhancement",
    "Eyebrow Raising": "eyebrow_raising",
    "Lip Widening": "lip_widening",
    "Face Slimming": "face_slimming",
    "Aging": "aging",
    "De-Aging": "de-aging",
}
REGION_OPTIONS = ["eyes", "eyebrows", "nose", "lips", "jawline", "cheeks"]


def _image_dict_to_rgb(image: dict) -> np.ndarray:
    pixels = image["pixels"]
    if pixels.dtype.kind == "f":
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)

    color_space = image.get("color_space", "BGR")
    if color_space == "RGB":
        return pixels
    if color_space == "BGR":
        return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    if color_space == "GRAYSCALE":
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Unsupported color space for display: {color_space}")


def _spectrum_to_rgb(spectrum: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _build_csv_payload(result: dict) -> str:
    metrics = result["metrics"]
    rows = [
        {"metric": "mse", "value": metrics["mse"]},
        {"metric": "psnr", "value": metrics["psnr"]},
        {"metric": "ssim", "value": metrics["ssim"]},
        {"metric": "mean_absolute_difference", "value": metrics["mean_absolute_difference"]},
        {"metric": "max_absolute_difference", "value": metrics["max_absolute_difference"]},
        {"metric": "original_high_low_ratio", "value": metrics["original_high_low_ratio"]},
        {"metric": "transformed_high_low_ratio", "value": metrics["transformed_high_low_ratio"]},
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["metric", "value"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Facial Image Warping", layout="wide")
    st.title("Facial Image Warping, Aging, and Expression Transformation")
    st.caption("Sprint 8 Streamlit GUI: upload an image, choose a transformation, inspect spectra, and review quantitative metrics.")

    uploaded_file = st.file_uploader("Upload a frontal face image", type=["jpg", "jpeg", "png"])
    transformation_label = st.selectbox("Transformation", list(TRANSFORMATION_OPTIONS.keys()))
    intensity = st.slider("Intensity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    show_landmarks = st.checkbox("Show facial landmarks", value=False)
    selected_regions = st.multiselect(
        "Highlighted landmark regions",
        REGION_OPTIONS,
        default=["eyes", "nose", "lips"],
        disabled=not show_landmarks,
    )

    if uploaded_file is None:
        st.info("Upload a JPG or PNG face image to start the pipeline.")
        return

    suffix = Path(uploaded_file.name).suffix or ".png"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        result = run_analysis_pipeline(
            temp_path,
            transformation=TRANSFORMATION_OPTIONS[transformation_label],
            intensity=intensity,
            show_landmarks=show_landmarks,
            selected_regions=selected_regions or ["eyes", "nose", "lips"],
        )

        original_face = result["face_detection"]["face_image"]
        transformed_image = result["transformation"]["image"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Face ROI")
            st.image(_image_dict_to_rgb(original_face), use_container_width=True)
        with col2:
            st.subheader("Transformed Result")
            st.image(_image_dict_to_rgb(transformed_image), use_container_width=True)

        if show_landmarks and result["landmarks"] is not None:
            st.subheader("Landmark Visualization")
            st.image(_image_dict_to_rgb(result["landmarks"]["visualization"]), use_container_width=True)

        st.subheader("Fourier Magnitude Spectrum")
        spectrum_col1, spectrum_col2 = st.columns(2)
        with spectrum_col1:
            st.image(_spectrum_to_rgb(result["original_frequency"]["magnitude_spectrum"]), caption="Original Spectrum", use_container_width=True)
        with spectrum_col2:
            st.image(_spectrum_to_rgb(result["transformed_frequency"]["magnitude_spectrum"]), caption="Transformed Spectrum", use_container_width=True)

        metrics = result["metrics"]
        metric_rows = [
            {"Metric": "MSE", "Value": metrics["mse"]},
            {"Metric": "PSNR", "Value": metrics["psnr"]},
            {"Metric": "SSIM", "Value": metrics["ssim"]},
            {"Metric": "Original High/Low Ratio", "Value": metrics["original_high_low_ratio"]},
            {"Metric": "Transformed High/Low Ratio", "Value": metrics["transformed_high_low_ratio"]},
        ]
        st.subheader("Metric Table")
        st.table(metric_rows)

        st.subheader("Difference Visualization")
        st.image(str(metrics["difference_path"]), use_container_width=True)

        csv_payload = _build_csv_payload(result)
        st.download_button(
            label="Download Metrics CSV",
            data=csv_payload,
            file_name="evaluation_metrics.csv",
            mime="text/csv",
        )
    except FileNotFoundError as exc:
        st.error(f"Image could not be found: {exc}")
    except ValueError as exc:
        st.error(f"Processing failed: {exc}")
    except RuntimeError as exc:
        st.error(f"Runtime dependency error: {exc}")
    except Exception as exc:  # pragma: no cover - GUI fallback path
        st.error(f"Unexpected error: {exc}")
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
