"""Streamlit GUI for facial image warping, aging, transfer, and evaluation."""
"""Streamlit GUI for facial image warping, aging, transfer, and evaluation."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import csv
import importlib
import tempfile
import threading

import cv2
import numpy as np
import streamlit as st

import app as pipeline_app
pipeline_app = importlib.reload(pipeline_app)

run_analysis_pipeline = getattr(pipeline_app, "run_analysis_pipeline")
run_realtime_frame_pipeline = getattr(pipeline_app, "run_realtime_frame_pipeline")
run_reference_expression_transfer_pipeline = getattr(pipeline_app, "run_reference_expression_transfer_pipeline")
compare_reference_expression_transfer_methods = getattr(pipeline_app, "compare_reference_expression_transfer_methods")
prepare_reference_expression_payload = getattr(pipeline_app, "prepare_reference_expression_payload")
DEFAULT_LANDMARK_REGIONS = getattr(pipeline_app, "DEFAULT_LANDMARK_REGIONS", ["eyes", "lips", "nose"])
DEFAULT_TRANSFER_REGIONS = getattr(pipeline_app, "DEFAULT_TRANSFER_REGIONS", ["eyes", "eyebrows", "lips"])
DEFAULT_TRANSFER_METHOD = getattr(pipeline_app, "DEFAULT_TRANSFER_METHOD", "expression_coefficients")

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    STREAMLIT_WEBRTC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency path
    av = None
    WebRtcMode = None
    webrtc_streamer = None
    STREAMLIT_WEBRTC_AVAILABLE = False


TRANSFORMATION_OPTIONS = {
    "Smile Enhancement": "smile_enhancement",
    "Eyebrow Raising": "eyebrow_raising",
    "Lip Widening": "lip_widening",
    "Face Slimming": "face_slimming",
    "Aging": "aging",
    "De-Aging": "de-aging",
}
REALTIME_TRANSFORMATION_OPTIONS = {
    **TRANSFORMATION_OPTIONS,
    "Reference Expression Transfer": "reference_expression_transfer",
}
REGION_OPTIONS = ["eyes", "eyebrows", "nose", "lips", "jawline", "cheeks"]
TRANSFER_METHOD_OPTIONS = {
    "Recommended Coefficients": "expression_coefficients",
    "Thin Plate Spline": "tps",
    "Safe Classical": "safe_classical",
    "Auto Compare All": "auto_compare",
}
STREAM_QUALITY_PROFILES = {
    "Low Latency": {"max_width": 480, "frame_skip_interval": 3, "frame_rate": 12},
    "Balanced": {"max_width": 640, "frame_skip_interval": 2, "frame_rate": 15},
    "High Quality": {"max_width": 960, "frame_skip_interval": 1, "frame_rate": 20},
}


def _resize_frame_for_realtime(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale the incoming webcam frame to reduce live processing latency."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / max(width, 1)
    return cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


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


def _enhance_preview(rgb: np.ndarray, min_width: int = 720) -> np.ndarray:
    """Upscale small camera crops for cleaner GUI previews without touching analysis pixels."""
    height, width = rgb.shape[:2]
    if width >= min_width:
        return rgb
    scale = min_width / max(width, 1)
    resized = cv2.resize(rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_CUBIC)
    softened = cv2.GaussianBlur(resized, (0, 0), 0.8)
    return cv2.addWeighted(resized, 1.12, softened, -0.12, 0)


def _spectrum_to_rgb(spectrum: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _build_csv_payload(metrics: dict) -> str:
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


def _save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def _show_metrics(metrics: dict) -> None:
    metric_rows = [
        {"Metric": "MSE", "Value": metrics["mse"]},
        {"Metric": "PSNR", "Value": metrics["psnr"]},
        {"Metric": "SSIM", "Value": metrics["ssim"]},
        {"Metric": "Original High/Low Ratio", "Value": metrics["original_high_low_ratio"]},
        {"Metric": "Transformed High/Low Ratio", "Value": metrics["transformed_high_low_ratio"]},
    ]
    st.subheader("Metric Table")
    if "transfer_quality_score" in metrics:
        metric_rows.extend(
            [
                {"Metric": "Transfer Score", "Value": metrics["transfer_quality_score"]},
                {"Metric": "Expression Match", "Value": metrics["expression_match_score"]},
                {"Metric": "Identity Preservation", "Value": metrics["identity_preservation_score"]},
            ]
        )
    st.table(metric_rows)
    st.subheader("Difference Visualization")
    st.image(str(metrics["difference_path"]), use_container_width=True)
    st.download_button(
        label="Download Metrics CSV",
        data=_build_csv_payload(metrics),
        file_name="evaluation_metrics.csv",
        mime="text/csv",
    )


def _show_frequency(original_frequency: dict, transformed_frequency: dict) -> None:
    st.subheader("Fourier Magnitude Spectrum")
    spectrum_col1, spectrum_col2 = st.columns(2)
    with spectrum_col1:
        st.image(_spectrum_to_rgb(original_frequency["magnitude_spectrum"]), caption="Original Spectrum", use_container_width=True)
    with spectrum_col2:
        st.image(_spectrum_to_rgb(transformed_frequency["magnitude_spectrum"]), caption="Transformed Spectrum", use_container_width=True)


def _cleanup_paths(paths: list[str | None]) -> None:
    for temp_path in paths:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _show_realism_guidance() -> None:
    with st.expander("How to get more realistic source/reference transfer"):
        st.markdown(
            """
1. `Source` ve `reference` yuzleri benzer kafa acisinda sec.
2. Referans yuzde gozluk, el, sac kapatmasi gibi `occlusion` varsa transfer kalitesi duser.
3. Ilk denemede sadece `eyes + eyebrows + lips` sec; `jawline` ve `cheeks` daha kolay artifact uretir.
4. `Blend factor` degerini once `0.45 - 0.70` araliginda tut; `1.0` bazen asiri deformasyon yaratir.
5. Landmark overlay'i acip agiz koseleri, kaslar ve gozlerin referans ile uyumlu oturdugunu kontrol et.
6. Referans ifadesi source yuze gore cok ekstremse once orta siddette bir referans sec.
            """
        )


def _render_standard_transform_tab() -> None:
    st.subheader("Standard Image Transform")
    source_mode = st.radio("Image source", ["Upload image", "Webcam capture"], horizontal=True)
    uploaded_file = st.file_uploader("Source face image", type=["jpg", "jpeg", "png"], key="standard_upload") if source_mode == "Upload image" else None
    camera_file = st.camera_input("Capture webcam frame", key="standard_camera") if source_mode == "Webcam capture" else None
    selected_file = uploaded_file or camera_file

    transformation_label = st.selectbox("Transformation", list(TRANSFORMATION_OPTIONS.keys()))
    intensity = st.slider("Intensity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    show_landmarks = st.checkbox("Show facial landmarks", value=False)
    selected_regions = st.multiselect(
        "Highlighted landmark regions",
        REGION_OPTIONS,
        default=list(DEFAULT_LANDMARK_REGIONS),
        disabled=not show_landmarks,
    )

    if selected_file is None:
        st.info("Upload a face image or capture one from the webcam.")
        return

    temp_path = None
    try:
        temp_path = _save_uploaded_file(selected_file)
        result = run_analysis_pipeline(
            temp_path,
            transformation=TRANSFORMATION_OPTIONS[transformation_label],
            intensity=intensity,
            show_landmarks=show_landmarks,
            selected_regions=selected_regions or list(DEFAULT_LANDMARK_REGIONS),
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Face ROI")
            st.image(_enhance_preview(_image_dict_to_rgb(result["face_detection"]["face_image"])), use_container_width=True)
        with col2:
            st.subheader("Transformed Result")
            st.image(_enhance_preview(_image_dict_to_rgb(result["transformation"]["image"])), use_container_width=True)

        if show_landmarks and result["landmarks"] is not None:
            st.subheader("Landmark Visualization")
            st.image(_enhance_preview(_image_dict_to_rgb(result["landmarks"]["visualization"])), use_container_width=True)

        _show_frequency(result["original_frequency"], result["transformed_frequency"])
        _show_metrics(result["metrics"])
    except FileNotFoundError as exc:
        st.error(f"Image could not be found: {exc}")
    except ValueError as exc:
        st.error(f"Processing failed: {exc}")
    except RuntimeError as exc:
        st.error(f"Runtime dependency error: {exc}")
    except Exception as exc:  # pragma: no cover
        st.error(f"Unexpected error: {exc}")
    finally:
        _cleanup_paths([temp_path])


def _render_transfer_tab() -> None:
    st.subheader("Advanced AI-Based Expression Transfer")
    st.caption("This mode uses AI-assisted facial landmark detection to transfer the expression geometry of a reference face onto the source face.")
    _show_realism_guidance()

    source_file = st.file_uploader("Source face image", type=["jpg", "jpeg", "png"], key="transfer_source")
    reference_mode = st.radio("Reference source", ["Upload reference image", "Webcam capture reference"], horizontal=True)
    reference_upload = st.file_uploader("Reference expression image", type=["jpg", "jpeg", "png"], key="transfer_reference") if reference_mode == "Upload reference image" else None
    reference_camera = st.camera_input("Capture reference expression from webcam", key="transfer_camera") if reference_mode == "Webcam capture reference" else None
    reference_file = reference_upload or reference_camera

    blend_factor = st.slider("Reference expression blend", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    show_landmarks = st.checkbox("Show source and reference landmarks", value=True, key="transfer_landmarks")
    selected_regions = st.multiselect(
        "Transfer regions",
        REGION_OPTIONS,
        default=list(DEFAULT_TRANSFER_REGIONS),
        key="transfer_regions",
    )
    transfer_method_label = st.selectbox(
        "Transfer method",
        list(TRANSFER_METHOD_OPTIONS.keys()),
        index=list(TRANSFER_METHOD_OPTIONS.values()).index(DEFAULT_TRANSFER_METHOD),
        key="transfer_method",
    )
    transfer_method = TRANSFER_METHOD_OPTIONS[transfer_method_label]

    if source_file is None or reference_file is None:
        st.info("Provide both source and reference face images to run reference expression transfer.")
        return

    source_temp = None
    reference_temp = None
    try:
        source_temp = _save_uploaded_file(source_file)
        reference_temp = _save_uploaded_file(reference_file)
        if transfer_method == "auto_compare":
            comparison = compare_reference_expression_transfer_methods(
                source_temp,
                reference_temp,
                blend_factor=blend_factor,
                show_landmarks=show_landmarks,
                selected_regions=selected_regions or list(DEFAULT_TRANSFER_REGIONS),
            )
            result = comparison["results"][comparison["best_method"]]
        else:
            comparison = None
            result = run_reference_expression_transfer_pipeline(
                source_temp,
                reference_temp,
                blend_factor=blend_factor,
                show_landmarks=show_landmarks,
                selected_regions=selected_regions or list(DEFAULT_TRANSFER_REGIONS),
                transfer_method=transfer_method,
            )

        top_left, top_right = st.columns(2)
        with top_left:
            st.subheader("Source Face ROI")
            st.image(_enhance_preview(_image_dict_to_rgb(result["source_face_detection"]["face_image"])), use_container_width=True)
        with top_right:
            st.subheader("Transferred Result")
            st.image(_enhance_preview(_image_dict_to_rgb(result["transformation"]["image"])), use_container_width=True)

        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            st.subheader("Reference Face ROI")
            st.image(_enhance_preview(_image_dict_to_rgb(result["reference_face_detection"]["face_image"])), use_container_width=True)
        with ref_col2:
            st.subheader("Transfer Explanation")
            st.write(f"Method: `{result.get('transfer_method', DEFAULT_TRANSFER_METHOD)}`")
            for line in result["transformation"]["explanation"]:
                st.write(f"- {line}")
            if comparison is not None:
                st.write("Ranking:")
                st.table(
                    [
                        {
                            "Method": method,
                            "Score": f"{comparison['results'][method]['metrics']['transfer_quality_score']:.2f}",
                        }
                        for method in comparison["ranked_methods"]
                    ]
                )

        if show_landmarks:
            source_landmark_col, reference_landmark_col = st.columns(2)
            with source_landmark_col:
                st.subheader("Source Landmarks")
                st.image(_enhance_preview(_image_dict_to_rgb(result["source_landmarks"]["visualization"])), use_container_width=True)
            with reference_landmark_col:
                st.subheader("Reference Landmarks")
                st.image(_enhance_preview(_image_dict_to_rgb(result["reference_landmarks"]["visualization"])), use_container_width=True)

        _show_frequency(result["original_frequency"], result["transformed_frequency"])
        _show_metrics(result["metrics"])
    except FileNotFoundError as exc:
        st.error(f"Image could not be found: {exc}")
    except ValueError as exc:
        st.error(f"Transfer failed: {exc}")
    except RuntimeError as exc:
        st.error(f"Runtime dependency error: {exc}")
    except Exception as exc:  # pragma: no cover
        st.error(f"Unexpected error: {exc}")
    finally:
        _cleanup_paths([source_temp, reference_temp])


class RealtimeVideoProcessor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.transformation = "aging"
        self.intensity = 0.55
        self.show_landmarks = False
        self.selected_regions = list(DEFAULT_LANDMARK_REGIONS)
        self.reference_payload = None
        self.last_error = None
        self.frame_index = 0
        self.last_output = None
        self.previous_landmarks = None
        self.previous_transformation = "aging"
        self.max_width = 640
        self.frame_skip_interval = 2
        self.transfer_method = DEFAULT_TRANSFER_METHOD

    def recv(self, frame):  # pragma: no cover - exercised through Streamlit runtime
        image = frame.to_ndarray(format="bgr24")
        try:
            with self.lock:
                transformation = self.transformation
                intensity = self.intensity
                show_landmarks = self.show_landmarks
                selected_regions = list(self.selected_regions)
                reference_payload = self.reference_payload
                max_width = self.max_width
                frame_skip_interval = self.frame_skip_interval
                transfer_method = self.transfer_method
            if transformation != self.previous_transformation:
                self.previous_landmarks = None
                self.last_output = None
                self.previous_transformation = transformation
            self.frame_index += 1
            resized_image = _resize_frame_for_realtime(image, max_width=max_width)
            should_reuse_last_frame = (
                frame_skip_interval > 1
                and self.last_output is not None
                and self.frame_index % frame_skip_interval != 1
            )
            if should_reuse_last_frame:
                output = self.last_output.copy()
            else:
                result = run_realtime_frame_pipeline(
                    resized_image,
                    transformation=transformation,
                    intensity=intensity,
                    show_landmarks=show_landmarks,
                    selected_regions=selected_regions,
                    reference_payload=reference_payload,
                    previous_landmarks=self.previous_landmarks,
                    transfer_method=transfer_method,
                )
                output = result["composite_frame"]
                self.last_output = output.copy()
                self.previous_landmarks = result["landmarks"]["landmarks"] if result.get("landmarks") else None
            self.last_error = None
        except Exception as exc:
            output = image.copy()
            cv2.putText(output, str(exc)[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            self.last_error = str(exc)
        return av.VideoFrame.from_ndarray(output, format="bgr24")


def _render_realtime_stream() -> None:
    if not STREAMLIT_WEBRTC_AVAILABLE:
        st.warning("Real-time Streamlit webcam mode requires `streamlit-webrtc` and `av`. Install dependencies and rerun the app.")
        st.code("pip install -e \".[dev]\"", language="bash")
        return

    st.subheader("Streamlit Real-Time Webcam")
    st.caption("Left side shows the live source frame, right side shows the transformed live frame. Adjust sliders and toggles while the stream is running.")
    _show_realism_guidance()

    transformation_label = st.selectbox("Real-time transformation", list(REALTIME_TRANSFORMATION_OPTIONS.keys()), key="realtime_transformation")
    realtime_transformation = REALTIME_TRANSFORMATION_OPTIONS[transformation_label]
    intensity = st.slider("Real-time intensity", min_value=0.0, max_value=1.0, value=0.55, step=0.05, key="realtime_intensity")
    show_landmarks = st.checkbox("Show live landmarks", value=False, key="realtime_landmarks")
    selected_regions = st.multiselect(
        "Live landmark/transfer regions",
        REGION_OPTIONS,
        default=list(DEFAULT_TRANSFER_REGIONS if realtime_transformation == "reference_expression_transfer" else DEFAULT_LANDMARK_REGIONS),
        key="realtime_regions",
    )
    quality_label = st.selectbox("Stream quality", list(STREAM_QUALITY_PROFILES.keys()), index=1, key="realtime_quality")
    quality_profile = STREAM_QUALITY_PROFILES[quality_label]

    reference_payload = None
    reference_temp = None
    if realtime_transformation == "reference_expression_transfer":
        live_transfer_method_label = st.selectbox(
            "Live transfer method",
            [label for label, value in TRANSFER_METHOD_OPTIONS.items() if value != "auto_compare"],
            index=[value for value in TRANSFER_METHOD_OPTIONS.values() if value != "auto_compare"].index(DEFAULT_TRANSFER_METHOD),
            key="realtime_transfer_method",
        )
        transfer_method = TRANSFER_METHOD_OPTIONS[live_transfer_method_label]
        reference_file = st.file_uploader("Reference image for live transfer", type=["jpg", "jpeg", "png"], key="realtime_reference")
        if reference_file is None:
            st.info("Upload a reference image to activate live reference expression transfer.")
        else:
            try:
                reference_temp = _save_uploaded_file(reference_file)
                reference_payload = prepare_reference_expression_payload(
                    reference_temp,
                    show_landmarks=show_landmarks,
                    selected_regions=selected_regions or list(DEFAULT_TRANSFER_REGIONS),
                )
                st.image(_enhance_preview(_image_dict_to_rgb(reference_payload["face_detection"]["face_image"])), caption="Live reference face ROI", use_container_width=False)
            except Exception as exc:
                st.error(f"Reference preparation failed: {exc}")
    else:
        transfer_method = DEFAULT_TRANSFER_METHOD

    webrtc_ctx = webrtc_streamer(
        key="facial-live-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {
                "width": {"ideal": quality_profile["max_width"]},
                "frameRate": {"ideal": quality_profile["frame_rate"]},
            },
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
        video_processor_factory=RealtimeVideoProcessor,
    )

    if webrtc_ctx.video_processor:
        with webrtc_ctx.video_processor.lock:
            webrtc_ctx.video_processor.transformation = realtime_transformation
            webrtc_ctx.video_processor.intensity = intensity
            webrtc_ctx.video_processor.show_landmarks = show_landmarks
            webrtc_ctx.video_processor.selected_regions = selected_regions or list(DEFAULT_LANDMARK_REGIONS)
            webrtc_ctx.video_processor.reference_payload = reference_payload
            webrtc_ctx.video_processor.max_width = quality_profile["max_width"]
            webrtc_ctx.video_processor.frame_skip_interval = quality_profile["frame_skip_interval"]
            webrtc_ctx.video_processor.transfer_method = transfer_method
        if webrtc_ctx.video_processor.last_error:
            st.warning(webrtc_ctx.video_processor.last_error)

    _cleanup_paths([reference_temp])


def _render_webcam_tab() -> None:
    st.subheader("Webcam Integration")
    _render_realtime_stream()
    st.divider()
    st.subheader("Standalone OpenCV Webcam Demo")
    st.code("python webcam_demo.py --transformation aging --intensity 0.6", language="bash")
    st.code("python webcam_demo.py --transformation reference_expression_transfer --reference-image samples/test_face_2.png --intensity 0.75", language="bash")
    st.markdown(
        """
- `Q` tusu ile OpenCV penceresini kapatirsin.
- `reference_expression_transfer` icin referans gorseli gerekir.
- Streamlit real-time modu GUI icinden intensity, region ve landmark kontrolu sunar.
        """
    )


def main() -> None:
    st.set_page_config(page_title="Facial Image Warping", layout="wide")
    st.title("Facial Image Warping, Aging, and Expression Transformation")
    st.caption("Streamlit GUI: standard transforms, reference-based expression transfer, and real-time webcam transformation controls.")

    tab1, tab2, tab3 = st.tabs(["Standard Transform", "Expression Transfer", "Webcam Integration"])
    with tab1:
        _render_standard_transform_tab()
    with tab2:
        _render_transfer_tab()
    with tab3:
        _render_webcam_tab()


if __name__ == "__main__":
    main()
