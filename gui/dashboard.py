from __future__ import annotations

from io import StringIO
from pathlib import Path
import csv
import importlib
import sys
import tempfile
import threading

import cv2
import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as pipeline_app
pipeline_app = importlib.reload(pipeline_app)

run_analysis_pipeline = getattr(pipeline_app, "run_analysis_pipeline")
run_reference_expression_transfer_pipeline = getattr(pipeline_app, "run_reference_expression_transfer_pipeline")
run_realtime_frame_pipeline = getattr(pipeline_app, "run_realtime_frame_pipeline")
prepare_reference_expression_payload = getattr(pipeline_app, "prepare_reference_expression_payload")
DEFAULT_LANDMARK_REGIONS = getattr(pipeline_app, "DEFAULT_LANDMARK_REGIONS", ["eyes", "lips", "nose"])
DEFAULT_TRANSFER_REGIONS = getattr(pipeline_app, "DEFAULT_TRANSFER_REGIONS", ["eyes", "eyebrows", "nose", "lips"])

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
    "Reference Expression Transfer": "reference_expression_transfer",
}
REGION_OPTIONS = ["eyes", "eyebrows", "nose", "lips", "jawline", "cheeks"]


class RealtimeVideoProcessor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.transformation = "aging"
        self.intensity = 0.55
        self.show_landmarks = False
        self.selected_regions = list(DEFAULT_LANDMARK_REGIONS)
        self.reference_payload = None
        self.last_error = None

    def recv(self, frame):  # pragma: no cover - exercised through Streamlit runtime
        image = frame.to_ndarray(format="bgr24")
        try:
            with self.lock:
                transformation = self.transformation
                intensity = self.intensity
                show_landmarks = self.show_landmarks
                selected_regions = list(self.selected_regions)
                reference_payload = self.reference_payload
            result = run_realtime_frame_pipeline(
                image,
                transformation=transformation,
                intensity=intensity,
                show_landmarks=show_landmarks,
                selected_regions=selected_regions,
                reference_payload=reference_payload,
            )
            output = result["composite_frame"]
            self.last_error = None
        except Exception as exc:
            output = image.copy()
            cv2.putText(output, str(exc)[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            self.last_error = str(exc)
        return av.VideoFrame.from_ndarray(output, format="bgr24")


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        [data-testid="stSidebar"] {background: linear-gradient(180deg, #f5f7fb 0%, #eef3f8 100%);}
        .card {
            border: 1px solid rgba(24, 36, 56, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: white;
            box-shadow: 0 10px 30px rgba(16, 24, 40, 0.06);
            margin-bottom: 1rem;
        }
        .metric-label {font-size: 0.85rem; color: #5b6472; margin-bottom: 0.15rem;}
        .metric-value {font-size: 1.2rem; font-weight: 700; color: #162033;}
        </style>
        """,
        unsafe_allow_html=True,
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


def _spectrum_to_rgb(spectrum: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def _cleanup_paths(paths: list[str | None]) -> None:
    for temp_path in paths:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _encode_rgb_image(rgb: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode PNG output.")
    return bytes(buffer)


def _read_bytes(path: str | Path | None) -> bytes | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


def _build_metrics_csv(metrics: dict) -> str:
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


def _show_metric(label: str, value: str) -> None:
    st.markdown(
        f"<div class='card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def _explanation_lines(result: dict) -> list[str]:
    transformation = result.get("transformation", {})
    if "filter_explanation" in transformation:
        return transformation["filter_explanation"]
    if "explanation" in transformation:
        return transformation["explanation"]
    operation = transformation.get("operation") or transformation.get("mode")
    if operation:
        return [f"Applied transformation: {operation}.", "Geometry and evaluation stages were executed through the shared backend pipeline."]
    return ["Transformation completed through the shared backend pipeline."]


def _render_status_panel(result: dict, is_transfer: bool = False) -> None:
    metrics = result["metrics"]
    face_detection = result["source_face_detection"] if is_transfer else result["face_detection"]
    landmarks = result.get("source_landmarks") if is_transfer else result.get("landmarks")
    bbox = face_detection["bounding_box"]

    st.markdown("### Analysis Panel")
    _show_metric("Face Detection", "Detected")
    _show_metric("Bounding Box", f"x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    _show_metric("Landmark Count", str(len(landmarks["landmarks"]) if landmarks else 0))
    _show_metric("MSE", f"{metrics['mse']:.4f}")
    _show_metric("PSNR", f"{metrics['psnr']:.4f}")
    _show_metric("SSIM", f"{metrics['ssim']:.4f}")
    _show_metric("High/Low Ratio", f"{metrics['transformed_high_low_ratio']:.6f}")

    st.markdown("### Explanation")
    for line in _explanation_lines(result):
        st.write(f"- {line}")


def _render_bottom_analysis(result: dict, transformed_rgb: np.ndarray, title_prefix: str = "") -> None:
    metrics = result["metrics"]
    st.markdown("## DSP Analysis")

    fourier_col1, fourier_col2 = st.columns(2)
    with fourier_col1:
        st.markdown("#### Fourier Before")
        st.image(_spectrum_to_rgb(result["original_frequency"]["magnitude_spectrum"]), use_container_width=True)
    with fourier_col2:
        st.markdown("#### Fourier After")
        st.image(_spectrum_to_rgb(result["transformed_frequency"]["magnitude_spectrum"]), use_container_width=True)

    diff_col1, diff_col2 = st.columns([2, 1])
    with diff_col1:
        st.markdown("#### Absolute Difference")
        st.image(str(metrics["difference_path"]), use_container_width=True)
    with diff_col2:
        st.markdown("#### Export")
        st.download_button(
            label="Download Metrics CSV",
            data=_build_metrics_csv(metrics),
            file_name=f"{title_prefix}metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            label="Download Transformed PNG",
            data=_encode_rgb_image(transformed_rgb),
            file_name=f"{title_prefix}transformed.png",
            mime="image/png",
            use_container_width=True,
        )
        diff_bytes = _read_bytes(metrics.get("difference_path"))
        if diff_bytes is not None:
            st.download_button(
                label="Download Difference PNG",
                data=diff_bytes,
                file_name=f"{title_prefix}difference.png",
                mime="image/png",
                use_container_width=True,
            )


def _render_image_mode() -> None:
    st.sidebar.markdown("## Controls")
    source_mode = st.sidebar.radio("Source input", ["Upload image", "Camera capture"], horizontal=False)
    uploaded_file = st.sidebar.file_uploader("Source image", type=["jpg", "jpeg", "png"], key="dashboard_source") if source_mode == "Upload image" else None
    camera_file = st.sidebar.camera_input("Capture source frame", key="dashboard_camera") if source_mode == "Camera capture" else None
    selected_file = uploaded_file or camera_file

    transformation_label = st.sidebar.selectbox("Transformation", list(TRANSFORMATION_OPTIONS.keys()))
    transformation = TRANSFORMATION_OPTIONS[transformation_label]
    intensity = st.sidebar.slider("Intensity", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
    show_landmarks = st.sidebar.toggle("Show landmarks", value=True)
    default_regions = DEFAULT_TRANSFER_REGIONS if transformation == "reference_expression_transfer" else DEFAULT_LANDMARK_REGIONS
    selected_regions = st.sidebar.multiselect("Regions", REGION_OPTIONS, default=list(default_regions))

    reference_file = None
    if transformation == "reference_expression_transfer":
        st.sidebar.markdown("## Reference")
        reference_file = st.sidebar.file_uploader("Reference image", type=["jpg", "jpeg", "png"], key="dashboard_reference")

    st.markdown("## Image Studio")
    st.caption("Left panel controls the pipeline. Center shows source and transformed previews. Right panel exposes engineering metrics and explanations.")

    if selected_file is None:
        st.info("Provide a source face image from upload or camera capture.")
        return
    if transformation == "reference_expression_transfer" and reference_file is None:
        st.info("Reference expression transfer requires a reference face image.")
        return

    source_temp = None
    reference_temp = None
    try:
        source_temp = _save_uploaded_file(selected_file)
        if reference_file is not None:
            reference_temp = _save_uploaded_file(reference_file)
            result = run_reference_expression_transfer_pipeline(
                source_temp,
                reference_temp,
                blend_factor=intensity,
                show_landmarks=show_landmarks,
                selected_regions=selected_regions or list(DEFAULT_TRANSFER_REGIONS),
            )
            source_preview = _image_dict_to_rgb(result["source_face_detection"]["face_image"])
            transformed_preview = _image_dict_to_rgb(result["transformation"]["image"])
            reference_preview = _image_dict_to_rgb(result["reference_face_detection"]["face_image"])
            is_transfer = True
        else:
            result = run_analysis_pipeline(
                source_temp,
                transformation=transformation,
                intensity=intensity,
                show_landmarks=show_landmarks,
                selected_regions=selected_regions or list(DEFAULT_LANDMARK_REGIONS),
            )
            source_preview = _image_dict_to_rgb(result["face_detection"]["face_image"])
            transformed_preview = _image_dict_to_rgb(result["transformation"]["image"])
            reference_preview = None
            is_transfer = False

        main_col, transformed_col, side_col = st.columns([1.4, 1.4, 1.0])
        with main_col:
            st.markdown("### Source Preview")
            st.image(source_preview, use_container_width=True)
            if show_landmarks:
                landmark_payload = result["source_landmarks"]["visualization"] if is_transfer else result["landmarks"]["visualization"]
                st.markdown("#### Landmark Overlay")
                st.image(_image_dict_to_rgb(landmark_payload), use_container_width=True)
        with transformed_col:
            st.markdown("### Transformed Preview")
            st.image(transformed_preview, use_container_width=True)
            if reference_preview is not None:
                st.markdown("#### Reference Preview")
                st.image(reference_preview, use_container_width=True)
        with side_col:
            _render_status_panel(result, is_transfer=is_transfer)

        _render_bottom_analysis(result, transformed_preview, title_prefix="transfer_" if is_transfer else "analysis_")
    except FileNotFoundError as exc:
        st.error(f"Image could not be found: {exc}")
    except ValueError as exc:
        st.error(f"Processing failed: {exc}")
    except RuntimeError as exc:
        st.error(f"Runtime dependency error: {exc}")
    finally:
        _cleanup_paths([source_temp, reference_temp])


def _render_realtime_mode() -> None:
    st.sidebar.markdown("## Real-Time Controls")
    transformation_label = st.sidebar.selectbox("Live transformation", list(TRANSFORMATION_OPTIONS.keys()), key="live_transform")
    transformation = TRANSFORMATION_OPTIONS[transformation_label]
    intensity = st.sidebar.slider("Live intensity", min_value=0.0, max_value=1.0, value=0.55, step=0.05, key="live_intensity")
    show_landmarks = st.sidebar.toggle("Live landmarks", value=False, key="live_landmarks")
    default_regions = DEFAULT_TRANSFER_REGIONS if transformation == "reference_expression_transfer" else DEFAULT_LANDMARK_REGIONS
    selected_regions = st.sidebar.multiselect("Live regions", REGION_OPTIONS, default=list(default_regions), key="live_regions")

    reference_payload = None
    reference_temp = None
    if transformation == "reference_expression_transfer":
        reference_file = st.sidebar.file_uploader("Live reference image", type=["jpg", "jpeg", "png"], key="live_reference")
        if reference_file is not None:
            try:
                reference_temp = _save_uploaded_file(reference_file)
                reference_payload = prepare_reference_expression_payload(
                    reference_temp,
                    show_landmarks=show_landmarks,
                    selected_regions=selected_regions or list(DEFAULT_TRANSFER_REGIONS),
                )
            except Exception as exc:
                st.sidebar.error(f"Reference preparation failed: {exc}")

    st.markdown("## Real-Time Lab")
    st.caption("Browser webcam mode keeps the backend pipeline intact. The live stream shows source on the left and transformed output on the right.")

    top_col, side_col = st.columns([2.2, 1.0])
    with top_col:
        if STREAMLIT_WEBRTC_AVAILABLE:
            webrtc_ctx = webrtc_streamer(
                key="modern-facial-live-stream",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                async_processing=True,
                video_processor_factory=RealtimeVideoProcessor,
            )
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.lock:
                    webrtc_ctx.video_processor.transformation = transformation
                    webrtc_ctx.video_processor.intensity = intensity
                    webrtc_ctx.video_processor.show_landmarks = show_landmarks
                    webrtc_ctx.video_processor.selected_regions = selected_regions or list(default_regions)
                    webrtc_ctx.video_processor.reference_payload = reference_payload
                if webrtc_ctx.video_processor.last_error:
                    st.warning(webrtc_ctx.video_processor.last_error)
        else:
            st.warning("Install `streamlit-webrtc` and `av` to enable browser real-time streaming.")
            st.code("pip install -e \".[dev]\"", language="bash")

        st.markdown("### Snapshot Analysis")
        snapshot = st.camera_input("Capture a frame for metrics, Fourier analysis, and exports.", key="realtime_snapshot")
    with side_col:
        st.markdown("### Live Status")
        _show_metric("Transformation", transformation)
        _show_metric("Intensity", f"{intensity:.2f}")
        _show_metric("Landmark Overlay", "On" if show_landmarks else "Off")
        _show_metric("Region Count", str(len(selected_regions or default_regions)))
        if reference_payload is not None:
            _show_metric("Reference Face", "Ready")
            st.image(_image_dict_to_rgb(reference_payload["face_detection"]["face_image"]), caption="Reference ROI", use_container_width=True)

    if snapshot is not None:
        temp_path = None
        try:
            temp_path = _save_uploaded_file(snapshot)
            if transformation == "reference_expression_transfer":
                if reference_temp is None:
                    raise ValueError("Upload a reference image before running real-time reference transfer snapshot analysis.")
                result = run_reference_expression_transfer_pipeline(
                    temp_path,
                    reference_temp,
                    blend_factor=intensity,
                    show_landmarks=show_landmarks,
                    selected_regions=selected_regions or list(DEFAULT_TRANSFER_REGIONS),
                )
                source_preview = _image_dict_to_rgb(result["source_face_detection"]["face_image"])
            else:
                result = run_analysis_pipeline(
                    temp_path,
                    transformation=transformation,
                    intensity=intensity,
                    show_landmarks=show_landmarks,
                    selected_regions=selected_regions or list(default_regions),
                )
                source_preview = _image_dict_to_rgb(result["face_detection"]["face_image"])
            transformed_preview = _image_dict_to_rgb(result["transformation"]["image"])
            preview_col1, preview_col2, preview_col3 = st.columns([1.25, 1.25, 1.0])
            with preview_col1:
                st.markdown("### Snapshot Source")
                st.image(source_preview, use_container_width=True)
            with preview_col2:
                st.markdown("### Snapshot Transformed")
                st.image(transformed_preview, use_container_width=True)
            with preview_col3:
                _render_status_panel(result, is_transfer=transformation == "reference_expression_transfer")
            _render_bottom_analysis(result, transformed_preview, title_prefix="realtime_")
        except Exception as exc:
            st.error(f"Snapshot analysis failed: {exc}")
        finally:
            _cleanup_paths([temp_path, reference_temp])
    else:
        _cleanup_paths([reference_temp])


def main() -> None:
    st.set_page_config(page_title="Facial Image Warping Dashboard", layout="wide")
    _apply_page_style()
    st.title("Facial Image Warping Engineering Dashboard")
    st.caption("Modern control surface on top of the existing face detection, landmark, warping, aging, transfer, Fourier, and evaluation backend modules.")

    workspace_mode = st.sidebar.radio("Workspace", ["Image Studio", "Real-Time Lab"], index=0)
    if workspace_mode == "Image Studio":
        _render_image_mode()
    else:
        _render_realtime_mode()


if __name__ == "__main__":
    main()



