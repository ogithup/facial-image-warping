from __future__ import annotations

from io import StringIO
from pathlib import Path
import csv
import importlib
import sys
import tempfile
import threading
import time
from typing import Any

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
compare_reference_expression_transfer_methods = getattr(pipeline_app, "compare_reference_expression_transfer_methods")
run_realtime_frame_pipeline = getattr(pipeline_app, "run_realtime_frame_pipeline")
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
    "Raw Camera Debug": "raw_camera_debug",
    "Smile Enhancement": "smile_enhancement",
    "Eyebrow Raising": "eyebrow_raising",
    "Lip Widening": "lip_widening",
    "Face Slimming": "face_slimming",
    "Aging": "aging",
    "De-Aging": "de-aging",
    "Reference Expression Transfer": "reference_expression_transfer",
}
REGION_OPTIONS = ["eyes", "eyebrows", "nose", "lips", "jawline", "cheeks"]
TRANSFER_METHOD_OPTIONS = {
    "Recommended Coefficients": "expression_coefficients",
    "Thin Plate Spline": "tps",
    "Safe Classical": "safe_classical",
    "Auto Compare All": "auto_compare",
}
CAMERA_SOURCE_PROFILES = {
    "Built-in Camera": {
        "key": "builtin",
        "description": "Laptop webcam for standard browser capture.",
        "realtime_width": 640,
        "realtime_height": 480,
        "frame_rate_cap": None,
        "aspect_ratio": None,
        "constraint_mode": "standard",
    },
    "DroidCam USB / External Virtual Camera": {
        "key": "droidcam",
        "description": "Use browser SELECT DEVICE and choose DroidCam. Optimized for stable 4:3 USB virtual webcam capture.",
        "realtime_width": 640,
        "realtime_height": 480,
        "frame_rate_cap": 12,
        "aspect_ratio": 4 / 3,
        "constraint_mode": "compatibility",
    },
}
STREAM_QUALITY_PROFILES = {
    "Ultra Safe": {
        "profile_name": "ultra_safe",
        "max_width": 320,
        "analysis_size": 224,
        "landmark_max_dimension": 256,
        "frame_skip_interval": 4,
        "detection_interval": 6,
        "frame_rate": 8,
        "smoothing_alpha": 0.5,
        "tracking_padding": 0.14,
        "reference_update_interval": 4,
    },
    "Fast": {
        "profile_name": "fast",
        "max_width": 432,
        "analysis_size": 256,
        "landmark_max_dimension": 320,
        "frame_skip_interval": 3,
        "detection_interval": 4,
        "frame_rate": 12,
        "smoothing_alpha": 0.55,
        "tracking_padding": 0.16,
        "reference_update_interval": 3,
    },
    "Balanced": {
        "profile_name": "balanced",
        "max_width": 640,
        "analysis_size": 320,
        "landmark_max_dimension": 384,
        "frame_skip_interval": 2,
        "detection_interval": 2,
        "frame_rate": 15,
        "smoothing_alpha": 0.65,
        "tracking_padding": 0.18,
        "reference_update_interval": 2,
    },
    "Quality": {
        "profile_name": "quality",
        "max_width": 896,
        "analysis_size": 384,
        "landmark_max_dimension": 512,
        "frame_skip_interval": 1,
        "detection_interval": 1,
        "frame_rate": 20,
        "smoothing_alpha": 0.72,
        "tracking_padding": 0.2,
        "reference_update_interval": 1,
    },
}

LOCAL_CAPTURE_STATE_KEY = "dashboard_local_capture"
LOCAL_LIVE_RUNNING_KEY = "dashboard_local_live_running"
LOCAL_USB_PERSISTENT_KEY = "dashboard_local_usb_persistent"
IMAGE_STUDIO_CAPTURE_FRAME_KEY = "dashboard_image_studio_capture_frame"
LOCAL_STREAM_FRAGMENT_INTERVAL_SECONDS = 0.10


def _resize_frame_for_realtime(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale the incoming webcam frame to keep real-time latency under control."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / max(width, 1)
    resized = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _build_live_media_constraints(quality_profile: dict, camera_profile: dict) -> dict:
    if camera_profile.get("constraint_mode") == "compatibility":
        return {
            "video": True,
            "audio": False,
        }

    width = int(min(quality_profile["max_width"], camera_profile["realtime_width"]))
    height = int(round(width / camera_profile["aspect_ratio"])) if camera_profile.get("aspect_ratio") else camera_profile["realtime_height"]
    frame_rate = quality_profile["frame_rate"]
    if camera_profile.get("frame_rate_cap") is not None:
        frame_rate = min(frame_rate, int(camera_profile["frame_rate_cap"]))
    video_constraints: dict[str, object] = {
        "width": {"ideal": width},
        "height": {"ideal": int(height)},
        "frameRate": {"ideal": int(frame_rate)},
    }
    if camera_profile.get("aspect_ratio") is not None:
        video_constraints["aspectRatio"] = {"ideal": float(camera_profile["aspect_ratio"])}
    return {"video": video_constraints, "audio": False}


def _ensure_local_capture(device_index: int) -> cv2.VideoCapture:
    capture = st.session_state.get(LOCAL_CAPTURE_STATE_KEY)
    current_index = st.session_state.get(f"{LOCAL_CAPTURE_STATE_KEY}_index")
    if capture is not None and current_index == device_index and capture.isOpened():
        return capture

    _release_local_capture()
    capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open local camera device index {device_index}.")
    st.session_state[LOCAL_CAPTURE_STATE_KEY] = capture
    st.session_state[f"{LOCAL_CAPTURE_STATE_KEY}_index"] = device_index
    return capture


def _release_local_capture() -> None:
    capture = st.session_state.get(LOCAL_CAPTURE_STATE_KEY)
    if capture is not None:
        try:
            capture.release()
        except Exception:
            pass
    st.session_state.pop(LOCAL_CAPTURE_STATE_KEY, None)
    st.session_state.pop(f"{LOCAL_CAPTURE_STATE_KEY}_index", None)


def _read_local_capture_frame(device_index: int, max_width: int) -> np.ndarray:
    capture = _ensure_local_capture(device_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Local camera device index {device_index} did not return a frame.")
    return _resize_frame_for_realtime(frame, max_width=max_width)


def _render_local_capture_frame(
    frame: np.ndarray,
    transformation: str,
    intensity: float,
    show_landmarks: bool,
    selected_regions: list[str],
    reference_payload: dict | None,
    transfer_method: str,
    quality_profile: dict,
    raw_preview_only: bool,
) -> tuple[np.ndarray, dict[str, float], str | None]:
    frame_start = time.perf_counter()
    if raw_preview_only or transformation == "raw_camera_debug":
        preview = frame.copy()
        cv2.putText(preview, "Raw camera debug - Local OpenCV mode", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return preview, {"pipeline_total_ms": float((time.perf_counter() - frame_start) * 1000.0)}, None

    previous_landmarks = st.session_state.get("dashboard_local_previous_landmarks")
    previous_bbox = st.session_state.get("dashboard_local_previous_bbox")
    frame_index = int(st.session_state.get("dashboard_local_frame_index", 0)) + 1
    st.session_state["dashboard_local_frame_index"] = frame_index
    try:
        result = run_realtime_frame_pipeline(
            frame,
            transformation=transformation,
            intensity=intensity,
            show_landmarks=show_landmarks,
            selected_regions=selected_regions,
            reference_payload=reference_payload,
            previous_landmarks=previous_landmarks,
            previous_bbox=previous_bbox,
            frame_index=frame_index,
            smoothing_alpha=float(quality_profile["smoothing_alpha"]),
            transfer_method=transfer_method,
            realtime_profile=quality_profile,
        )
        st.session_state["dashboard_local_previous_landmarks"] = result["landmarks"]["landmarks"] if result.get("landmarks") else None
        st.session_state["dashboard_local_previous_bbox"] = result["face_detection"]["bounding_box"]
        st.session_state["dashboard_local_last_good_frame"] = result["composite_frame"].copy()
        return result["composite_frame"], result["timings"], None
    except Exception as exc:
        fallback = st.session_state.get("dashboard_local_last_good_frame")
        warning_frame = fallback.copy() if fallback is not None else frame.copy()
        cv2.putText(
            warning_frame,
            str(exc)[:100],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            warning_frame,
            "Stream kept alive - waiting for a clearer face",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        return (
            warning_frame,
            {"pipeline_total_ms": float((time.perf_counter() - frame_start) * 1000.0)},
            str(exc),
        )


def _reset_local_pipeline_state() -> None:
    st.session_state["dashboard_local_previous_landmarks"] = None
    st.session_state["dashboard_local_previous_bbox"] = None
    st.session_state["dashboard_local_frame_index"] = 0
    st.session_state["dashboard_local_last_good_frame"] = None


def _render_local_stream_frame(
    local_device_index: int,
    transformation: str,
    intensity: float,
    show_landmarks: bool,
    selected_regions: list[str],
    default_regions: list[str],
    reference_payload: dict | None,
    transfer_method: str,
    quality_profile: dict,
    raw_preview_only: bool,
) -> None:
    if not st.session_state.get(LOCAL_LIVE_RUNNING_KEY, False):
        st.info("Choose `Local OpenCV Device`, set `device index = 4` for DroidCam, then enable `Keep USB stream active` or press `Start Local Stream`.")
        return

    try:
        frame = _read_local_capture_frame(local_device_index, max_width=int(quality_profile["max_width"]))
        rendered_frame, local_timings, local_warning = _render_local_capture_frame(
            frame,
            transformation=transformation,
            intensity=intensity,
            show_landmarks=show_landmarks,
            selected_regions=selected_regions or list(default_regions),
            reference_payload=reference_payload,
            transfer_method=transfer_method,
            quality_profile=quality_profile,
            raw_preview_only=raw_preview_only,
        )
        st.image(cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(
            f"Local OpenCV device {local_device_index} | frame {st.session_state.get('dashboard_local_frame_index', 0)} | "
            f"pipeline {local_timings.get('pipeline_total_ms', 0.0):.1f} ms"
        )
        if local_warning:
            st.warning(local_warning)
    except Exception as exc:
        st.error(f"Local OpenCV stream failed: {exc}")
        st.session_state[LOCAL_LIVE_RUNNING_KEY] = False
        _release_local_capture()
        _reset_local_pipeline_state()


if hasattr(st, "fragment"):
    @st.fragment(run_every=LOCAL_STREAM_FRAGMENT_INTERVAL_SECONDS)
    def _render_local_stream_fragment(
        local_device_index: int,
        transformation: str,
        intensity: float,
        show_landmarks: bool,
        selected_regions: list[str],
        default_regions: list[str],
        reference_payload: dict | None,
        transfer_method: str,
        quality_profile: dict,
        raw_preview_only: bool,
    ) -> None:
        _render_local_stream_frame(
            local_device_index=local_device_index,
            transformation=transformation,
            intensity=intensity,
            show_landmarks=show_landmarks,
            selected_regions=selected_regions,
            default_regions=default_regions,
            reference_payload=reference_payload,
            transfer_method=transfer_method,
            quality_profile=quality_profile,
            raw_preview_only=raw_preview_only,
        )
else:
    def _render_local_stream_fragment(
        local_device_index: int,
        transformation: str,
        intensity: float,
        show_landmarks: bool,
        selected_regions: list[str],
        default_regions: list[str],
        reference_payload: dict | None,
        transfer_method: str,
        quality_profile: dict,
        raw_preview_only: bool,
    ) -> None:
        _render_local_stream_frame(
            local_device_index=local_device_index,
            transformation=transformation,
            intensity=intensity,
            show_landmarks=show_landmarks,
            selected_regions=selected_regions,
            default_regions=default_regions,
            reference_payload=reference_payload,
            transfer_method=transfer_method,
            quality_profile=quality_profile,
            raw_preview_only=raw_preview_only,
        )
        if st.session_state.get(LOCAL_LIVE_RUNNING_KEY, False):
            time.sleep(0.12 if transformation == "reference_expression_transfer" else 0.08)
            st.rerun()


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
        self.previous_bbox = None
        self.previous_transformation = "aging"
        self.max_width = 640
        self.frame_skip_interval = 2
        self.transfer_method = DEFAULT_TRANSFER_METHOD
        self.realtime_profile = dict(STREAM_QUALITY_PROFILES["Balanced"])
        self.last_stats = None
        self.stats_history: list[dict[str, float]] = []
        self.max_stats_history = 30
        self.raw_preview_only = False

    def recv(self, frame):  # pragma: no cover - exercised through Streamlit runtime
        recv_start = time.perf_counter()
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
                realtime_profile = dict(self.realtime_profile)
                raw_preview_only = self.raw_preview_only
            if transformation != self.previous_transformation:
                self.previous_landmarks = None
                self.previous_bbox = None
                self.last_output = None
                self.previous_transformation = transformation
            self.frame_index += 1
            resize_start = time.perf_counter()
            resized_image = _resize_frame_for_realtime(image, max_width=max_width)
            resize_ms = float((time.perf_counter() - resize_start) * 1000.0)
            effective_skip_interval = frame_skip_interval
            if transformation == "reference_expression_transfer":
                effective_skip_interval = max(
                    frame_skip_interval,
                    int(realtime_profile.get("reference_update_interval", frame_skip_interval)),
                )
            should_reuse_last_frame = (
                effective_skip_interval > 1
                and self.last_output is not None
                and self.frame_index % effective_skip_interval != 1
            )
            if should_reuse_last_frame:
                output = self.last_output.copy()
                last_pipeline_total = self.last_stats["pipeline_total_ms"] if self.last_stats else 0.0
                self.last_stats = {
                    **(self.last_stats or {}),
                    "frame_index": float(self.frame_index),
                    "frame_reused": 1.0,
                    "resize_ms": resize_ms,
                    "capture_ms": float((time.perf_counter() - recv_start) * 1000.0),
                    "estimated_streamlit_render_ms": max(0.0, float((time.perf_counter() - recv_start) * 1000.0) - last_pipeline_total),
                }
                self.stats_history.append(self.last_stats)
                if len(self.stats_history) > self.max_stats_history:
                    self.stats_history = self.stats_history[-self.max_stats_history :]
            else:
                if raw_preview_only or transformation == "raw_camera_debug":
                    output = resized_image.copy()
                    cv2.putText(
                        output,
                        "Raw camera debug - processing bypassed",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    self.last_output = output.copy()
                    self.previous_landmarks = None
                    self.previous_bbox = None
                    pipeline_total_ms = float((time.perf_counter() - recv_start) * 1000.0)
                    self.last_stats = {
                        "frame_prepare_ms": 0.0,
                        "face_detect_ms": 0.0,
                        "landmark_detect_ms": 0.0,
                        "transform_ms": 0.0,
                        "overlay_ms": 0.0,
                        "compose_ms": 0.0,
                        "pipeline_total_ms": pipeline_total_ms,
                        "frame_index": float(self.frame_index),
                        "frame_reused": 0.0,
                        "resize_ms": resize_ms,
                        "capture_ms": pipeline_total_ms,
                        "estimated_streamlit_render_ms": 0.0,
                        "detection_reused": 0.0,
                    }
                else:
                    result = run_realtime_frame_pipeline(
                        resized_image,
                        transformation=transformation,
                        intensity=intensity,
                        show_landmarks=show_landmarks,
                        selected_regions=selected_regions,
                        reference_payload=reference_payload,
                        previous_landmarks=self.previous_landmarks,
                        previous_bbox=self.previous_bbox,
                        frame_index=self.frame_index,
                        smoothing_alpha=float(realtime_profile["smoothing_alpha"]),
                        transfer_method=transfer_method,
                        realtime_profile=realtime_profile,
                    )
                    output = result["composite_frame"]
                    self.last_output = output.copy()
                    self.previous_landmarks = result["landmarks"]["landmarks"] if result.get("landmarks") else None
                    self.previous_bbox = result["face_detection"]["bounding_box"]
                    self.last_stats = {
                        **result["timings"],
                        "frame_index": float(self.frame_index),
                        "frame_reused": 0.0,
                        "resize_ms": resize_ms,
                        "capture_ms": float((time.perf_counter() - recv_start) * 1000.0),
                        "estimated_streamlit_render_ms": max(
                            0.0,
                            float((time.perf_counter() - recv_start) * 1000.0) - result["timings"]["pipeline_total_ms"],
                        ),
                        "detection_reused": 1.0 if result["debug"]["reused_bbox"] else 0.0,
                    }
                self.stats_history.append(self.last_stats)
                if len(self.stats_history) > self.max_stats_history:
                    self.stats_history = self.stats_history[-self.max_stats_history :]
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


def _enhance_preview(rgb: np.ndarray, min_width: int = 720) -> np.ndarray:
    """Upscale small camera crops for clearer dashboard previews."""
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


def _save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def _save_bytes_to_temp(data: bytes, suffix: str = ".png") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(data)
        return temp_file.name


def _encode_bgr_image(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode captured camera frame.")
    return bytes(buffer)


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


def _summarize_realtime_stats(history: list[dict[str, float]] | None) -> dict[str, float] | None:
    if not history:
        return None
    keys = [
        "capture_ms",
        "resize_ms",
        "pipeline_total_ms",
        "face_detect_ms",
        "landmark_detect_ms",
        "transform_ms",
        "overlay_ms",
        "compose_ms",
        "estimated_streamlit_render_ms",
        "frame_reused",
        "detection_reused",
    ]
    summary: dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in history if key in item]
        if values:
            summary[key] = float(np.mean(values))
            summary[f"{key}_p95"] = float(np.percentile(values, 95))
    if summary.get("capture_ms", 0.0) > 0.0:
        summary["estimated_fps"] = 1000.0 / summary["capture_ms"]
    return summary


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
    if is_transfer and "transfer_quality_score" in metrics:
        _show_metric("Transfer Score", f"{metrics['transfer_quality_score']:.2f}")
        _show_metric("Expr Match", f"{metrics['expression_match_score']:.2f}")
        _show_metric("Identity Keep", f"{metrics['identity_preservation_score']:.2f}")

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
    selected_file = None
    selected_bytes = None
    if source_mode == "Camera capture":
        camera_profile_label = st.sidebar.selectbox("Camera profile", list(CAMERA_SOURCE_PROFILES.keys()), index=1, key="dashboard_camera_profile")
        camera_profile = CAMERA_SOURCE_PROFILES[camera_profile_label]
        capture_backend = st.sidebar.selectbox(
            "Capture backend",
            ["Browser Camera", "Local OpenCV Device"],
            index=1,
            key="dashboard_capture_backend",
        )
    else:
        camera_profile_label = "Built-in Camera"
        camera_profile = CAMERA_SOURCE_PROFILES[camera_profile_label]
        capture_backend = "Browser Camera"
    uploaded_file = st.sidebar.file_uploader("Source image", type=["jpg", "jpeg", "png"], key="dashboard_source") if source_mode == "Upload image" else None
    if source_mode == "Upload image":
        selected_file = uploaded_file
        st.session_state.pop(IMAGE_STUDIO_CAPTURE_FRAME_KEY, None)
    elif capture_backend == "Browser Camera":
        camera_file = st.sidebar.camera_input("Capture source frame", key="dashboard_camera")
        selected_file = camera_file
        if camera_file is not None:
            st.session_state.pop(IMAGE_STUDIO_CAPTURE_FRAME_KEY, None)
    else:
        local_capture_index = st.sidebar.number_input(
            "Capture device index",
            min_value=0,
            max_value=16,
            value=4,
            step=1,
            key="dashboard_capture_device_index",
        )
        preview_requested = st.sidebar.button("Preview USB Frame", key="dashboard_preview_usb_frame", use_container_width=True)
        capture_requested = st.sidebar.button("Capture From USB", key="dashboard_capture_usb_frame", use_container_width=True)
        if preview_requested or capture_requested:
            try:
                captured_frame = _read_local_capture_frame(local_capture_index, max_width=int(camera_profile["realtime_width"]))
                capture_payload = {
                    "bytes": _encode_bgr_image(captured_frame),
                    "rgb": cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB),
                    "device_index": int(local_capture_index),
                }
                st.session_state[IMAGE_STUDIO_CAPTURE_FRAME_KEY] = capture_payload
                if capture_requested:
                    st.sidebar.success(f"Captured frame from local device {local_capture_index}.")
            except Exception as exc:
                st.sidebar.error(f"Local USB capture failed: {exc}")
        stored_capture = st.session_state.get(IMAGE_STUDIO_CAPTURE_FRAME_KEY)
        if stored_capture is not None:
            selected_bytes = stored_capture["bytes"]

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
        transfer_method_label = st.sidebar.selectbox(
            "Transfer method",
            list(TRANSFER_METHOD_OPTIONS.keys()),
            index=list(TRANSFER_METHOD_OPTIONS.values()).index(DEFAULT_TRANSFER_METHOD),
            key="dashboard_transfer_method",
        )
        transfer_method = TRANSFER_METHOD_OPTIONS[transfer_method_label]
    else:
        transfer_method = DEFAULT_TRANSFER_METHOD

    st.markdown("## Image Studio")
    st.caption("Left panel controls the pipeline. Center shows source and transformed previews. Right panel exposes engineering metrics and explanations.")
    if source_mode == "Camera capture":
        if capture_backend == "Local OpenCV Device":
            st.info(
                f"Camera profile: `{camera_profile_label}`. Local OpenCV capture is enabled for device-index based USB/DroidCam snapshots. "
                "Use `Preview USB Frame` to verify the feed and `Capture From USB` to push the frame into Image Studio."
            )
            stored_capture = st.session_state.get(IMAGE_STUDIO_CAPTURE_FRAME_KEY)
            if stored_capture is not None:
                st.markdown("### Captured Camera Preview")
                st.image(_enhance_preview(stored_capture["rgb"]), caption=f"Local device {stored_capture['device_index']}", use_container_width=True)
        else:
            st.info(
                f"Camera profile: `{camera_profile_label}`. {camera_profile['description']} "
                "If multiple cameras exist, pick the external device from the browser camera selector before capture."
            )

    if selected_file is None and selected_bytes is None:
        st.info("Provide a source face image from upload or camera capture.")
        return
    if transformation == "reference_expression_transfer" and reference_file is None:
        st.info("Reference expression transfer requires a reference face image.")
        return

    source_temp = None
    reference_temp = None
    try:
        if selected_bytes is not None:
            source_temp = _save_bytes_to_temp(selected_bytes, suffix=".png")
        else:
            source_temp = _save_uploaded_file(selected_file)
        comparison = None
        if reference_file is not None:
            reference_temp = _save_uploaded_file(reference_file)
            if transfer_method == "auto_compare":
                comparison = compare_reference_expression_transfer_methods(
                    source_temp,
                    reference_temp,
                    blend_factor=intensity,
                    show_landmarks=show_landmarks,
                    selected_regions=selected_regions or list(DEFAULT_TRANSFER_REGIONS),
                )
                result = comparison["results"][comparison["best_method"]]
            else:
                result = run_reference_expression_transfer_pipeline(
                    source_temp,
                    reference_temp,
                    blend_factor=intensity,
                    show_landmarks=show_landmarks,
                    selected_regions=selected_regions or list(DEFAULT_TRANSFER_REGIONS),
                    transfer_method=transfer_method,
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
            st.image(_enhance_preview(source_preview), use_container_width=True)
            if show_landmarks:
                landmark_payload = result["source_landmarks"]["visualization"] if is_transfer else result["landmarks"]["visualization"]
                st.markdown("#### Landmark Overlay")
                st.image(_enhance_preview(_image_dict_to_rgb(landmark_payload)), use_container_width=True)
        with transformed_col:
            st.markdown("### Transformed Preview")
            st.image(_enhance_preview(transformed_preview), use_container_width=True)
            if reference_preview is not None:
                st.markdown("#### Reference Preview")
                st.image(_enhance_preview(reference_preview), use_container_width=True)
        with side_col:
            _render_status_panel(result, is_transfer=is_transfer)
            if is_transfer:
                _show_metric("Method", result.get("transfer_method", DEFAULT_TRANSFER_METHOD))
                if comparison is not None:
                    st.markdown("### Method Ranking")
                    st.table(
                        [
                            {
                                "Method": method,
                                "Score": f"{comparison['results'][method]['metrics']['transfer_quality_score']:.2f}",
                            }
                            for method in comparison["ranked_methods"]
                        ]
                    )

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
    live_backend = st.sidebar.selectbox("Live backend", ["Browser WebRTC", "Local OpenCV Device"], index=1, key="live_backend")
    transformation_label = st.sidebar.selectbox("Live transformation", list(TRANSFORMATION_OPTIONS.keys()), key="live_transform")
    transformation = TRANSFORMATION_OPTIONS[transformation_label]
    camera_profile_label = st.sidebar.selectbox(
        "Live camera profile",
        list(CAMERA_SOURCE_PROFILES.keys()),
        index=1,
        key="live_camera_profile",
    )
    camera_profile = CAMERA_SOURCE_PROFILES[camera_profile_label]
    intensity = st.sidebar.slider("Live intensity", min_value=0.0, max_value=1.0, value=0.55, step=0.05, key="live_intensity")
    show_landmarks = st.sidebar.toggle("Live landmarks", value=False, key="live_landmarks")
    default_regions = DEFAULT_TRANSFER_REGIONS if transformation == "reference_expression_transfer" else DEFAULT_LANDMARK_REGIONS
    selected_regions = st.sidebar.multiselect("Live regions", REGION_OPTIONS, default=list(default_regions), key="live_regions")
    quality_label = st.sidebar.selectbox("Stream quality", list(STREAM_QUALITY_PROFILES.keys()), index=1, key="live_quality")
    quality_profile = STREAM_QUALITY_PROFILES[quality_label]
    raw_preview_only = st.sidebar.toggle("Raw preview only", value=transformation == "raw_camera_debug", key="live_raw_preview")
    local_device_index = st.sidebar.number_input("Local device index", min_value=0, max_value=16, value=4, step=1, key="live_local_device_index")
    persistent_usb_stream = st.sidebar.toggle(
        "Keep USB stream active",
        value=st.session_state.get(LOCAL_USB_PERSISTENT_KEY, True),
        key=LOCAL_USB_PERSISTENT_KEY,
        help="When enabled, the Local OpenCV USB camera restarts automatically after Streamlit reruns.",
    )

    if live_backend != "Local OpenCV Device":
        persistent_usb_stream = False
        st.session_state[LOCAL_USB_PERSISTENT_KEY] = False
        if st.session_state.get(LOCAL_LIVE_RUNNING_KEY, False):
            st.session_state[LOCAL_LIVE_RUNNING_KEY] = False
            _release_local_capture()
            _reset_local_pipeline_state()
    elif persistent_usb_stream:
        st.session_state[LOCAL_LIVE_RUNNING_KEY] = True

    reference_payload = None
    reference_temp = None
    if transformation == "reference_expression_transfer":
        reference_file = st.sidebar.file_uploader("Live reference image", type=["jpg", "jpeg", "png"], key="live_reference")
        live_transfer_method_label = st.sidebar.selectbox(
            "Live transfer method",
            [label for label, value in TRANSFER_METHOD_OPTIONS.items() if value != "auto_compare"],
            index=[value for value in TRANSFER_METHOD_OPTIONS.values() if value != "auto_compare"].index(DEFAULT_TRANSFER_METHOD),
            key="live_transfer_method",
        )
        transfer_method = TRANSFER_METHOD_OPTIONS[live_transfer_method_label]
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
    else:
        transfer_method = DEFAULT_TRANSFER_METHOD

    st.markdown("## Real-Time Lab")
    st.caption("Live preview runs in a latency-oriented pipeline. Snapshot analysis below uses the full offline Image Studio path for higher-quality metrics and exports.")
    st.info(
        f"Live camera profile: `{camera_profile_label}`. {camera_profile['description']} "
        "For DroidCam USB, start DroidCam first, then click `SELECT DEVICE` in the player and choose the DroidCam device. "
        "This profile uses compatibility mode, so the browser can open the selected external camera with fewer constraint conflicts."
    )
    st.caption("Recommended USB setup: `Live backend = Local OpenCV Device`, `camera profile = DroidCam USB`, `device index = 4`, and `Keep USB stream active = On`.")
    st.caption("Browser `SELECT DEVICE` is optional now. For persistent USB capture, prefer the Local OpenCV backend.")

    top_col, side_col = st.columns([2.2, 1.0])
    with top_col:
        if live_backend == "Browser WebRTC" and STREAMLIT_WEBRTC_AVAILABLE:
            webrtc_ctx = webrtc_streamer(
                key="modern-facial-live-stream",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints=_build_live_media_constraints(quality_profile, camera_profile),
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
                    webrtc_ctx.video_processor.max_width = quality_profile["max_width"]
                    webrtc_ctx.video_processor.frame_skip_interval = quality_profile["frame_skip_interval"]
                    webrtc_ctx.video_processor.transfer_method = transfer_method
                    webrtc_ctx.video_processor.realtime_profile = dict(quality_profile)
                    webrtc_ctx.video_processor.raw_preview_only = raw_preview_only
                if webrtc_ctx.video_processor.last_error:
                    st.warning(webrtc_ctx.video_processor.last_error)
        elif live_backend == "Browser WebRTC":
            st.warning("Install `streamlit-webrtc` and `av` to enable browser real-time streaming.")
            st.code("pip install -e \".[dev]\"", language="bash")
        else:
            st.markdown("### Local OpenCV Stream")
            st.caption("This mode bypasses browser camera access and keeps the selected USB/OpenCV camera active directly inside Streamlit.")
            control_col1, control_col2 = st.columns([1, 1])
            with control_col1:
                if st.button("Start Local Stream", key="start_local_stream", use_container_width=True):
                    st.session_state[LOCAL_LIVE_RUNNING_KEY] = True
                    st.session_state[LOCAL_USB_PERSISTENT_KEY] = True
                    _reset_local_pipeline_state()
            with control_col2:
                if st.button("Release USB Camera", key="stop_local_stream", use_container_width=True):
                    st.session_state[LOCAL_LIVE_RUNNING_KEY] = False
                    st.session_state[LOCAL_USB_PERSISTENT_KEY] = False
                    _release_local_capture()
                    _reset_local_pipeline_state()
            if st.session_state.get(LOCAL_USB_PERSISTENT_KEY, False):
                st.success(f"Persistent USB stream active on device index {local_device_index}. The stream will auto-recover after normal Streamlit reruns.")
            _render_local_stream_fragment(
                local_device_index=local_device_index,
                transformation=transformation,
                intensity=intensity,
                show_landmarks=show_landmarks,
                selected_regions=selected_regions,
                default_regions=list(default_regions),
                reference_payload=reference_payload,
                transfer_method=transfer_method,
                quality_profile=quality_profile,
                raw_preview_only=raw_preview_only,
            )

        st.markdown("### Snapshot Analysis")
        if STREAMLIT_WEBRTC_AVAILABLE:
            st.info(
                "Live stream and Streamlit `camera_input` should not access the same webcam at the same time. "
                "Use Image Studio > Camera Capture for a fresh snapshot, or upload a saved frame below."
            )
            snapshot = st.file_uploader(
                "Upload a snapshot frame for metrics, Fourier analysis, and exports.",
                type=["jpg", "jpeg", "png"],
                key="realtime_snapshot_upload",
            )
        else:
            snapshot = st.camera_input(
                "Capture a frame for metrics, Fourier analysis, and exports.",
                key="realtime_snapshot",
            )
    with side_col:
        st.markdown("### Live Status")
        _show_metric("Backend", live_backend)
        _show_metric("Transformation", transformation)
        _show_metric("Intensity", f"{intensity:.2f}")
        _show_metric("Landmark Overlay", "On" if show_landmarks else "Off")
        _show_metric("Region Count", str(len(selected_regions or default_regions)))
        _show_metric("Stream Quality", quality_label)
        _show_metric("Camera Profile", camera_profile_label)
        if live_backend == "Local OpenCV Device":
            _show_metric("Device Index", str(local_device_index))
            _show_metric("Persistent USB", "On" if st.session_state.get(LOCAL_USB_PERSISTENT_KEY, False) else "Off")
        _show_metric("Detect Every", f"{quality_profile['detection_interval']} frame")
        _show_metric("Analysis Face", f"{quality_profile['analysis_size']} px")
        if transformation == "reference_expression_transfer":
            _show_metric("Method", transfer_method)
        if reference_payload is not None:
            _show_metric("Reference Face", "Ready")
            st.image(_enhance_preview(_image_dict_to_rgb(reference_payload["face_detection"]["face_image"])), caption="Reference ROI", use_container_width=True)
        if STREAMLIT_WEBRTC_AVAILABLE and "webrtc_ctx" in locals() and webrtc_ctx.video_processor:
            stats_summary = _summarize_realtime_stats(webrtc_ctx.video_processor.stats_history)
            if stats_summary is not None:
                st.markdown("### Live Profiler")
                _show_metric("Avg Total", f"{stats_summary['capture_ms']:.1f} ms")
                _show_metric("p95 Total", f"{stats_summary['capture_ms_p95']:.1f} ms")
                _show_metric("Estimated FPS", f"{stats_summary['estimated_fps']:.1f}")
                _show_metric("Resize", f"{stats_summary.get('resize_ms', 0.0):.1f} ms")
                _show_metric("Detect", f"{stats_summary.get('face_detect_ms', 0.0):.1f} ms")
                _show_metric("Landmark", f"{stats_summary.get('landmark_detect_ms', 0.0):.1f} ms")
                _show_metric("Transform", f"{stats_summary.get('transform_ms', 0.0):.1f} ms")
                _show_metric("Overlay", f"{stats_summary.get('overlay_ms', 0.0):.1f} ms")
                _show_metric("Streamlit Overhead", f"{stats_summary.get('estimated_streamlit_render_ms', 0.0):.1f} ms")
                _show_metric("BBox Reuse", f"{stats_summary.get('detection_reused', 0.0) * 100:.0f}%")
                _show_metric("Frame Reuse", f"{stats_summary.get('frame_reused', 0.0) * 100:.0f}%")
                st.caption("`Streamlit Overhead` is an estimated UI/transport cost measured outside the backend pipeline.")

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
                    transfer_method=transfer_method,
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
                st.image(_enhance_preview(source_preview), use_container_width=True)
            with preview_col2:
                st.markdown("### Snapshot Transformed")
                st.image(_enhance_preview(transformed_preview), use_container_width=True)
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



