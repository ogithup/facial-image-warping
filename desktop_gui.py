from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import app as pipeline_app


TRANSFORMATION_OPTIONS = {
    "Smile Enhancement": "smile_enhancement",
    "Eyebrow Raising": "eyebrow_raising",
    "Lip Widening": "lip_widening",
    "Face Slimming": "face_slimming",
    "Aging": "aging",
    "De-Aging": "de-aging",
    "Reference Expression Transfer": "reference_expression_transfer",
    "Raw Camera Debug": "raw_camera_debug",
}
TRANSFER_METHOD_OPTIONS = {
    "Recommended Coefficients": "expression_coefficients",
    "Thin Plate Spline": "tps",
    "Safe Classical": "safe_classical",
    "Auto Compare All": "auto_compare",
}
REGION_OPTIONS = ["eyes", "eyebrows", "nose", "lips", "jawline", "cheeks"]
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
IMAGE_PREVIEW_MAX_SIZES = {
    "source": (760, 520),
    "transformed": (760, 520),
    "overlay": (640, 460),
    "reference": (640, 460),
    "fourier_before": (640, 460),
    "difference": (760, 460),
    "live": (1200, 760),
}


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


def _resize_rgb(rgb: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    max_width, max_height = max_size
    height, width = rgb.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return rgb
    return cv2.resize(
        rgb,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _preview_photo(rgb: np.ndarray, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
    resized = _resize_rgb(rgb, max_size=max_size)
    return ImageTk.PhotoImage(Image.fromarray(resized))


def _read_rgb_path(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_frame_for_realtime(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / max(width, 1)
    return cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _save_bgr_frame_to_temp(frame: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", frame)
    if not success:
        raise ValueError("Failed to encode frame to PNG.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(bytes(buffer))
        return temp_file.name


def _format_metrics(result: dict, is_transfer: bool) -> str:
    metrics = result["metrics"]
    face_detection = result["source_face_detection"] if is_transfer else result["face_detection"]
    bbox = face_detection["bounding_box"]
    lines = [
        "Analysis Panel",
        "",
        f"Bounding Box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}",
        f"MSE: {metrics['mse']:.4f}",
        f"PSNR: {metrics['psnr']:.4f}",
        f"SSIM: {metrics['ssim']:.4f}",
        f"Original High/Low: {metrics['original_high_low_ratio']:.6f}",
        f"Transformed High/Low: {metrics['transformed_high_low_ratio']:.6f}",
    ]
    if is_transfer:
        lines.extend(
            [
                f"Transfer Score: {metrics.get('transfer_quality_score', 0.0):.2f}",
                f"Expression Match: {metrics.get('expression_match_score', 0.0):.2f}",
                f"Identity Preserve: {metrics.get('identity_preservation_score', 0.0):.2f}",
                f"Method: {result.get('transfer_method', pipeline_app.DEFAULT_TRANSFER_METHOD)}",
            ]
        )
    return "\n".join(lines)


def _copy_if_exists(source: str | Path | None, destination: str | Path) -> None:
    if not source:
        raise FileNotFoundError("No source file available to export.")
    shutil.copyfile(source, destination)


@dataclass
class ExportState:
    transformed_rgb: np.ndarray | None = None
    metrics_csv_path: str | None = None
    difference_path: str | None = None


class ScrollableTab(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.content = ttk.Frame(self.canvas, padding=4)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_content_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if self.winfo_exists():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class DesktopDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Facial Image Warping Desktop Dashboard")
        self.geometry("1800x1120")
        self.minsize(1440, 920)

        self.temp_paths: list[str] = []
        self.source_path: str | None = None
        self.reference_path: str | None = None
        self.captured_source_path: str | None = None
        self.captured_source_rgb: np.ndarray | None = None
        self.export_state = ExportState()

        self.live_capture: cv2.VideoCapture | None = None
        self.live_running = False
        self.live_worker: threading.Thread | None = None
        self.live_lock = threading.Lock()
        self.live_display_rgb: np.ndarray | None = None
        self.live_status_text = "Idle"
        self.live_snapshot_frame: np.ndarray | None = None
        self.live_processor_state: dict[str, object] = {
            "previous_landmarks": None,
            "previous_bbox": None,
            "frame_index": 0,
        }

        self._build_variables()
        self._build_ui()
        self._schedule_live_refresh()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _format_intensity_percent(value: float) -> str:
        return f"{int(round(float(value) * 100.0))}%"

    def _on_image_intensity_change(self, *_args) -> None:
        self.image_intensity_text.set(self._format_intensity_percent(self.image_intensity.get()))

    def _on_live_intensity_change(self, *_args) -> None:
        self.live_intensity_text.set(self._format_intensity_percent(self.live_intensity.get()))

    def _build_variables(self) -> None:
        self.image_source_mode = tk.StringVar(value="upload")
        self.image_transformation_label = tk.StringVar(value="Smile Enhancement")
        self.image_intensity = tk.DoubleVar(value=0.55)
        self.image_intensity_text = tk.StringVar(value=self._format_intensity_percent(self.image_intensity.get()))
        self.image_show_landmarks = tk.BooleanVar(value=True)
        self.image_transfer_method_label = tk.StringVar(value="Recommended Coefficients")
        self.image_camera_index = tk.IntVar(value=4)
        self.image_status = tk.StringVar(value="Ready")

        self.live_transformation_label = tk.StringVar(value="Raw Camera Debug")
        self.live_intensity = tk.DoubleVar(value=0.55)
        self.live_intensity_text = tk.StringVar(value=self._format_intensity_percent(self.live_intensity.get()))
        self.live_show_landmarks = tk.BooleanVar(value=False)
        self.live_transfer_method_label = tk.StringVar(value="Recommended Coefficients")
        self.live_quality_label = tk.StringVar(value="Fast")
        self.live_device_index = tk.IntVar(value=4)
        self.live_raw_preview_only = tk.BooleanVar(value=True)
        self.live_status_var = tk.StringVar(value="Idle")

        default_image_regions = pipeline_app.DEFAULT_LANDMARK_REGIONS
        default_transfer_regions = pipeline_app.DEFAULT_TRANSFER_REGIONS
        self.image_region_vars = {
            region: tk.BooleanVar(value=region in default_image_regions or region in default_transfer_regions)
            for region in REGION_OPTIONS
        }
        self.live_region_vars = {
            region: tk.BooleanVar(value=region in default_image_regions or region in default_transfer_regions)
            for region in REGION_OPTIONS
        }
        self.image_intensity.trace_add("write", self._on_image_intensity_change)
        self.live_intensity.trace_add("write", self._on_live_intensity_change)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_tab_container = ScrollableTab(notebook)
        self.live_tab_container = ScrollableTab(notebook)
        self.image_tab = self.image_tab_container.content
        self.live_tab = self.live_tab_container.content
        notebook.add(self.image_tab_container, text="Image Studio")
        notebook.add(self.live_tab_container, text="Real-Time Lab")

        self._build_image_tab()
        self._build_live_tab()

    def _build_image_tab(self) -> None:
        self.image_tab.columnconfigure(1, weight=1)
        self.image_tab.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(self.image_tab, text="Controls", padding=12)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        workspace = ttk.PanedWindow(self.image_tab, orient="horizontal")
        workspace.grid(row=0, column=1, sticky="nsew")

        center_panel = ttk.Frame(workspace, padding=8)
        center_panel.columnconfigure(0, weight=1)
        center_panel.rowconfigure(0, weight=1)
        center_panel.rowconfigure(1, weight=1)

        side_panel = ttk.LabelFrame(workspace, text="Metrics", padding=8)
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(0, weight=1)

        workspace.add(center_panel, weight=5)
        workspace.add(side_panel, weight=2)

        top_previews = ttk.PanedWindow(center_panel, orient="horizontal")
        top_previews.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        source_frame = ttk.LabelFrame(top_previews, text="Source Preview", padding=8)
        transformed_frame = ttk.LabelFrame(top_previews, text="Transformed Preview", padding=8)
        for frame in (source_frame, transformed_frame):
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
        top_previews.add(source_frame, weight=1)
        top_previews.add(transformed_frame, weight=1)

        bottom_notebook = ttk.Notebook(center_panel)
        bottom_notebook.grid(row=1, column=0, sticky="nsew")

        ttk.Label(controls, text="Source Input").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Upload Image", value="upload", variable=self.image_source_mode, command=self._toggle_image_source_mode).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Camera Capture", value="camera", variable=self.image_source_mode, command=self._toggle_image_source_mode).grid(row=2, column=0, sticky="w")

        ttk.Button(controls, text="Choose Source Image", command=self._choose_source_image).grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.source_path_label = ttk.Label(controls, text="No source image selected.", wraplength=260)
        self.source_path_label.grid(row=4, column=0, sticky="w", pady=(4, 10))

        ttk.Label(controls, text="Camera Device Index").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(controls, from_=0, to=16, textvariable=self.image_camera_index, width=8).grid(row=6, column=0, sticky="w")
        ttk.Button(controls, text="Preview Camera Frame", command=self._preview_camera_source).grid(row=7, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Capture From Camera", command=self._capture_camera_source).grid(row=8, column=0, sticky="ew", pady=(6, 10))

        ttk.Label(controls, text="Transformation").grid(row=9, column=0, sticky="w")
        self.image_transformation_combo = ttk.Combobox(
            controls,
            state="readonly",
            values=list(TRANSFORMATION_OPTIONS.keys())[:-1],
            textvariable=self.image_transformation_label,
        )
        self.image_transformation_combo.grid(row=10, column=0, sticky="ew")
        self.image_transformation_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_reference_controls())

        intensity_row = ttk.Frame(controls)
        intensity_row.grid(row=11, column=0, sticky="ew", pady=(8, 0))
        intensity_row.columnconfigure(0, weight=1)
        ttk.Label(intensity_row, text="Intensity").grid(row=0, column=0, sticky="w")
        ttk.Label(intensity_row, textvariable=self.image_intensity_text).grid(row=0, column=1, sticky="e")
        ttk.Scale(controls, from_=0.0, to=1.0, variable=self.image_intensity, orient="horizontal").grid(row=12, column=0, sticky="ew")
        ttk.Checkbutton(controls, text="Show landmarks", variable=self.image_show_landmarks).grid(row=13, column=0, sticky="w", pady=(6, 0))

        ttk.Label(controls, text="Regions").grid(row=14, column=0, sticky="w", pady=(8, 0))
        image_regions_frame = ttk.Frame(controls)
        image_regions_frame.grid(row=15, column=0, sticky="w")
        for index, region in enumerate(REGION_OPTIONS):
            ttk.Checkbutton(image_regions_frame, text=region, variable=self.image_region_vars[region]).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 10))

        ttk.Separator(controls).grid(row=16, column=0, sticky="ew", pady=10)
        ttk.Label(controls, text="Reference Image").grid(row=17, column=0, sticky="w")
        ttk.Button(controls, text="Choose Reference Image", command=self._choose_reference_image).grid(row=18, column=0, sticky="ew")
        self.reference_path_label = ttk.Label(controls, text="No reference image selected.", wraplength=260)
        self.reference_path_label.grid(row=19, column=0, sticky="w", pady=(4, 8))

        ttk.Label(controls, text="Transfer Method").grid(row=20, column=0, sticky="w")
        self.image_transfer_method_combo = ttk.Combobox(
            controls,
            state="readonly",
            values=list(TRANSFER_METHOD_OPTIONS.keys()),
            textvariable=self.image_transfer_method_label,
        )
        self.image_transfer_method_combo.grid(row=21, column=0, sticky="ew")

        ttk.Button(controls, text="Run Pipeline", command=self._run_image_pipeline).grid(row=22, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(controls, text="Save Transformed PNG", command=self._save_transformed_png).grid(row=23, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Save Metrics CSV", command=self._save_metrics_csv).grid(row=24, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Save Difference PNG", command=self._save_difference_png).grid(row=25, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(controls, textvariable=self.image_status, wraplength=260, foreground="#1f3f7f").grid(row=26, column=0, sticky="w", pady=(10, 0))

        self.image_labels = {}

        source_label = ttk.Label(source_frame, text="No image", anchor="center", justify="center")
        source_label.grid(row=0, column=0, sticky="nsew")
        self.image_labels["source"] = source_label

        transformed_label = ttk.Label(transformed_frame, text="No image", anchor="center", justify="center")
        transformed_label.grid(row=0, column=0, sticky="nsew")
        self.image_labels["transformed"] = transformed_label

        tab_panels = [
            ("overlay", "Landmark Overlay"),
            ("reference", "Reference Preview"),
            ("fourier_before", "Fourier Before"),
            ("difference", "Absolute Difference"),
        ]
        for key, title in tab_panels:
            tab = ttk.Frame(bottom_notebook, padding=8)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
            label = ttk.Label(tab, text="No image", anchor="center", justify="center")
            label.grid(row=0, column=0, sticky="nsew")
            self.image_labels[key] = label
            bottom_notebook.add(tab, text=title)

        metrics_frame = ttk.Frame(side_panel)
        metrics_frame.grid(row=0, column=0, sticky="nsew")
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(0, weight=1)

        metrics_scroll = ttk.Scrollbar(metrics_frame, orient="vertical")
        self.image_metrics_text = tk.Text(
            metrics_frame,
            width=36,
            height=36,
            wrap="word",
            yscrollcommand=metrics_scroll.set,
        )
        metrics_scroll.configure(command=self.image_metrics_text.yview)
        self.image_metrics_text.grid(row=0, column=0, sticky="nsew")
        metrics_scroll.grid(row=0, column=1, sticky="ns")
        self.image_metrics_text.insert("1.0", "Run the pipeline to see metrics and method ranking.")
        self.image_metrics_text.configure(state="disabled")

        self._sync_reference_controls()

    def _build_live_tab(self) -> None:
        self.live_tab.columnconfigure(1, weight=1)
        self.live_tab.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.live_tab, padding=12)
        controls.grid(row=0, column=0, sticky="nsw")

        content = ttk.Frame(self.live_tab, padding=12)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        ttk.Label(controls, text="Live Device Index").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(controls, from_=0, to=16, textvariable=self.live_device_index, width=8).grid(row=1, column=0, sticky="w")

        ttk.Label(controls, text="Live Transformation").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.live_transformation_combo = ttk.Combobox(
            controls,
            state="readonly",
            values=list(TRANSFORMATION_OPTIONS.keys()),
            textvariable=self.live_transformation_label,
        )
        self.live_transformation_combo.grid(row=3, column=0, sticky="ew")
        self.live_transformation_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_live_reference_controls())

        ttk.Label(controls, text="Stream Quality").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(controls, state="readonly", values=list(STREAM_QUALITY_PROFILES.keys()), textvariable=self.live_quality_label).grid(row=5, column=0, sticky="ew")

        live_intensity_row = ttk.Frame(controls)
        live_intensity_row.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        live_intensity_row.columnconfigure(0, weight=1)
        ttk.Label(live_intensity_row, text="Intensity").grid(row=0, column=0, sticky="w")
        ttk.Label(live_intensity_row, textvariable=self.live_intensity_text).grid(row=0, column=1, sticky="e")
        ttk.Scale(controls, from_=0.0, to=1.0, variable=self.live_intensity, orient="horizontal").grid(row=7, column=0, sticky="ew")
        ttk.Checkbutton(controls, text="Show landmarks", variable=self.live_show_landmarks).grid(row=8, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(controls, text="Raw preview only", variable=self.live_raw_preview_only).grid(row=9, column=0, sticky="w")

        ttk.Label(controls, text="Live Regions").grid(row=10, column=0, sticky="w", pady=(10, 0))
        live_regions_frame = ttk.Frame(controls)
        live_regions_frame.grid(row=11, column=0, sticky="w")
        for index, region in enumerate(REGION_OPTIONS):
            ttk.Checkbutton(live_regions_frame, text=region, variable=self.live_region_vars[region]).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 10))

        ttk.Separator(controls).grid(row=12, column=0, sticky="ew", pady=10)
        ttk.Button(controls, text="Choose Live Reference Image", command=self._choose_reference_image).grid(row=13, column=0, sticky="ew")
        self.live_reference_path_label = ttk.Label(controls, text="Shared reference image: none", wraplength=260)
        self.live_reference_path_label.grid(row=14, column=0, sticky="w", pady=(4, 8))

        ttk.Label(controls, text="Live Transfer Method").grid(row=15, column=0, sticky="w")
        self.live_transfer_method_combo = ttk.Combobox(
            controls,
            state="readonly",
            values=[label for label, value in TRANSFER_METHOD_OPTIONS.items() if value != "auto_compare"],
            textvariable=self.live_transfer_method_label,
        )
        self.live_transfer_method_combo.grid(row=16, column=0, sticky="ew")

        ttk.Button(controls, text="Start Live", command=self._start_live).grid(row=17, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(controls, text="Stop Live", command=self._stop_live).grid(row=18, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Analyze Current Frame in Image Studio", command=self._analyze_live_snapshot).grid(row=19, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(controls, textvariable=self.live_status_var, wraplength=260, foreground="#1f3f7f").grid(row=20, column=0, sticky="w", pady=(10, 0))

        preview_frame = ttk.LabelFrame(content, text="Live Preview", padding=8)
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.live_image_label = ttk.Label(preview_frame, text="Live stream is stopped.", anchor="center")
        self.live_image_label.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.LabelFrame(content, text="Live Status", padding=8)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.live_text = tk.Text(status_frame, width=110, height=8, wrap="word")
        self.live_text.pack(fill="both", expand=True)
        self.live_text.insert("1.0", "Local OpenCV desktop live mode is ready.")
        self.live_text.configure(state="disabled")

        self._sync_live_reference_controls()
        self._sync_reference_labels()

    def _toggle_image_source_mode(self) -> None:
        if self.image_source_mode.get() == "upload":
            self.captured_source_path = None
            self.captured_source_rgb = None
            self.image_status.set("Switched to upload mode.")

    def _choose_source_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose source image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if path:
            self.source_path = path
            self.image_source_mode.set("upload")
            self.source_path_label.configure(text=path)
            self.image_status.set("Source image selected.")

    def _choose_reference_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose reference image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if path:
            self.reference_path = path
            self._sync_reference_labels()

    def _sync_reference_labels(self) -> None:
        text = self.reference_path or "No reference image selected."
        self.reference_path_label.configure(text=text)
        self.live_reference_path_label.configure(text=f"Shared reference image: {self.reference_path or 'none'}")

    def _sync_reference_controls(self) -> None:
        needs_reference = TRANSFORMATION_OPTIONS[self.image_transformation_label.get()] == "reference_expression_transfer"
        state = "readonly" if needs_reference else "disabled"
        self.image_transfer_method_combo.configure(state=state)

    def _sync_live_reference_controls(self) -> None:
        needs_reference = TRANSFORMATION_OPTIONS[self.live_transformation_label.get()] == "reference_expression_transfer"
        state = "readonly" if needs_reference else "disabled"
        self.live_transfer_method_combo.configure(state=state)

    def _selected_regions(self, vars_map: dict[str, tk.BooleanVar], default_regions: list[str]) -> list[str]:
        selected = [region for region, var in vars_map.items() if var.get()]
        return selected or list(default_regions)

    def _capture_single_frame(self, device_index: int, max_width: int) -> np.ndarray:
        capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open camera index {device_index}.")
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Camera index {device_index} did not return a frame.")
            return _resize_frame_for_realtime(frame, max_width=max_width)
        finally:
            capture.release()

    def _preview_camera_source(self) -> None:
        try:
            frame = self._capture_single_frame(self.image_camera_index.get(), max_width=640)
            self.captured_source_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._set_preview("source", self.captured_source_rgb)
            self.image_status.set(f"Previewed camera frame from device {self.image_camera_index.get()}.")
        except Exception as exc:
            messagebox.showerror("Camera Preview Failed", str(exc))

    def _capture_camera_source(self) -> None:
        try:
            frame = self._capture_single_frame(self.image_camera_index.get(), max_width=896)
            self.captured_source_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.captured_source_path = _save_bgr_frame_to_temp(frame)
            self.temp_paths.append(self.captured_source_path)
            self.image_source_mode.set("camera")
            self.source_path_label.configure(text=f"Captured from camera index {self.image_camera_index.get()}")
            self._set_preview("source", self.captured_source_rgb)
            self.image_status.set(f"Captured source frame from device {self.image_camera_index.get()}.")
        except Exception as exc:
            messagebox.showerror("Camera Capture Failed", str(exc))

    def _resolve_source_path(self) -> str | None:
        if self.image_source_mode.get() == "camera":
            return self.captured_source_path
        return self.source_path

    def _run_image_pipeline(self) -> None:
        source_path = self._resolve_source_path()
        if not source_path:
            messagebox.showinfo("Source Required", "Choose a source image or capture a camera frame first.")
            return

        transformation = TRANSFORMATION_OPTIONS[self.image_transformation_label.get()]
        show_landmarks = self.image_show_landmarks.get()
        default_regions = (
            pipeline_app.DEFAULT_TRANSFER_REGIONS
            if transformation == "reference_expression_transfer"
            else pipeline_app.DEFAULT_LANDMARK_REGIONS
        )
        selected_regions = self._selected_regions(self.image_region_vars, default_regions)
        intensity = float(self.image_intensity.get())
        transfer_method = TRANSFER_METHOD_OPTIONS[self.image_transfer_method_label.get()]

        if transformation == "reference_expression_transfer" and not self.reference_path:
            messagebox.showinfo("Reference Required", "Choose a reference image for reference expression transfer.")
            return

        self.image_status.set("Running pipeline...")
        threading.Thread(
            target=self._run_image_pipeline_worker,
            args=(source_path, transformation, intensity, show_landmarks, selected_regions, transfer_method),
            daemon=True,
        ).start()

    def _run_image_pipeline_worker(
        self,
        source_path: str,
        transformation: str,
        intensity: float,
        show_landmarks: bool,
        selected_regions: list[str],
        transfer_method: str,
    ) -> None:
        try:
            comparison = None
            if transformation == "reference_expression_transfer":
                if transfer_method == "auto_compare":
                    comparison = pipeline_app.compare_reference_expression_transfer_methods(
                        source_path,
                        self.reference_path,
                        blend_factor=intensity,
                        show_landmarks=show_landmarks,
                        selected_regions=selected_regions,
                    )
                    result = comparison["results"][comparison["best_method"]]
                else:
                    result = pipeline_app.run_reference_expression_transfer_pipeline(
                        source_path,
                        self.reference_path,
                        blend_factor=intensity,
                        show_landmarks=show_landmarks,
                        selected_regions=selected_regions,
                        transfer_method=transfer_method,
                    )
                is_transfer = True
            else:
                result = pipeline_app.run_analysis_pipeline(
                    source_path,
                    transformation=transformation,
                    intensity=intensity,
                    show_landmarks=show_landmarks,
                    selected_regions=selected_regions,
                )
                is_transfer = False
            self.after(0, lambda: self._render_image_result(result, is_transfer, comparison))
        except Exception as exc:
            self.after(0, lambda: self.image_status.set(f"Pipeline failed: {exc}"))
            self.after(0, lambda: messagebox.showerror("Pipeline Failed", str(exc)))

    def _render_image_result(self, result: dict, is_transfer: bool, comparison: dict | None) -> None:
        source_preview = _image_dict_to_rgb(result["source_face_detection"]["face_image"] if is_transfer else result["face_detection"]["face_image"])
        transformed_preview = _image_dict_to_rgb(result["transformation"]["image"])
        reference_preview = _image_dict_to_rgb(result["reference_face_detection"]["face_image"]) if is_transfer else None

        self._set_preview("source", source_preview)
        self._set_preview("transformed", transformed_preview)
        self._set_preview("reference", reference_preview)

        if is_transfer and result.get("source_landmarks") and self.image_show_landmarks.get():
            overlay = _image_dict_to_rgb(result["source_landmarks"]["visualization"])
        elif not is_transfer and result.get("landmarks") and self.image_show_landmarks.get():
            overlay = _image_dict_to_rgb(result["landmarks"]["visualization"])
        else:
            overlay = None
        self._set_preview("overlay", overlay)

        fourier_before = cv2.applyColorMap(
            cv2.normalize(result["original_frequency"]["magnitude_spectrum"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_MAGMA,
        )
        self._set_preview("fourier_before", cv2.cvtColor(fourier_before, cv2.COLOR_BGR2RGB))
        self._set_preview("difference", _read_rgb_path(result["metrics"].get("difference_path")))

        metrics_text = _format_metrics(result, is_transfer=is_transfer)
        if comparison is not None:
            ranking_lines = [
                "",
                "Method Ranking",
                *[
                    f"{method}: {comparison['results'][method]['metrics']['transfer_quality_score']:.2f}"
                    for method in comparison["ranked_methods"]
                ],
            ]
            metrics_text = metrics_text + "\n" + "\n".join(ranking_lines)
        self._set_text(self.image_metrics_text, metrics_text)

        self.export_state = ExportState(
            transformed_rgb=transformed_preview,
            metrics_csv_path=result["metrics"].get("csv_path"),
            difference_path=result["metrics"].get("difference_path"),
        )
        self.image_status.set("Pipeline completed.")

    def _set_preview(self, key: str, rgb: np.ndarray | None) -> None:
        label = self.image_labels[key] if key in self.image_labels else self.live_image_label
        if rgb is None:
            label.configure(text="No image", image="")
            label.image = None
            return
        preview_key = key if key in IMAGE_PREVIEW_MAX_SIZES else "live"
        photo = _preview_photo(rgb, max_size=IMAGE_PREVIEW_MAX_SIZES[preview_key])
        label.configure(image=photo, text="")
        label.image = photo

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _save_transformed_png(self) -> None:
        if self.export_state.transformed_rgb is None:
            messagebox.showinfo("Nothing to Save", "Run the image pipeline first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if path:
            bgr = cv2.cvtColor(self.export_state.transformed_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)

    def _save_metrics_csv(self) -> None:
        if not self.export_state.metrics_csv_path:
            messagebox.showinfo("Nothing to Save", "No metrics CSV available yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file", "*.csv")])
        if path:
            _copy_if_exists(self.export_state.metrics_csv_path, path)

    def _save_difference_png(self) -> None:
        if not self.export_state.difference_path:
            messagebox.showinfo("Nothing to Save", "No difference image available yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if path:
            _copy_if_exists(self.export_state.difference_path, path)

    def _start_live(self) -> None:
        if self.live_running:
            return
        self.live_running = True
        self.live_processor_state = {
            "previous_landmarks": None,
            "previous_bbox": None,
            "frame_index": 0,
        }
        self.live_status_var.set("Starting local OpenCV live stream...")
        self.live_worker = threading.Thread(target=self._live_worker_loop, daemon=True)
        self.live_worker.start()

    def _stop_live(self) -> None:
        self.live_running = False
        capture = self.live_capture
        self.live_capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        self.live_status_var.set("Live stream stopped.")

    def _build_live_reference_payload(self, show_landmarks: bool, selected_regions: list[str]) -> dict | None:
        transformation = TRANSFORMATION_OPTIONS[self.live_transformation_label.get()]
        if transformation != "reference_expression_transfer":
            return None
        if not self.reference_path:
            raise ValueError("Choose a reference image before starting reference expression transfer.")
        return pipeline_app.prepare_reference_expression_payload(
            self.reference_path,
            show_landmarks=show_landmarks,
            selected_regions=selected_regions,
        )

    def _live_worker_loop(self) -> None:
        device_index = int(self.live_device_index.get())
        capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            self.live_running = False
            self.after(0, lambda: self.live_status_var.set(f"Failed to open camera index {device_index}."))
            return
        self.live_capture = capture
        reference_signature: tuple | None = None
        reference_payload: dict | None = None

        try:
            while self.live_running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    time.sleep(0.03)
                    continue

                try:
                    quality_profile = STREAM_QUALITY_PROFILES[self.live_quality_label.get()]
                    transformation = TRANSFORMATION_OPTIONS[self.live_transformation_label.get()]
                    selected_regions = self._selected_regions(
                        self.live_region_vars,
                        pipeline_app.DEFAULT_TRANSFER_REGIONS if transformation == "reference_expression_transfer" else pipeline_app.DEFAULT_LANDMARK_REGIONS,
                    )
                    resized_frame = _resize_frame_for_realtime(frame, max_width=int(quality_profile["max_width"]))
                    self.live_snapshot_frame = resized_frame.copy()

                    if self.live_raw_preview_only.get() or transformation == "raw_camera_debug":
                        rendered = resized_frame.copy()
                        cv2.putText(rendered, "Raw camera debug - desktop GUI", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        status_text = f"Device {device_index} | raw preview | {rendered.shape[1]}x{rendered.shape[0]}"
                    else:
                        current_signature = (
                            self.reference_path,
                            tuple(selected_regions),
                            bool(self.live_show_landmarks.get()),
                            transformation,
                        )
                        if current_signature != reference_signature:
                            reference_payload = self._build_live_reference_payload(self.live_show_landmarks.get(), selected_regions)
                            reference_signature = current_signature
                        self.live_processor_state["frame_index"] = int(self.live_processor_state.get("frame_index", 0)) + 1
                        result = pipeline_app.run_realtime_frame_pipeline(
                            resized_frame,
                            transformation=transformation,
                            intensity=float(self.live_intensity.get()),
                            show_landmarks=bool(self.live_show_landmarks.get()),
                            selected_regions=selected_regions,
                            reference_payload=reference_payload,
                            previous_landmarks=self.live_processor_state.get("previous_landmarks"),
                            previous_bbox=self.live_processor_state.get("previous_bbox"),
                            frame_index=int(self.live_processor_state["frame_index"]),
                            smoothing_alpha=float(quality_profile["smoothing_alpha"]),
                            transfer_method=TRANSFER_METHOD_OPTIONS[self.live_transfer_method_label.get()],
                            realtime_profile=quality_profile,
                        )
                        rendered = result["composite_frame"]
                        self.live_processor_state["previous_landmarks"] = result["landmarks"]["landmarks"] if result.get("landmarks") else None
                        self.live_processor_state["previous_bbox"] = result["face_detection"]["bounding_box"]
                        timings = result["timings"]
                        status_text = (
                            f"Device {device_index} | frame {self.live_processor_state['frame_index']} | "
                            f"pipeline {timings.get('pipeline_total_ms', 0.0):.1f} ms | "
                            f"detect {timings.get('face_detect_ms', 0.0):.1f} ms | "
                            f"landmark {timings.get('landmark_detect_ms', 0.0):.1f} ms | "
                            f"transform {timings.get('transform_ms', 0.0):.1f} ms"
                        )

                    with self.live_lock:
                        self.live_display_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                        self.live_status_text = status_text
                    frame_rate = max(1, int(quality_profile["frame_rate"]))
                    time.sleep(1.0 / frame_rate)
                except Exception as exc:
                    fallback = resized_frame.copy() if "resized_frame" in locals() else frame.copy()
                    cv2.putText(fallback, str(exc)[:100], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    with self.live_lock:
                        self.live_display_rgb = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                        self.live_status_text = f"Live processing warning: {exc}"
                    time.sleep(0.08)
        finally:
            capture.release()
            if self.live_capture is capture:
                self.live_capture = None

    def _schedule_live_refresh(self) -> None:
        with self.live_lock:
            live_rgb = self.live_display_rgb.copy() if self.live_display_rgb is not None else None
            status_text = self.live_status_text
        if live_rgb is not None:
            photo = _preview_photo(live_rgb, max_size=(1080, 680))
            self.live_image_label.configure(image=photo, text="")
            self.live_image_label.image = photo
        self.live_status_var.set(status_text)
        self._set_text(self.live_text, status_text)
        self.after(80, self._schedule_live_refresh)

    def _analyze_live_snapshot(self) -> None:
        if self.live_snapshot_frame is None:
            messagebox.showinfo("No Snapshot", "Start the live stream first so a frame is available.")
            return
        try:
            path = _save_bgr_frame_to_temp(self.live_snapshot_frame)
            self.temp_paths.append(path)
            self.captured_source_path = path
            self.captured_source_rgb = cv2.cvtColor(self.live_snapshot_frame, cv2.COLOR_BGR2RGB)
            self.image_source_mode.set("camera")
            self.source_path_label.configure(text=f"Captured from live device {self.live_device_index.get()}")
            self._set_preview("source", self.captured_source_rgb)
            self.image_status.set("Current live frame copied into Image Studio.")
        except Exception as exc:
            messagebox.showerror("Snapshot Failed", str(exc))

    def _on_close(self) -> None:
        self._stop_live()
        for path in self.temp_paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        self.destroy()


def main() -> None:
    app = DesktopDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
