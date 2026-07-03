# Facial Image Warping, Aging, and Expression Transformation

This project provides a professional Python starter architecture for a digital signal processing (DSP) oriented facial image transformation system. It is designed to support expression editing, geometric facial warping, frequency-based aging and de-aging, Fourier analysis, and quantitative evaluation in a modular way.

Sprint 1 now implements the image input and preprocessing foundations using OpenCV, NumPy, Pillow, and Matplotlib.
Sprint 2 now implements classical face detection using OpenCV Haar Cascade.

## System Purpose

The system processes a frontal face image through a structured DSP workflow:

`User Input -> Image Acquisition -> Preprocessing -> DSP/Image Processing -> Output Generation -> Performance Evaluation`

The architecture is intentionally modular so each step can be developed, tested, and replaced independently.

## High-Level Architecture

### Pipeline Stages

1. `input_module.py`
   Handles image path validation and image loading requests.
2. `preprocessing.py`
   Normalizes image size, color space, and grayscale conversion for downstream processing.
3. `face_detection.py`
   Detects and crops the face region from the input image.
4. `landmark_detection.py`
   Extracts facial landmarks and prepares geometric control points.
5. `geometric_warping.py`
   Applies expression manipulation such as smile enhancement, eyebrow raising, lip widening, and face slimming.
6. `aging_filter.py`
   Applies frequency-inspired aging and de-aging filters.
7. `fourier_analysis.py`
   Computes FFT-based magnitude and energy measurements.
8. `evaluation.py`
   Produces quantitative metrics such as MSE, PSNR, and SSIM.
9. `visualization.py`
   Prepares image comparison views, landmark overlays, and structured result summaries.
10. `app.py`
    Serves as the application entry point and orchestrates the end-to-end flow.

## Project Structure

```text
facial-image-warping/
├── app.py
├── pyproject.toml
├── README.md
├── src/
│   └── facial_image_warping/
│       ├── __init__.py
│       ├── aging_filter.py
│       ├── evaluation.py
│       ├── face_detection.py
│       ├── fourier_analysis.py
│       ├── geometric_warping.py
│       ├── input_module.py
│       ├── landmark_detection.py
│       ├── preprocessing.py
│       └── visualization.py
└── tests/
    ├── __init__.py
    └── test_smoke.py
```

## Data Flow

1. The user provides a facial image path or upload.
2. The system validates and loads the image.
3. The image is normalized for analysis.
4. The face region is detected and cropped.
5. Facial landmarks are extracted.
6. One or more transformations are applied:
   - geometric warping
   - aging
   - de-aging
7. The original and transformed images are analyzed in the frequency domain.
8. Evaluation metrics are computed.
9. The results are prepared for display or export.

## Expected Outputs

The completed implementation is expected to produce:

- Original image preview
- Face crop and normalized image
- Landmark overlay visualization
- Expression-transformed image
- Aged and de-aged image variants
- Fourier magnitude spectrum visualizations
- Frequency energy comparison tables
- Evaluation metrics including MSE, PSNR, and SSIM
- Before/after comparison views
- Exportable analysis results

## Development Notes

Sprint 1 includes:

- loading JPG and PNG files
- validating file format and existence
- converting between BGR, RGB, and grayscale
- resizing images to a standard resolution
- normalizing pixel values to the `[0, 1]` range
- generating and saving histograms
- saving processed artifacts into `outputs/`

Sprint 2 includes:

- frontal-face detection with OpenCV Haar Cascade
- grayscale conversion for detection input
- bounding box generation
- ROI cropping for the detected face
- normalized face resizing to a standard output size
- explicit error handling when no face is detected
- saving detection previews into `outputs/faces/`

## Quick Start

1. Create and activate a virtual environment.
2. Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

3. Run the tests:

```bash
pytest
```

4. Use Sprint 1 from Python:

```python
from app import run_preprocessing_pipeline

result = run_preprocessing_pipeline("path/to/face.png")
print(result["processed_image_path"])
print(result["histogram_path"])
```

Processed files are written under `outputs/`.

Use Sprint 2 from Python:

```python
from app import run_face_detection_pipeline

result = run_face_detection_pipeline("path/to/face.png")
print(result["bounding_box"])
print(result["cropped_face_path"])
```

Detected-face previews are written under `outputs/faces/`.

Sprint1 commands
deactivate
Remove-Item .venv -Recurse -Force
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pytest -q

Sprint2 commands
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'CascadeClassifier'))"
python -c "from app import run_face_detection_pipeline; r = run_face_detection_pipeline('ornek_yuz.png'); print(r['bounding_box']); print(r['cropped_face_path'])"
