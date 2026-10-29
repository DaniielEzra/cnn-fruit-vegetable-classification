"""
VISION - CNN-Based Image Classification System
FastAPI backend for the Fruit & Vegetable Classifier.

What this file does:
  * Loads the trained Keras model ONE time, when the server starts.
  * GET  /         -> serves the website (index.html)
  * GET  /health   -> reports whether the server and model are ready
  * POST /predict  -> receives an image, preprocesses it exactly like the
                      notebook did, runs the CNN and returns the scores

Run it from the project folder with:
    uvicorn app:app --reload        (or: python app.py)
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import logging
import os
import threading
import tokenize
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# All paths are relative to THIS file, so the project works on any computer.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("VISION_MODEL_PATH", BASE_DIR / "vision_outputs" / "model" / "vision_model.keras"))
MODEL_INFO_PATH = MODEL_PATH.parent / "model_info.json"
CLASS_NAMES_PATH = Path(os.getenv("VISION_CLASS_NAMES_PATH", BASE_DIR / "class_names.json"))
INDEX_HTML_PATH = BASE_DIR / "index.html"
STATIC_DIR = BASE_DIR / "static"
# Files the web page needs. If one is missing the page cannot work.
WEBSITE_FILES = [INDEX_HTML_PATH, STATIC_DIR / "style.css", STATIC_DIR / "app.js"]

# If the highest probability is below this value, the answer is "unknown".
# The value actually used is read from model/model_info.json, which the
# notebook writes, so the website always uses the notebook's threshold.
# This default only shows on the page while no model is loaded, and it must
# equal CONFIDENCE_THRESHOLD in notebook.ipynb (tests/test_consistency.py checks it).
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
MIN_IMAGE_PIXEL_STD = 8.0

# Upload limits (checked on the server, not only in the browser).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP", "GIF", "TIFF"}  # detected by Pillow

# Refuse absurdly large images ("decompression bombs") before they use all the RAM.
Image.MAX_IMAGE_PIXELS = 50_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("vision")


# ---------------------------------------------------------------------------
# Preprocessing
# IMPORTANT: this function is copied EXACTLY into notebook.ipynb.
# The model was trained on images prepared by this code, so the website must
# prepare uploaded images in the same way. If you change one, change both.
# ---------------------------------------------------------------------------
def preprocess_pil_image(image: Image.Image, image_size: int = 100) -> np.ndarray:
    """Turn any PIL image into a (image_size, image_size, 3) uint8 RGB array.

    Steps:
      1. Rotate the image according to its EXIF orientation (phone photos).
      2. Put transparent images (PNG with alpha) on a white background.
      3. Convert to RGB (3 colour channels).
      4. Pad to a square with white, keeping the whole object visible.
         (Fruits 360 images are square with a white background.)
      5. Resize to image_size x image_size with bilinear interpolation.

    Pixel values stay in the range 0-255. The model divides by 255 itself
    (it has a Rescaling layer), so normalisation can never be forgotten.
    """
    image = ImageOps.exif_transpose(image)

    has_alpha = image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)

    image = image.convert("RGB")

    width, height = image.size
    if width != height:
        side = max(width, height)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - width) // 2, (side - height) // 2))
        image = square

    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    return np.asarray(image, dtype=np.uint8)


def source_fingerprint(func) -> str | None:
    """Short hash of a function's code, ignoring comments and blank lines.

    The notebook stores the fingerprint of its preprocess_pil_image() in
    model_info.json. The server compares it with its own copy, so a model
    trained with different preprocessing is refused instead of silently
    giving wrong answers. (Also copied exactly into notebook.ipynb.)
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    lines = source.splitlines(keepends=True)
    indent = len(lines[0]) - len(lines[0].lstrip())
    source = "".join(line[indent:] if line.strip() else "\n" for line in lines)
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
    parts = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in skip:
            continue
        parts.append(tokenize.tok_name[tok.type] if tok.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE) else tok.string)
    return hashlib.sha256(" ".join(parts).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Model state (filled once at startup)
# ---------------------------------------------------------------------------
class ModelState:
    model = None
    class_names: list[str] = []
    image_size: int = 100
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    load_error: str | None = None
    warning: str | None = None
    model_info: dict = {}
    lock = threading.Lock()  # one prediction at a time keeps things simple and safe


state = ModelState()


def load_class_names() -> list[str]:
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("class_names.json must contain a JSON list of strings.")
    return names


def model_folder_hint() -> str:
    return "Train the model in notebook.ipynb (Google Colab) and copy vision_model.keras and model_info.json into the model/ folder (README section 3)."


def prediction_debug_enabled() -> bool:
    return os.getenv("VISION_DEBUG_PREDICTIONS", "").strip().lower() in {"1", "true", "yes", "on"}


def check_model_info(info: dict, class_names: list[str]) -> list[str]:
    """Compare model_info.json (written by the notebook) with this server."""
    problems = []
    if info.get("class_names") != class_names:
        problems.append(
            f"class order in model_info.json {info.get('class_names')} differs from class_names.json {class_names}"
        )
    threshold = info.get("confidence_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        problems.append(f"confidence_threshold in model_info.json is invalid: {threshold!r}")
    trained_fp = info.get("preprocessing_fingerprint")
    server_fp = source_fingerprint(preprocess_pil_image)
    if trained_fp and server_fp and trained_fp != server_fp:
        problems.append(
            "preprocess_pil_image() in app.py is not the same code that prepared the training images "
            f"(fingerprint {server_fp} here, {trained_fp} in the notebook)"
        )
    return problems


def load_model_once() -> None:
    """Load and verify the model, class names and model_info.json.
    Errors are stored, not raised, so the website still opens and /health
    can explain the problem."""
    state.model, state.load_error, state.warning, state.model_info = None, None, None, {}
    state.threshold = DEFAULT_CONFIDENCE_THRESHOLD

    try:
        state.class_names = load_class_names()
    except Exception as exc:  # noqa: BLE001
        state.load_error = f"Could not read {CLASS_NAMES_PATH.name}: {exc}"
        logger.error(state.load_error)
        return

    shown_path = MODEL_PATH.relative_to(BASE_DIR) if MODEL_PATH.is_relative_to(BASE_DIR) else MODEL_PATH
    if not MODEL_PATH.exists():
        state.load_error = f"Model file not found at '{shown_path}'. " + model_folder_hint()
        logger.error(state.load_error)
        return
    if not MODEL_INFO_PATH.exists():
        state.load_error = (
            f"model_info.json not found next to the model. It holds the class order and confidence "
            f"threshold used in training. " + model_folder_hint()
        )
        logger.error(state.load_error)
        return

    try:
        info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        state.load_error = f"model_info.json could not be read: {exc}"
        logger.error(state.load_error)
        return
    problems = check_model_info(info, state.class_names)
    if problems:
        state.load_error = (
            "The model does not match this website: " + "; ".join(problems)
            + ". Copy class_names.json, vision_model.keras and model_info.json from the same vision_outputs.zip."
        )
        logger.error(state.load_error)
        return

    try:
        import keras  # imported here so the website can start even if TensorFlow is slow to import

        if prediction_debug_enabled():
            digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            logger.info("Prediction debug: loading model path=%s sha256=%s", MODEL_PATH.resolve(), digest)
        logger.info("Loading model from %s ...", MODEL_PATH)
        model = keras.models.load_model(MODEL_PATH, compile=False)

        # Read the input size from the model itself so it always matches training.
        _, height, width, channels = model.input_shape
        if height != width or channels != 3:
            raise ValueError(f"Unexpected model input shape {model.input_shape}.")
        if list(info.get("input_size", [height, width, 3])) != [height, width, 3]:
            raise ValueError(
                f"The model input is {height}x{width}x3 but model_info.json says {info.get('input_size')}."
            )
        n_outputs = model.output_shape[-1]
        if n_outputs != len(state.class_names):
            raise ValueError(
                f"The model has {n_outputs} outputs but class_names.json lists "
                f"{len(state.class_names)} classes. Use the class_names.json created by the same notebook run."
            )

        # Warm-up prediction, so the first real request is fast.
        model(np.full((1, height, width, 3), 255, dtype=np.float32), training=False)
    except Exception as exc:  # noqa: BLE001
        trained = info.get("tensorflow_version", "unknown")
        state.load_error = (
            f"The model file exists but could not be loaded: {exc} "
            f"(trained with TensorFlow {trained}; see README troubleshooting)"
        )
        logger.exception("Model loading failed")
        return

    state.model = model
    state.image_size = int(height)
    state.model_info = info
    state.threshold = float(info["confidence_threshold"])
    logger.info(
        "Model ready. Input %sx%s, classes: %s, threshold %.2f",
        height, width, state.class_names, state.threshold,
    )

    warnings_found = []
    if info.get("quick_test"):
        warnings_found.append(
            "This model was trained with QUICK_TEST = True and is only a pipeline check. "
            "Retrain with QUICK_TEST = False before submitting."
        )
    if not info.get("preprocessing_fingerprint"):
        warnings_found.append("model_info.json has no preprocessing fingerprint, so preprocessing could not be verified.")
    try:
        import tensorflow as tf

        trained = str(info.get("tensorflow_version", ""))
        if trained and trained.split(".")[:2] != tf.__version__.split(".")[:2]:
            warnings_found.append(
                f"Model trained with TensorFlow {trained}, this computer has {tf.__version__}. "
                "If predictions fail, install the training version (README section 4.3)."
            )
    except Exception:  # noqa: BLE001
        pass
    state.warning = " ".join(warnings_found) or None
    if state.warning:
        logger.warning(state.warning)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model_once()  # runs once when the server starts
    yield


app = FastAPI(
    title="VISION - Fruit & Vegetable Classifier",
    description="CNN image classifier for Apple, Avocado, Banana, Eggplant and Tomato.",
    version="1.0.0",
    lifespan=lifespan,
)

def missing_website_files() -> list[str]:
    """Website files that do not exist (for example after copying only part of the project)."""
    return [str(path.relative_to(BASE_DIR)).replace("\\", "/") for path in WEBSITE_FILES if not path.is_file()]


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if missing_website_files():
    logger.error(
        "Website files missing in %s: %s. The page will not work. Copy them from the complete "
        "project ZIP (project03_vision_full.zip).", BASE_DIR, ", ".join(missing_website_files()),
    )


# ---------------------------------------------------------------------------
# Clear JSON errors:  {"error": {"code": "...", "message": "..."}}
# ---------------------------------------------------------------------------
def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "missing_file",
                "message": "Send the image as multipart/form-data in a field named 'file'.",
            }
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    missing = missing_website_files()
    if missing:
        # Explain the problem in the browser instead of an unstyled page or "Internal Server Error".
        items = "".join(f"<li><code>{name}</code></li>" for name in missing)
        return HTMLResponse(
            status_code=500,
            content=(
                "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
                "<title>VISION - files missing</title></head>"
                "<body style='font-family: system-ui, sans-serif; max-width: 42rem; margin: 3rem auto; padding: 0 1rem; line-height: 1.5'>"
                "<h1>The website files are incomplete</h1>"
                f"<p>The server is running, but these files are missing from <code>{BASE_DIR}</code>:</p>"
                f"<ul>{items}</ul>"
                "<p>Extract <strong>project03_vision_full.zip</strong> again (it contains the "
                "<code>static</code> folder), then restart the server.</p>"
                "</body></html>"
            ),
        )
    return FileResponse(INDEX_HTML_PATH)


@app.get("/health")
def health():
    """Ready = the model is loaded and predictions will work."""
    ready = state.model is not None
    body = {
        "status": "ok" if ready else "model_not_loaded",
        "model_loaded": ready,
        "classes": state.class_names,
        "input_size": [state.image_size, state.image_size] if ready else None,
        "confidence_threshold": state.threshold,
        "test_accuracy": state.model_info.get("test_accuracy") if ready else None,
        "message": "Model is ready." if ready else state.load_error,
        "warning": state.warning,
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


def read_upload_limited(upload: UploadFile) -> bytes:
    """Read the upload but stop as soon as it is bigger than the limit."""
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise api_error(413, "file_too_large", f"The image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise api_error(400, "empty_file", "The uploaded file is empty.")
    return data


def open_image(data: bytes) -> Image.Image:
    """Open the bytes with Pillow and check that it is a supported image."""
    try:
        image = Image.open(io.BytesIO(data))
        image_format = (image.format or "").upper()
        if image_format not in ALLOWED_FORMATS:
            raise api_error(
                415,
                "unsupported_format",
                f"'{image_format or 'unknown'}' images are not supported. Use JPEG, PNG, WEBP, BMP, GIF or TIFF.",
            )
        image.load()  # actually decode the pixels (detects broken files)
        return image
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise api_error(413, "image_too_large", "The image has too many pixels. Use a smaller image.")
    except UnidentifiedImageError:
        raise api_error(
            415,
            "not_an_image",
            "The file is not a readable image. Use JPEG, PNG, WEBP, BMP, GIF or TIFF "
            "(iPhone HEIC photos must be converted to JPEG first).",
        )
    except Exception:  # noqa: BLE001  (truncated or corrupted files)
        raise api_error(422, "corrupted_image", "The image file is damaged and could not be read.")


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Classify one uploaded image.

    This is a normal (non-async) function, so FastAPI runs it in a worker
    thread and the slow model call does not block the server.
    """
    if state.model is None:
        raise api_error(503, "model_not_loaded", state.load_error or "The model is not loaded yet.")

    data = read_upload_limited(file)
    image = open_image(data)

    try:
        if prediction_debug_enabled():
            original_size = image.size
            original_mode = image.mode
            exif_orientation = image.getexif().get(274, 1)
            oriented_image = ImageOps.exif_transpose(image)
            oriented_size = oriented_image.size
            square_side = max(oriented_size)
            pad_left = (square_side - oriented_size[0]) // 2
            pad_top = (square_side - oriented_size[1]) // 2
            has_alpha = original_mode in ("RGBA", "LA", "PA") or (
                original_mode == "P" and "transparency" in image.info
            )
            logger.info(
                "Prediction debug: upload size=%sx%s mode=%s format=%s exif_orientation=%s; "
                "EXIF-oriented size=%sx%s; alpha_composited_on_white=%s; converted_to_RGB=True; "
                "square_pad=%s at left=%s top=%s; resized_to=%sx%s with bilinear=%s",
                original_size[0], original_size[1], original_mode, image.format,
                exif_orientation, oriented_size[0], oriented_size[1], has_alpha,
                square_side if oriented_size[0] != oriented_size[1] else "not-needed",
                pad_left, pad_top, state.image_size, state.image_size,
                oriented_size != (state.image_size, state.image_size),
            )
        pixels = preprocess_pil_image(image, state.image_size)
        if prediction_debug_enabled():
            logger.info(
                "Prediction debug: model_input shape=%s dtype=%s pixel_min=%s pixel_max=%s "
                "pixel_mean=%.3f pixel_std=%.3f; model_rescaling_layer_handles_0_to_1=True",
                pixels.shape, pixels.dtype, int(pixels.min()), int(pixels.max()),
                float(pixels.mean()), float(pixels.std()),
            )
        if float(np.std(pixels)) < MIN_IMAGE_PIXEL_STD:
            if prediction_debug_enabled():
                logger.info("Prediction debug: featureless input rejected before model inference")
            return {
                "prediction": "unknown",
                "is_unknown": True,
                "top_class": "unknown",
                "confidence": 0.0,
                "threshold": state.threshold,
                "scores": [
                    {"class_name": name, "probability": 0.0, "percent": 0.0}
                    for name in state.class_names
                ],
                "message": "There is too little visual detail to classify this image.",
                "image": {"filename": file.filename, "width": image.width, "height": image.height},
                "inference_ms": 0.0,
            }
        batch = pixels.astype(np.float32)[np.newaxis, ...]  # shape (1, H, W, 3), values 0-255
        start = time.perf_counter()
        with state.lock:
            probabilities = state.model(batch, training=False).numpy()[0]
        inference_ms = (time.perf_counter() - start) * 1000
    except Exception:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise api_error(500, "prediction_failed", "The model could not process this image. Try another image.")

    probabilities = probabilities.astype(float)
    if prediction_debug_enabled():
        logger.info(
            "Prediction debug: probabilities=%s",
            json.dumps({name: round(float(score), 8) for name, score in zip(state.class_names, probabilities)}),
        )
    best_index = int(np.argmax(probabilities))
    best_class = state.class_names[best_index]
    best_probability = float(probabilities[best_index])
    threshold = state.threshold
    is_unknown = best_probability < threshold

    return {
        "prediction": "unknown" if is_unknown else best_class,
        "is_unknown": is_unknown,
        "top_class": best_class,
        "confidence": round(best_probability, 6),
        "threshold": threshold,
        # Same order as class_names.json (and the notebook).
        "scores": [
            {"class_name": name, "probability": round(float(p), 6), "percent": round(float(p) * 100, 2)}
            for name, p in zip(state.class_names, probabilities)
        ],
        "message": (
            f"Not confident enough: the best guess ({best_class}) is below {threshold:.0%}."
            if is_unknown
            else f"The image looks like {best_class.lower()}."
        ),
        "image": {"filename": file.filename, "width": image.width, "height": image.height},
        "inference_ms": round(inference_ms, 1),
    }


if __name__ == "__main__":
    # Lets Windows users start the server with:  python app.py
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
