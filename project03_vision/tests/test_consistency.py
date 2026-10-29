"""The notebook and the website must use the SAME preprocessing, image size,
class order and confidence threshold. These tests compare the actual code."""
import re

import app
from conftest import (APP_PATH, PROJECT_DIR, find_function_source, notebook_code_cells,
                      notebook_constant, notebook_function_source)

APP_SOURCE = APP_PATH.read_text(encoding="utf-8")


def test_preprocessing_function_is_identical():
    assert notebook_function_source("preprocess_pil_image") == find_function_source(
        APP_SOURCE, "preprocess_pil_image"
    )


def test_fingerprint_function_is_identical():
    assert notebook_function_source("source_fingerprint") == find_function_source(
        APP_SOURCE, "source_fingerprint"
    )


def test_class_order_matches_class_names_json(class_names):
    assert notebook_constant("CLASS_NAMES") == class_names
    assert class_names == ["Apple", "Avocado", "Banana", "Eggplant", "Tomato"]


def test_confidence_threshold_matches():
    assert notebook_constant("CONFIDENCE_THRESHOLD") == app.DEFAULT_CONFIDENCE_THRESHOLD == 0.60


def test_image_size_matches():
    assert notebook_constant("IMAGE_SIZE") == 100
    # The app reads the size from the model; its fallback default must agree.
    assert app.ModelState.image_size == 100
    assert "def preprocess_pil_image(image: Image.Image, image_size: int = 100)" in APP_SOURCE


def test_notebook_records_what_the_app_checks():
    joined = "\n".join(notebook_code_cells())
    for key in ('"class_names": CLASS_NAMES', '"confidence_threshold": CONFIDENCE_THRESHOLD',
                '"preprocessing_fingerprint": PREPROCESSING_FINGERPRINT',
                '"input_size": [IMAGE_SIZE, IMAGE_SIZE, 3]', '"quick_test": QUICK_TEST',
                '"tensorflow_version": tf.__version__'):
        assert key in joined, f"model_info.json must contain {key}"
    for name in ('"vision_model.keras"', '"model_info.json"', '"class_names.json"', '"vision_outputs.zip"'):
        assert name in joined


def test_fingerprint_matches_between_copies():
    """Run the notebook's copy of the code and compare fingerprints."""
    namespace = {}
    header = "import hashlib, inspect, io, tokenize\nimport numpy as np\nfrom PIL import Image, ImageOps\n"
    code = header + notebook_function_source("preprocess_pil_image") + "\n\n" + notebook_function_source("source_fingerprint")
    # Write to a real file so inspect.getsource works, like in a notebook.
    path = PROJECT_DIR / "tests" / "_nb_copy_tmp.py"
    path.write_text(code, encoding="utf-8")
    try:
        exec(compile(code, str(path), "exec"), namespace)
        nb_fp = namespace["source_fingerprint"](namespace["preprocess_pil_image"])
    finally:
        path.unlink()
    assert nb_fp == app.source_fingerprint(app.preprocess_pil_image)
    assert re.fullmatch(r"[0-9a-f]{16}", nb_fp)


def test_upload_limit_matches_javascript():
    js = (PROJECT_DIR / "static" / "app.js").read_text(encoding="utf-8")
    assert "const MAX_BYTES = 10 * 1024 * 1024;" in js
    assert app.MAX_UPLOAD_BYTES == 10 * 1024 * 1024


def test_no_threshold_override_in_app():
    assert "VISION_CONFIDENCE_THRESHOLD" not in APP_SOURCE
