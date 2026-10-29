"""FastAPI tests.

IMPORTANT: the models built here are tiny TEST-ONLY models with hand-set
weights (no training). They exist only in a temporary folder during the test
and only check the server's plumbing. The real model must come from
notebook.ipynb in Google Colab.
"""
import io
import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app
from conftest import PROJECT_DIR, SAMPLE_DIR

CLASSES = ["Apple", "Avocado", "Banana", "Eggplant", "Tomato"]


def make_test_model(folder, bias, size=100):
    """Constant-output model: softmax(bias), whatever the image."""
    import keras
    from keras import layers

    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(len(bias), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.layers[-1].set_weights([np.zeros((3, len(bias)), "float32"), np.array(bias, "float32")])
    (folder / "model").mkdir(parents=True, exist_ok=True)
    model.save(folder / "model" / "vision_model.keras")


def write_info(folder, **changes):
    import tensorflow as tf

    info = {
        "class_names": CLASSES,
        "input_size": [100, 100, 3],
        "confidence_threshold": 0.60,
        "preprocessing_fingerprint": app.source_fingerprint(app.preprocess_pil_image),
        "quick_test": False,
        "tensorflow_version": tf.__version__,
        "test_accuracy": 0.99,
    }
    info.update(changes)
    (folder / "model").mkdir(parents=True, exist_ok=True)
    (folder / "model" / "model_info.json").write_text(json.dumps(info))
    (folder / "class_names.json").write_text(json.dumps(CLASSES))


@pytest.fixture
def client_for(tmp_path, monkeypatch):
    """client_for(folder) -> TestClient whose server loads from that folder."""
    def make(folder):
        monkeypatch.setattr(app, "MODEL_PATH", folder / "model" / "vision_model.keras")
        monkeypatch.setattr(app, "MODEL_INFO_PATH", folder / "model" / "model_info.json")
        monkeypatch.setattr(app, "CLASS_NAMES_PATH", folder / "class_names.json")
        return TestClient(app.app)
    return make


BANANA_BIAS = [0, 0, 3, 0, 0]  # softmax -> Banana about 0.83


@pytest.fixture(scope="module")
def banana_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("banana")
    make_test_model(folder, BANANA_BIAS)
    write_info(folder)
    return folder


def jpeg_bytes(size=(120, 80), color=(250, 220, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "JPEG")
    return buffer.getvalue()


def post(client, data, name="x.jpg", mime="image/jpeg"):
    return client.post("/predict", files={"file": (name, data, mime)})


# ---------- without a model ----------
def test_without_model_site_opens_and_explains(tmp_path, client_for):
    shutil.copy(PROJECT_DIR / "class_names.json", tmp_path / "class_names.json")
    with client_for(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Classify Image" in page.text
        for asset in ("/static/app.js", "/static/style.css"):
            assert client.get(asset).status_code == 200
        health = client.get("/health")
        assert health.status_code == 503
        body = health.json()
        assert body["model_loaded"] is False
        assert "notebook.ipynb" in body["message"] and "Colab" in body["message"]
        assert body["confidence_threshold"] == 0.60
        response = post(client, jpeg_bytes())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_not_loaded"


def test_missing_model_info_is_refused(tmp_path, client_for):
    make_test_model(tmp_path, BANANA_BIAS)
    shutil.copy(PROJECT_DIR / "class_names.json", tmp_path / "class_names.json")
    with client_for(tmp_path) as client:
        body = client.get("/health").json()
        assert body["model_loaded"] is False and "model_info.json" in body["message"]


@pytest.mark.parametrize("changes, expected", [
    ({"class_names": ["Avocado", "Apple", "Banana", "Eggplant", "Tomato"]}, "class order"),
    ({"preprocessing_fingerprint": "0123456789abcdef"}, "preprocess_pil_image"),
    ({"confidence_threshold": 1.5}, "confidence_threshold"),
])
def test_mismatched_training_settings_are_refused(tmp_path, client_for, changes, expected):
    make_test_model(tmp_path, BANANA_BIAS)
    write_info(tmp_path, **changes)
    with client_for(tmp_path) as client:
        body = client.get("/health").json()
        assert body["model_loaded"] is False and expected in body["message"]


def test_wrong_input_size_is_refused(tmp_path, client_for):
    make_test_model(tmp_path, BANANA_BIAS, size=64)
    write_info(tmp_path)
    with client_for(tmp_path) as client:
        body = client.get("/health").json()
        assert body["model_loaded"] is False and "input" in body["message"]


# ---------- with a (test-only) model ----------
def test_bundled_samples_predict_all_five_classes():
    with TestClient(app.app) as client:
        for class_name in CLASSES:
            image_path = SAMPLE_DIR / f"{class_name.lower()}.jpg"
            response = post(client, image_path.read_bytes(), image_path.name)
            assert response.status_code == 200
            assert response.json()["prediction"] == class_name


def test_featureless_image_is_unknown():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), (128, 128, 128)).save(buffer, "PNG")
    with TestClient(app.app) as client:
        response = post(client, buffer.getvalue(), "blank.png", "image/png")
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "unknown"
        assert data["confidence"] == 0.0
        assert "too little visual detail" in data["message"]


def test_predict_success_schema(banana_dir, client_for):
    with client_for(banana_dir) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["model_loaded"] is True
        assert health.json()["warning"] is None
        response = post(client, (SAMPLE_DIR / "banana.jpg").read_bytes(), "banana.jpg")
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Banana" and data["is_unknown"] is False
        assert [s["class_name"] for s in data["scores"]] == CLASSES  # model output order
        assert abs(sum(s["probability"] for s in data["scores"]) - 1) < 1e-4
        assert data["confidence"] == pytest.approx(np.exp(3) / (np.exp(3) + 4), rel=1e-4)
        assert data["image"] == {"filename": "banana.jpg", "width": 100, "height": 100}


def test_threshold_comes_from_model_info(tmp_path, client_for):
    """Same model, threshold 0.90 in model_info.json -> 0.83 becomes unknown."""
    make_test_model(tmp_path, BANANA_BIAS)
    write_info(tmp_path, confidence_threshold=0.90)
    with client_for(tmp_path) as client:
        assert client.get("/health").json()["confidence_threshold"] == 0.90
        data = post(client, jpeg_bytes()).json()
        assert data["prediction"] == "unknown" and data["is_unknown"] is True
        assert data["top_class"] == "Banana" and data["threshold"] == 0.90


def test_uniform_scores_are_unknown(tmp_path, client_for):
    make_test_model(tmp_path, [0, 0, 0, 0, 0])
    write_info(tmp_path)
    with client_for(tmp_path) as client:
        data = post(client, jpeg_bytes()).json()
        assert data["prediction"] == "unknown" and data["confidence"] == pytest.approx(0.2)


def test_quick_test_model_shows_warning(tmp_path, client_for):
    make_test_model(tmp_path, BANANA_BIAS)
    write_info(tmp_path, quick_test=True)
    with client_for(tmp_path) as client:
        body = client.get("/health").json()
        assert body["model_loaded"] is True and "QUICK_TEST" in body["warning"]


@pytest.mark.parametrize("fmt", ["PNG", "WEBP", "BMP", "GIF", "TIFF"])
def test_other_image_formats(banana_dir, client_for, fmt):
    buffer = io.BytesIO()
    Image.new("RGB", (60, 90), (200, 30, 30)).save(buffer, fmt)
    with client_for(banana_dir) as client:
        assert post(client, buffer.getvalue(), f"x.{fmt.lower()}").status_code == 200


@pytest.mark.parametrize("data, status, code", [
    (b"", 400, "empty_file"),
    (b"this is plain text, not an image", 415, "not_an_image"),
    (b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n", 415, "not_an_image"),
])
def test_bad_uploads(banana_dir, client_for, data, status, code):
    with client_for(banana_dir) as client:
        response = post(client, data)
        assert response.status_code == status and response.json()["error"]["code"] == code


def test_truncated_jpeg(banana_dir, client_for):
    with client_for(banana_dir) as client:
        response = post(client, jpeg_bytes((400, 400))[:600])
        assert response.status_code == 422 and response.json()["error"]["code"] == "corrupted_image"


def test_file_too_large(banana_dir, client_for):
    with client_for(banana_dir) as client:
        response = post(client, b"\xff\xd8" + b"0" * (10 * 1024 * 1024))
        assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"


def test_missing_file_field(banana_dir, client_for):
    with client_for(banana_dir) as client:
        response = client.post("/predict", data={"something": "else"})
        assert response.status_code == 422 and response.json()["error"]["code"] == "missing_file"


# ---------- incomplete project folder ----------
@pytest.mark.parametrize("missing", [["static/style.css", "static/app.js"], ["index.html"]])
def test_missing_website_files_are_explained(monkeypatch, tmp_path, missing):
    paths = {"index.html": tmp_path / "index.html", "static/style.css": tmp_path / "static" / "style.css",
             "static/app.js": tmp_path / "static" / "app.js"}
    (tmp_path / "static").mkdir()
    for name, path in paths.items():
        if name not in missing:
            path.write_text("x")
    monkeypatch.setattr(app, "BASE_DIR", tmp_path)
    monkeypatch.setattr(app, "WEBSITE_FILES", list(paths.values()))
    with TestClient(app.app) as client:
        page = client.get("/")
        assert page.status_code == 500
        assert "The website files are incomplete" in page.text
        for name in missing:
            assert name in page.text
