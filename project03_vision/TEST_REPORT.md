# Test report

These checks were run on **23 October 2026** while preparing the project, on a Linux machine with 1 CPU core and about 3 GB of RAM (Python 3.12.3, TensorFlow-CPU 2.21.0, Keras 3.15.1, FastAPI 0.161.1). No GPU was used and **no real training was done**.

## Passed

| Area | What was checked | Result |
|---|---|---|
| Automated tests (`pytest`) | 48 tests in `tests/`: notebook/app consistency, notebook structure and syntax, preprocessing, API with no model, refused models, test-only models, and missing website files | **48 passed, 0 failed** |
| Incomplete project folder (fix of 29 Oct 2026) | Server started with `static/` removed, and with `index.html` removed | Before the fix: unstyled page with CSS/JS 404, or *Internal Server Error*. After: a page naming the missing files, plus an error in the terminal |
| Tests detect mistakes | Four deliberate breakages, each undone afterwards: notebook resize changed to bicubic, notebook threshold changed to 0.70, `class_names.json` order swapped, `app.py` padding changed to black | **All four caught** by failing tests |
| Dataset download patterns | The notebook's sparse-checkout patterns compared with the real folder list of `github.com/fruits-360/fruits-360-100x100` (commit 18fbaa1) | **Exactly the 62 class folders** in both `Training` and `Test`; `Pineapple` excluded |
| Whole notebook | `tests/smoke_run_notebook.py`: all 17 code cells run unchanged except `QUICK_TEST = True`, on about 2,100 real Fruits 360 images | **No errors**, 0.7 min on CPU; outputs, class order, threshold, input size and preprocessing fingerprint all match `app.py` |
| Live server, no model | `uvicorn`, then `/`, `/static/app.js`, `/health`, `/predict` | Page 200; `/health` and `/predict` 503 with the Colab instructions |
| Live server, with model | The quick-test model from the smoke run (temporary folder, deleted, **not** in this project) | Loaded; quick-test warning shown; 20 to 50 ms per prediction; `test_api.py` works |
| Notebook = website | Same 5 images through the notebook's own `load_images()` + `model.predict` and through `POST /predict` | Largest probability difference 4.2e-07 (website rounds to 6 decimals) |
| Browser (Chromium 141) | Model status, threshold text from server, warning, preview, Classify, 5 bars, Unknown sticker, Download Image, Choose Another Image, non-image rejected, phone width 390 px in dark mode, result scrolls into view, camera input | **All passed**, no JavaScript exceptions |
| Windows packages | PyPI lists TensorFlow 2.21.0 wheels for Windows (win_amd64) for Python 3.10, 3.11, 3.12 and 3.13 | Available |

The quick-test accuracy in the smoke run (60%, 382 training images, 3 epochs) only proves the code runs. It says nothing about the real model.

## Failed

Nothing in the final runs. A browser check first reported the server's intended 503 answers (no model loaded) as console errors; the check was corrected to ignore expected HTTP status logs and then passed.

## Not tested

| Item | Why |
|---|---|
| Full training in Google Colab with a GPU, and the final test accuracy | Needs Colab; you run it (README section 2) |
| Downloading the full five classes (about 320 MB) inside Colab | Only the folder list and a small subset were downloaded here |
| `DATASET_SOURCE = "kaggle"` | Needs a Kaggle API key |
| `files.download` and the Google Drive fallback in Colab | Only exist inside Colab |
| Installing and running on Windows, PowerShell activation, VS Code | No Windows machine available |
| A model trained with a different TensorFlow version than the one on your computer | Depends on Colab's version on the day you train |
| A real phone camera, ngrok and cloud deployment (bonus) | Need your accounts and devices (`DEPLOYMENT.md`) |
| Your own photos (busy backgrounds, several fruits) | Needs the real model |

## Run the checks yourself

```powershell
pip install -r requirements-dev.txt
pytest
```
