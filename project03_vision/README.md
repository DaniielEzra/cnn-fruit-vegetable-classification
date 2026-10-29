# VISION: CNN-Based Image Classification System

A website where you upload a photo of a fruit or vegetable, click **Classify Image**, and a Convolutional Neural Network (CNN) tells you what it sees.

* **Classes (5):** Apple, Avocado, Banana, Eggplant, Tomato
* **Dataset:** Fruits 360 (100 x 100 images)
* **Training:** Google Colab, TensorFlow/Keras (`notebook.ipynb`)
* **Website:** FastAPI backend (`app.py`) + HTML, CSS and JavaScript
* **Bonus implemented:** answer **unknown** when the best confidence is below 60%

> **Important:** the trained model is **not** included in this folder until you create it. You must run `notebook.ipynb` in Google Colab, download the result and copy it into the `model/` folder (steps below). Until then the website opens, but it shows **Model not loaded** and explains what is missing.

---

## Contents

1. [Project structure](#1-project-structure)
2. [Part 1: Train the model in Google Colab](#2-part-1-train-the-model-in-google-colab)
3. [Put the trained files in the project](#3-put-the-trained-files-in-the-project)
4. [Part 2: Run the website on Windows](#4-part-2-run-the-website-on-windows)
5. [Test the API (/health and /predict)](#5-test-the-api)
6. [How it works](#6-how-it-works)
7. [Results](#7-results)
8. [Troubleshooting](#8-troubleshooting)
9. [Submit to GitHub](#9-submit-to-github)
10. [Bonus tasks](#10-bonus-tasks)
11. [Automated tests](#11-automated-tests)

---

## 1. Project structure

```
project03_vision/
├── notebook.ipynb        Colab notebook: data, training, evaluation, saving
├── app.py                FastAPI server (loads the model once, serves the website)
├── index.html            The web page
├── static/
│   ├── style.css         Page design
│   └── app.js            Upload, preview, classify, results
├── class_names.json      Class order: ["Apple", "Avocado", "Banana", "Eggplant", "Tomato"]
├── model/
│   ├── vision_model.keras    (created by the notebook, you add it)
│   └── model_info.json       (created by the notebook, you add it)
├── test_api.py           Script that tests /health and /predict on a running server
├── sample_images/        5 Fruits 360 images (one per class) to try the website
├── tests/                Automated tests (pytest) and the notebook smoke-run script
├── requirements.txt      Python packages for the website
├── requirements-dev.txt  Extra packages for the automated tests
├── DEPLOYMENT.md         Bonus: ngrok, phone testing, cloud deployment
├── screenshots/          Bonus: put your phone screenshot here
├── .gitignore
└── README.md
```

---

## 2. Part 1: Train the model in Google Colab

### 2.1 Open the notebook

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with a Google account.
2. Choose **File > Upload notebook** and select `notebook.ipynb` from this folder.

### 2.2 Turn on the GPU

1. **Runtime > Change runtime type**
2. **Hardware accelerator: T4 GPU** (any GPU is fine)
3. **Save**

The notebook also works without a GPU, but training is much slower. The second cell prints whether a GPU was found.

### 2.3 (Optional) Quick test first

In the **Settings** cell, set `QUICK_TEST = True` and run everything (**Runtime > Run all**). It uses 150 images per class and 3 epochs, so it finishes in a few minutes and shows that every step works. **The quick-test accuracy is low on purpose. Do not submit a quick-test model.**

Then set `QUICK_TEST = False` again and continue.

### 2.4 Full training

1. Make sure `QUICK_TEST = False`.
2. **Runtime > Run all**.

The notebook then:

| Step | What happens |
|---|---|
| Download dataset | Downloads only the five needed classes of Fruits 360 from the official GitHub repository (no account needed, about 320 MB) |
| Inspect | Lists the variety folders (for example `Apple Golden 1`, `Banana Red 1`) and checks that all five classes exist |
| Select and split | 700 training images per class from `Training/`; 400 per class from `Test/`, split 50/50 into validation and test |
| Train | CNN with data augmentation, early stopping, up to 25 epochs |
| Evaluate | Accuracy and loss graphs, test accuracy, classification report, confusion matrix |
| Predict | One image with the confidence for all five classes |
| Save | `vision_model.keras`, `class_names.json`, `model_info.json` |
| Reload test | Loads the saved file again and checks it gives identical predictions |
| Download | Creates and downloads `vision_outputs.zip` |

**Downloading the dataset from Kaggle instead (optional).** Set `DATASET_SOURCE = "kaggle"`. On kaggle.com open **Settings > API > Create New Token** to get `kaggle.json`; the notebook will ask you to upload it. Never put `kaggle.json` on GitHub.

### 2.5 Download your results

The last cell downloads **`vision_outputs.zip`** (a few MB). If your browser blocks it:

1. Click the **folder icon** on the left side of Colab, right-click `vision_outputs.zip`, choose **Download**.
2. If that also fails (some school networks block downloads), copy it to your Google Drive. Add a new cell at the end and run:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   !cp vision_outputs.zip /content/drive/MyDrive/
   ```

   Then download `vision_outputs.zip` from [drive.google.com](https://drive.google.com) (**My Drive**).

Check the last cell's output before closing Colab: it must say `Reload test passed` (in the cell above it) and print the TensorFlow version.

Also save the notebook **with outputs**: **File > Download > Download .ipynb**. Replace `notebook.ipynb` in your project with this file, because the version with graphs and accuracy is the one to submit.

> Colab deletes everything when the session ends. Download before closing the tab.

---

## 3. Put the trained files in the project

Unzip `vision_outputs.zip`. Copy the files like this:

| From the zip | To the project |
|---|---|
| `vision_outputs/class_names.json` | `project03_vision/class_names.json` (replace) |
| `vision_outputs/model/vision_model.keras` | `project03_vision/model/vision_model.keras` |
| `vision_outputs/model/model_info.json` | `project03_vision/model/model_info.json` |

**All three files must come from the same zip.** `model_info.json` is required: it holds the class order, the confidence threshold and a fingerprint of the preprocessing code used in training, and the website checks all three when it starts (see section 6).

Open `model/model_info.json` and **write down `tensorflow_version`**. You need it in step 4.3.

---

## 4. Part 2: Run the website on Windows

### 4.1 Install Python

Install **Python 3.11 or 3.12** (64-bit) from [python.org](https://www.python.org/downloads/). In the installer, tick **"Add python.exe to PATH"**.

TensorFlow does not support every Python version. If `pip install` later says *"No matching distribution found for tensorflow"*, your Python version is too new or too old for TensorFlow; check the supported versions at [tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).

### 4.2 Open the project in VS Code and create a virtual environment

1. Open VS Code, choose **File > Open Folder** and select `project03_vision`.
2. Open a terminal: **Terminal > New Terminal**.
3. Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

(Use `py -3.11` if you installed 3.11.) After activation the terminal line starts with `(.venv)`.

If PowerShell says *running scripts is disabled*, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

In VS Code, press **Ctrl+Shift+P > Python: Select Interpreter** and pick the `.venv` interpreter (it shows `('.venv': venv)`). New terminals then activate it automatically.

Useful VS Code extensions (**View > Extensions**): **Python** (Microsoft, required for the interpreter and tests) and **Jupyter** (optional, to open `notebook.ipynb` and look at the outputs; training still happens in Colab).

### 4.3 Install the packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Now make TensorFlow match the version from `model/model_info.json`. For example, if it says `"tensorflow_version": "2.19.0"`:

```powershell
pip install tensorflow==2.19.0
```

Check it:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 4.4 Start the server

From the `project03_vision` folder, with `(.venv)` active:

```powershell
uvicorn app:app --reload
```

(`python app.py` also works. If `uvicorn` is not found, the environment is not active: run `.venv\Scripts\activate`, or use `python -m uvicorn app:app --reload`.)

Wait until you see `Model ready. Input 100x100, classes: [...], threshold 0.60` and `Application startup complete.` The first start can take 10 to 30 seconds while TensorFlow loads.

### 4.5 Open the website

Go to **http://127.0.0.1:8000** in your browser.

1. Click **Choose Image** (or drag an image onto the box, or **Take Photo** on a phone).
2. Check the preview.
3. Click **Classify Image**.
4. Read the prediction and the five confidence bars.
5. **Download Image** saves the selected image. **Choose Another Image** starts again.

The badge at the top right says **Model ready** when everything works. If it says **Model not loaded**, the message under the buttons explains why (usually the model file is missing). A yellow note appears if the model is a quick-test model or was trained with a different TensorFlow version.

To try it without your own photos, use the images in `sample_images/`.

Stop the server with **Ctrl+C** in the terminal.

---

## 5. Test the API

### In the browser

* **http://127.0.0.1:8000/health**: shows the server status as JSON.
* **http://127.0.0.1:8000/docs**: FastAPI's interactive page. Open **POST /predict > Try it out**, choose an image, click **Execute**.

### With the test script

Keep the server running. Open a **second** terminal, activate the environment and run:

```powershell
.venv\Scripts\activate
python test_api.py path\to\banana.jpg
```

### With curl (Windows 10/11)

Use `curl.exe` (not `curl`, which is a different PowerShell command):

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST -F "file=@C:\path\to\banana.jpg" http://127.0.0.1:8000/predict
```

### API reference

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/` | 200, the website | |
| GET | `/health` | 200 `{"status": "ok", "model_loaded": true, "confidence_threshold": 0.6, "warning": null, ...}` | 503 when the model is not loaded (`message` says why) |
| POST | `/predict` | 200 prediction JSON (below) | 400 empty file, 413 file > 10 MB or too many pixels, 415 not an image or unsupported format, 422 damaged image or missing `file` field, 500 prediction failed, 503 model not loaded |

The image must be sent as `multipart/form-data` in a field called **`file`**.

Example `/predict` answer (the numbers here are only an example):

```json
{
  "prediction": "Banana",
  "is_unknown": false,
  "top_class": "Banana",
  "confidence": 0.9731,
  "threshold": 0.6,
  "scores": [
    {"class_name": "Apple",    "probability": 0.0042, "percent": 0.42},
    {"class_name": "Avocado",  "probability": 0.0110, "percent": 1.10},
    {"class_name": "Banana",   "probability": 0.9731, "percent": 97.31},
    {"class_name": "Eggplant", "probability": 0.0021, "percent": 0.21},
    {"class_name": "Tomato",   "probability": 0.0096, "percent": 0.96}
  ],
  "message": "The image looks like banana.",
  "image": {"filename": "banana.jpg", "width": 1200, "height": 900},
  "inference_ms": 31.4
}
```

When the best score is below 60%, `prediction` is `"unknown"`, `is_unknown` is `true`, and `top_class` still shows the best guess.

Error example:

```json
{"error": {"code": "unsupported_format", "message": "'PDF' images are not supported. Use JPEG, PNG, WEBP, BMP, GIF or TIFF."}}
```

---

## 6. How it works

### Same preprocessing in the notebook and the website

The function `preprocess_pil_image()` is **identical** in `notebook.ipynb` and `app.py`:

1. Rotate according to EXIF orientation (photos from phones are often stored sideways).
2. Transparent PNG/GIF/WEBP: place on a **white** background (Fruits 360 has white backgrounds).
3. Convert to RGB.
4. Pad to a square with white so the object is not stretched.
5. Resize to **100 x 100** (bilinear).

The model itself starts with a `Rescaling(1/255)` layer, so normalisation is saved inside `vision_model.keras`. The server sends pixel values 0 to 255, exactly like the notebook did during training. The server also reads the input size from the model and checks that the model has as many outputs as `class_names.json` has classes.

### Same class order

`class_names.json` is written by the notebook from the fixed list `["Apple", "Avocado", "Banana", "Eggplant", "Tomato"]`. Output number 0 of the model is Apple, 1 is Avocado, and so on. The server reads the same file.

### How the website proves it matches the notebook

The notebook writes `model/model_info.json` next to the model. When the server starts it refuses to load the model (and `/health` explains why) if:

| Check | Compared with |
|---|---|
| `class_names` in model_info.json | `class_names.json` |
| `input_size` in model_info.json | the model's real input shape (100 x 100 x 3) |
| number of model outputs | number of classes |
| `preprocessing_fingerprint` (hash of the notebook's `preprocess_pil_image()` code, comments ignored) | hash of the copy in `app.py` |
| `confidence_threshold` | must be a number between 0 and 1 |

The server then uses the **threshold from model_info.json**, so the website always uses exactly the value the notebook used in its evaluation. `DEFAULT_CONFIDENCE_THRESHOLD = 0.60` in `app.py` is only shown on the page while no model is loaded.

To change the threshold, change `CONFIDENCE_THRESHOLD` in the notebook's settings cell **and** `DEFAULT_CONFIDENCE_THRESHOLD` in `app.py` (the tests check that they are equal), then run the notebook again.

### Dataset mapping

Fruits 360 has one folder per variety, for example `Apple Golden 1`, `Tomato Cherry Red 1`, `Eggplant long 1`. A folder belongs to a class when its **first word** is the class name, so all apple varieties become *Apple*, but `Pineapple 1` is not an apple.

### No data leakage

Fruits 360 images were filmed while the fruit rotated, so neighbouring images are almost the same. The notebook therefore trains only on the official `Training` folder and uses the official `Test` folder for validation and test. A check in the notebook proves no file is in two sets.

### CNN architecture

Data augmentation (flip, rotation, zoom, shift, contrast; only active during training) → Rescaling → 4 × [Conv2D → BatchNorm → ReLU → MaxPooling] with 32, 64, 128, 128 filters → GlobalAveragePooling → Dropout 0.4 → Dense 128 → Dropout 0.3 → Dense 5 softmax. Optimiser Adam (learning rate 0.001), loss sparse categorical cross-entropy, early stopping on validation loss.

### Limitations (please read)

* The model knows **only five classes**. Any other object (a cat, a car, an orange) will still get scores for these five classes. It may be marked *unknown*, but it can also get a **confident wrong answer**. The 60% threshold helps but cannot detect every unfamiliar object.
* Fruits 360 photos show **one fruit on a white background**. Real photos with busy backgrounds, several fruits, cut fruit or bad light are harder, so accuracy on your own photos will be lower than the test accuracy. Photos of a single fruit on a plain light background work best.
* iPhone **HEIC** photos are not supported by the server. Most phones send JPEG when you upload through the browser; if not, convert the photo to JPEG.

---

## 7. Results

Fill this in from **your own** Colab run (do not copy numbers from anywhere else):

| Item | Value |
|---|---|
| TensorFlow version (Colab) | |
| Training images / class | 700 |
| Epochs trained (after early stopping) | |
| Test accuracy | |
| Lowest-scoring class (from the report) | |

Add the accuracy/loss graph and confusion matrix screenshots to `screenshots/` if your instructor wants them in the README.

---

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| Badge says **Model not loaded**, message says *Model file not found* | Copy `vision_model.keras` into `project03_vision/model/`. The exact name must be `vision_model.keras`. |
| *The model file exists but could not be loaded* | Your TensorFlow version differs from Colab's. Install the version in `model/model_info.json`: `pip install tensorflow==X.Y.Z`. |
| *The model has N outputs but class_names.json lists M classes* | Use the `class_names.json` from the **same** `vision_outputs.zip` as the model. |
| Browser: *This site can't be reached* / *ERR_CONNECTION_REFUSED* at 127.0.0.1:8000 | The server is not running. Start it (section 4.4) and keep that terminal open while you use the website. The link only works on the computer running the server. |
| *Error loading ASGI app. Could not import module "app"* | The terminal is in the wrong folder. `cd` into the folder that contains `app.py`, then start the server again. |
| Page without design, buttons do nothing, or *The website files are incomplete* | The `static` folder or `index.html` is missing (for example after using a partial ZIP). Extract **project03_vision_full.zip** again and restart the server. |
| *model_info.json not found next to the model* | Copy `model_info.json` from the zip into `model/`, next to `vision_model.keras`. |
| *The model does not match this website: class order ...* | `class_names.json` and `model_info.json` come from different runs. Copy all three files from the same zip. |
| *preprocess_pil_image() in app.py is not the same code ...* | Someone edited the preprocessing in only one place. Make the function identical in `app.py` and the notebook, then retrain. `pytest tests/test_consistency.py` shows the difference. |
| Yellow note: *trained with QUICK_TEST = True* | You copied a quick-test model. Run the notebook again with `QUICK_TEST = False`. |
| `uvicorn` is not recognised | Activate the environment (`.venv\Scripts\activate`) or run `python -m uvicorn app:app --reload`. |
| *No matching distribution found for tensorflow* | Use Python 3.11 or 3.12 (64-bit). Delete `.venv` and create it again with that version. |
| `ImportError: DLL load failed` when importing TensorFlow | Install the latest **Microsoft Visual C++ Redistributable (x64)** and restart VS Code. |
| *Running scripts is disabled on this system* | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| *Address already in use* / port 8000 busy | Close the other server, or run `uvicorn app:app --reload --port 8001` and open http://127.0.0.1:8001 |
| Website opens but has no styling | Start uvicorn **from the project folder**, so the `static/` folder is found. |
| *Cannot reach the server* in the page | The terminal with uvicorn was closed or crashed. Start it again and read the error in the terminal. |
| Every photo becomes *unknown* or the wrong fruit | Use a single fruit on a plain, light background, filling most of the photo. Check that you trained with `QUICK_TEST = False`. |
| Colab: `git` download fails | Run the cell again (temporary network error) or use `DATASET_SOURCE = "kaggle"`. |
| Colab: out of memory | Lower `MAX_TRAIN_PER_CLASS` (for example 400) or `BATCH_SIZE` (16). |
| Colab: no GPU available | Training still works on CPU, just slower. Try again later for a GPU. |

---

## 9. Submit to GitHub

> Follow your instructor's PDF for the exact repository name, visibility (public/private) and how to send the link. The steps below are the general method.

### 9.1 What to upload

* Source code: `app.py`, `index.html`, `static/`, `test_api.py`, `tests/`, `sample_images/`
* `notebook.ipynb` **with outputs** (downloaded from Colab after full training)
* `model/vision_model.keras`, `model/model_info.json`, `class_names.json`
* `README.md`, `requirements.txt`, `requirements-dev.txt`, `DEPLOYMENT.md`, `TEST_REPORT.md`, `.gitignore`
* Optional: `screenshots/` with your phone screenshot (bonus)

Do **not** upload `.venv/`, the dataset, `vision_outputs.zip` or `kaggle.json`. The `.gitignore` already excludes them.

### 9.2 Model size

GitHub refuses files larger than **100 MB** and warns above 50 MB. This CNN is small (a few MB), so it can be uploaded normally. Check the size of `model/vision_model.keras` in File Explorer. If you change the architecture and it grows above 100 MB, either:

* use **Git LFS**: `git lfs install`, `git lfs track "*.keras"`, `git add .gitattributes` before adding the model, or
* upload the model to Google Drive, share it as *Anyone with the link can view*, and put the link in the README with instructions to place it in `model/`.

### 9.3 Upload with the GitHub website (easiest)

1. On github.com click **New repository**, enter a name (for example `project03_vision`), create it **without** a README.
2. Click **uploading an existing file**.
3. Drag all project files and folders **except** `.venv`. (The web uploader may hide `.gitignore`; you can create it on GitHub with **Add file > Create new file**.)
4. Write a commit message and click **Commit changes**.

### 9.4 Upload with Git (from the VS Code terminal)

Install [Git for Windows](https://git-scm.com/download/win) first. Then in the project folder:

```powershell
git init
git add .
git status            # check that .venv and the dataset are NOT listed
git commit -m "VISION project: CNN fruit and vegetable classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/project03_vision.git
git push -u origin main
```

Open your repository page and check that `model/vision_model.keras` and the notebook outputs (graphs) are visible.

---

## 10. Bonus tasks

| Bonus | Status |
|---|---|
| Return *unknown* below 60% confidence | **Implemented** in the notebook, `app.py` and the website. The value is saved in `model_info.json` by the notebook and read by the server (section 6) |
| Expose the local app with ngrok | Instructions in `DEPLOYMENT.md` (you must set it up yourself) |
| Test from a phone and take a screenshot | Instructions in `DEPLOYMENT.md`; save the screenshot in `screenshots/` |
| Deploy to a public cloud service | Optional instructions in `DEPLOYMENT.md` (not set up) |

---

## 11. Automated tests

These tests run on your computer in a few seconds. They do **not** need the dataset or a trained model.

```powershell
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
```

In VS Code you can also use the **Testing** panel (flask icon on the left): **Configure Python Tests > pytest > tests**.

| File | What it checks |
|---|---|
| `tests/test_consistency.py` | The notebook and `app.py` have the identical `preprocess_pil_image()` and fingerprint function, the same class order, image size (100) and threshold (0.60); the upload limit in `app.py` and `static/app.js` match |
| `tests/test_notebook.py` | The notebook is valid, every code cell compiles, all sections exist, full training is the default, Colab-only code only runs in Colab |
| `tests/test_preprocessing.py` | Fruits 360 images pass through unchanged; padding, transparency, grayscale and phone rotation (EXIF) |
| `tests/test_app.py` | The website and API with no model, with refused models (wrong class order, wrong preprocessing, wrong input size, missing model_info.json) and with tiny **test-only** models: success answer, *unknown*, threshold read from model_info.json, every error code |

The models in `tests/test_app.py` are built inside the test with fixed weights and deleted afterwards. They only test the server, they are not trained and are never saved in `model/`.

**Optional notebook smoke run.** `tests/smoke_run_notebook.py` runs the whole notebook with `QUICK_TEST = True` on a small local copy of Fruits 360 and checks the files it writes. It needs `requirements-dev.txt` and a folder with `Training/` and `Test/`:

```powershell
python tests/smoke_run_notebook.py --data C:\path\to\fruits-360 --workdir smoke_run
```

See `TEST_REPORT.md` for the results of the checks run when this project was prepared.
