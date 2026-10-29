"""Optional: run notebook.ipynb end to end on a SMALL local copy of Fruits 360.

This is a pipeline check, not real training. It sets QUICK_TEST = True (the
notebook's own quick mode, max 3 epochs) and nothing else, executes every cell,
and checks the files the notebook writes. The model it produces is a
quick-test model: never submit it.

Usage (from the project folder):
    python tests/smoke_run_notebook.py --data PATH_TO_FOLDER_WITH_Training_AND_Test --workdir smoke_run

The data folder can be a partial Fruits 360 download, as long as all five
classes exist in both Training/ and Test/.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--cell-timeout", type=int, default=3600)
    args = parser.parse_args()

    data = args.data.resolve()
    assert (data / "Training").is_dir() and (data / "Test").is_dir(), "need Training/ and Test/"
    work = args.workdir.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        (work / "fruits-360").symlink_to(data, target_is_directory=True)
    except OSError:  # Windows without symlink rights
        shutil.copytree(data, work / "fruits-360")

    nb = nbformat.read(PROJECT_DIR / "notebook.ipynb", as_version=4)
    settings = next(c for c in nb.cells if c.cell_type == "code" and "QUICK_TEST = False" in c.source)
    settings.source = settings.source.replace("QUICK_TEST = False", "QUICK_TEST = True", 1)

    start = time.time()
    client = NotebookClient(nb, timeout=args.cell_timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(work)}})
    try:
        client.execute()
        error = None
    except Exception as exc:  # noqa: BLE001
        error = exc
    nbformat.write(nb, work / "notebook_executed.ipynb")
    minutes = (time.time() - start) / 60
    if error:
        print(f"FAILED after {minutes:.1f} min: {type(error).__name__}: {str(error)[-1500:]}")
        return 1

    out = work / "vision_outputs"
    info = json.loads((out / "model" / "model_info.json").read_text())
    classes = json.loads((out / "class_names.json").read_text())
    expected_classes = json.loads((PROJECT_DIR / "class_names.json").read_text())

    import app

    checks = {
        "every cell ran without error": True,
        "vision_outputs.zip created": (work / "vision_outputs.zip").is_file(),
        "vision_model.keras created": (out / "model" / "vision_model.keras").is_file(),
        "class_names.json equals project file": classes == expected_classes,
        "model_info class_names equal": info["class_names"] == expected_classes,
        "model_info threshold equals app default": info["confidence_threshold"] == app.DEFAULT_CONFIDENCE_THRESHOLD,
        "model_info input size 100x100x3": info["input_size"] == [100, 100, 3],
        "preprocessing fingerprint equals app.py": info["preprocessing_fingerprint"]
        == app.source_fingerprint(app.preprocess_pil_image),
        "marked as quick test": info["quick_test"] is True,
    }
    code_cells = sum(c.cell_type == "code" for c in nb.cells)
    print(f"Executed {code_cells} code cells in {minutes:.1f} min "
          f"(TF {info['tensorflow_version']}, {info['epochs_trained']} epochs, "
          f"quick-test accuracy {info['test_accuracy']} on {info['test_images']} images)")
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
