"""Structure and syntax of notebook.ipynb (no dataset, no training)."""
import ast

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

from conftest import NOTEBOOK_PATH, notebook_code_cells, notebook_constant


def test_notebook_is_valid_nbformat():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(nb)
    assert nb.metadata["kernelspec"]["language"] == "python"
    assert nb.metadata.get("accelerator") == "GPU"  # Colab opens it with a GPU runtime


def test_every_code_cell_compiles():
    transformer = TransformerManager()
    for number, cell in enumerate(notebook_code_cells()):
        python = transformer.transform_cell(cell)  # handles any %magic or !command
        compile(python, f"<notebook code cell {number}>", "exec")


def test_required_sections_present():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    headings = " | ".join(c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown")
    for words in ("Settings", "Download the Fruits 360 dataset", "Inspect", "Select images and split",
                  "Preprocessing", "Build the CNN", "Train the model", "Evaluate", "Predict a single image",
                  "Save the model", "Reload test", "Download the files"):
        assert words in headings, f"Missing section: {words}"


def test_full_training_is_the_default():
    assert notebook_constant("QUICK_TEST") is False
    assert notebook_constant("DATASET_SOURCE") == "github"


def test_colab_only_code_is_guarded():
    """google.colab only exists in Colab; every use must be inside 'if IN_COLAB'."""
    for cell in notebook_code_cells():
        tree = ast.parse(cell)
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "IN_COLAB" in ast.unparse(node.test):
                for child in ast.walk(node):
                    guarded.add(id(child))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "google.colab":
                assert id(node) in guarded, f"Unguarded google.colab import:\n{cell[:200]}"


def test_model_contains_rescaling_and_softmax():
    joined = "\n".join(notebook_code_cells())
    assert 'layers.Rescaling(1.0 / 255' in joined
    assert 'layers.Dense(num_classes, activation="softmax"' in joined
    assert "keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)" in joined


def test_no_leakage_checks_present():
    joined = "\n".join(notebook_code_cells())
    assert "assert all(TRAIN_DIR in p.parents for p in train_paths)" in joined
    assert "assert all(TEST_DIR in p.parents for p in val_paths + test_paths)" in joined
