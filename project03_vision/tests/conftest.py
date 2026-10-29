"""Shared helpers for the tests. Run all tests from the project folder:  pytest"""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

NOTEBOOK_PATH = PROJECT_DIR / "notebook.ipynb"
APP_PATH = PROJECT_DIR / "app.py"
SAMPLE_DIR = PROJECT_DIR / "sample_images"


def notebook_code_cells():
    import nbformat

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def find_function_source(source: str, name: str) -> str | None:
    """Exact text of a top-level function definition, or None."""
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None


def notebook_function_source(name: str) -> str:
    found = [s for s in (find_function_source(c, name) for c in notebook_code_cells()) if s]
    assert len(found) == 1, f"{name}() must be defined exactly once in the notebook, found {len(found)}"
    return found[0]


def notebook_constant(name: str):
    """Value of a top-level 'NAME = literal' assignment in the notebook."""
    values = []
    for cell in notebook_code_cells():
        for node in ast.parse(cell).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                try:
                    values.append(ast.literal_eval(node.value))
                except ValueError:
                    pass
    assert values, f"{name} not found in the notebook"
    return values[0]


@pytest.fixture(scope="session")
def class_names():
    return json.loads((PROJECT_DIR / "class_names.json").read_text(encoding="utf-8"))
