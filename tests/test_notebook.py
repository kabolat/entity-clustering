from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_example_notebook_executes_without_errors():
    root = Path(__file__).parents[1]
    notebook_path = root / "notebooks/01_example_workflow.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(notebook_path.parent)}}
    ).execute()
