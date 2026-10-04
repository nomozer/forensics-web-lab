import json
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/phase_4c2h_development_colab.ipynb"


def test_notebook_is_exact_self_staging_pilot_launcher() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 5
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "MODE = 'pilot'" in source
    assert "ALLOW_FULL_150_FITS = False" in source
    assert "ml.training.run_phase_4c2h" in source
    assert "--snapshot-manifest" in source
    assert "--code-snapshot-archive" in source
    assert "--dataset-archive" in source
    assert "phase_4c2h_code_75568d1.tar.gz" in source
    assert "75568d1c03d89e02ad654df75c969924b857ee78" in source
    assert "2bd393873be3291084f86a329a34566bd9aa46c0efd8aa9f155e47571b132c18" in source
    assert "CODE_ARCHIVE_BYTES = 9509956" in source
    assert "phase_4c1_binary_n250_reusable.tar" in source
    assert "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27" in source
    assert "411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d" in source
    assert "frozen_features" not in source
    assert "fit_receipt.json" in source
    assert "safe_extract" in source
    assert "shutil.disk_usage" in source
    assert "locked_test" not in source.lower()
    assert "run_phase_4c2h_development_diagnostics.py" not in source


def test_notebook_code_cells_parse() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
