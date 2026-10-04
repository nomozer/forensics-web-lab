import json
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/phase_4c2h_development_colab.ipynb"


def test_notebook_is_compact_development_only_launcher() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "RUN_DEVELOPMENT_SMOKE = True" in source
    assert "run_phase_4c2h_development_diagnostics.py" in source
    assert "--data-root" in source
    assert "phase_4c1_binary_n250_reusable.tar" in source
    assert "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27" in source
    assert "411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d" in source
    assert "len(rows) == 341" in source
    assert "locked_test" not in source.lower()
    assert "EXECUTE_OFFICIAL" not in source
