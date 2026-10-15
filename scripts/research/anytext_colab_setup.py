#!/usr/bin/env python3
"""AnyText Colab launcher backend: environment setup, smoke test, concise run, result lookup.

Subcommands (all driven by the short notebook):
  env     system Python: clone AnyText at the plan's locked commit, stage adapter modules,
          create the isolated Python 3.10 venv, install pinned requirements, `pip check`, then `smoke`.
  smoke   venv Python (cwd = AnyText): pinned versions, full inference import graph, glyph generation,
          CUDA fp16 xformers attention. Proves imports/attention dtype path only, NOT that inference works.
  run     spawn run_anytext_colab.py under the venv with a concise console and a full log file.
  result  print the current run's status and archive path (from latest_run.json, never a glob).

Single sources of truth: scripts/research/anytext_requirements.txt (dependencies),
research/evidence/document_pilot/anytext_feasibility_plan.json (commit, revision, attempts),
scripts/research/run_anytext_colab.py (runner).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
REQUIREMENTS = HERE / "anytext_requirements.txt"
PLAN = REPO_ROOT / "research" / "evidence" / "document_pilot" / "anytext_feasibility_plan.json"
STAGED_MODULES = ("anytext_document_adapter.py", "document_feasibility_geometry.py")
ANYTEXT_DIR = Path("/content/AnyText")
ENV_DIR = Path("/content/anytext_env")
TOOLCHAIN = ["pip==23.3.2", "setuptools==68.2.2", "wheel==0.42.0"]
TORCH_PINS = ["torch==2.0.1", "torchvision==0.15.2"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"


def read_pins(path: Path = REQUIREMENTS) -> Dict[str, str]:
    """Parses `name==version` pins (comments and blank lines ignored)."""
    pins: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def _run(cmd: List[str], **kw: Any) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd[:6]), "..." if len(cmd) > 6 else "")
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def _ensure_python310() -> None:
    py = ENV_DIR / "bin" / "python"
    if py.is_file():
        return
    sys_py = shutil.which("python3.10")
    if not sys_py:
        print("[INFO] python3.10 not found; trying deadsnakes PPA")
        subprocess.run(["apt-get", "install", "-y", "-qq", "software-properties-common"], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(["add-apt-repository", "-y", "ppa:deadsnakes/ppa"], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(["apt-get", "update", "-qq"], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(["apt-get", "install", "-y", "-qq", "python3.10", "python3.10-venv", "python3.10-dev"], check=False)
        sys_py = shutil.which("python3.10")
    if sys_py:
        _run([sys_py, "-m", "venv", ENV_DIR])
    else:
        print("[INFO] bootstrapping Python 3.10 with uv")
        _run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        _run(["uv", "venv", ENV_DIR, "--python", "3.10", "--seed"])
    version = subprocess.check_output([str(py), "--version"]).decode().strip()
    assert "3.10" in version, f"Expected Python 3.10 in {ENV_DIR}, got {version}"


def _clone_anytext(commit: str) -> None:
    if not (ANYTEXT_DIR / ".git").is_dir():
        ANYTEXT_DIR.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q"], cwd=ANYTEXT_DIR)
        _run(["git", "remote", "add", "origin", "https://github.com/tyxsspa/AnyText.git"], cwd=ANYTEXT_DIR)
    _run(["git", "fetch", "-q", "--depth", "1", "origin", commit], cwd=ANYTEXT_DIR)
    _run(["git", "checkout", "-q", "-f", commit], cwd=ANYTEXT_DIR)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ANYTEXT_DIR).decode().strip()
    assert head == commit, f"AnyText commit mismatch: expected {commit}, got {head}"
    print(f"[PASS] AnyText source at locked commit {head}")


def _stage_modules(repo_root: Path) -> None:
    dest = ANYTEXT_DIR / "ml" / "evaluation"  # namespace package on purpose: the repo's ml/__init__ imports sklearn
    dest.mkdir(parents=True, exist_ok=True)
    for name in STAGED_MODULES:
        shutil.copy2(repo_root / "ml" / "evaluation" / name, dest / name)
    print(f"[PASS] Staged {', '.join(STAGED_MODULES)} into {dest}")


def cmd_env(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    plan = json.loads((repo_root / PLAN.relative_to(REPO_ROOT)).read_text(encoding="utf-8"))
    commit = plan["model_specification"]["code_commit_sha"]
    subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-dejavu-core"], check=False, stdout=subprocess.DEVNULL)
    _clone_anytext(commit)
    _stage_modules(repo_root)
    _ensure_python310()
    pip = ENV_DIR / "bin" / "pip"
    py = ENV_DIR / "bin" / "python"
    _run([pip, "install", "-q", *TOOLCHAIN])
    _run([pip, "install", "-q", *TORCH_PINS, "--index-url", TORCH_INDEX])
    _run([pip, "install", "-q", "-r", repo_root / "scripts" / "research" / "anytext_requirements.txt"])
    check = subprocess.run([str(pip), "check"], capture_output=True, text=True)
    print(check.stdout.strip())
    assert check.returncode == 0, f"pip check failed:\n{check.stdout}\n{check.stderr}"
    print("[PASS] pip check: no broken requirements")
    env = {**os.environ, "PYTHONPATH": str(ANYTEXT_DIR)}
    smoke = subprocess.run(
        [str(py), str(repo_root / "scripts" / "research" / "anytext_colab_setup.py"), "smoke"],
        cwd=ANYTEXT_DIR, env=env,
    )
    return smoke.returncode


def cmd_smoke(args: argparse.Namespace) -> int:
    from importlib import metadata

    assert sys.version_info[:2] == (3, 10), f"Python 3.10 required, got {sys.version}"
    pins = read_pins()
    pins.update({"torch": "2.0.1", "torchvision": "0.15.2"})
    for name, want in sorted(pins.items()):
        try:
            got = metadata.version(name).split("+")[0]
        except metadata.PackageNotFoundError:
            raise AssertionError(f"{name}=={want} is pinned in anytext_requirements.txt but not installed") from None
        assert got == want, f"{name}: pinned {want}, installed {got}"
    print(f"[PASS] {len(pins)} pinned packages match; python {sys.version.split()[0]}")

    import torch

    assert torch.cuda.is_available(), "CUDA not available: enable a GPU runtime"
    print(f"[PASS] CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}")

    import cv2, einops, omegaconf, transformers, open_clip, addict, yapf, timm, safetensors, easydict  # noqa: F401
    import skimage, six, tqdm, ujson, sentencepiece, diffusers  # noqa: F401
    from huggingface_hub import cached_download  # noqa: F401  (diffusers 0.10.2 needs it)
    from modelscope.pipelines import pipeline  # noqa: F401
    import cldm.model, cldm.cldm, cldm.ddim_hacked, cldm.recognizer  # noqa: F401
    import t3_dataset, util, bert_tokenizer, lora_util  # noqa: F401
    from ml.evaluation import anytext_document_adapter  # noqa: F401
    print("[PASS] inference import graph (incl. lora_util->diffusers, hub, sentencepiece, ujson) imports cleanly")

    import numpy as np
    from PIL import ImageFont

    font = ImageFont.truetype(args.font, size=60)
    for text, poly in (("350,000", [[100, 200], [300, 200], [300, 280], [100, 280]]),
                       ("Magnum Gold Almond", [[50, 150], [450, 150], [450, 250], [50, 250]])):
        g = t3_dataset.draw_glyph(font, text)
        gp = t3_dataset.draw_glyph2(font, text, np.array(poly), width=512, height=512)
        assert g.shape == (80, 512, 1) and g.sum() > 0, f"glyph failed for {text!r}: {g.shape}"
        assert gp.shape == (512, 512, 1) and gp.sum() > 0, f"position guidance failed for {text!r}: {gp.shape}"
    print("[PASS] glyph + position guidance generated for both targets")

    import ldm.modules.attention as attn

    assert attn.XFORMERS_IS_AVAILBLE, "xformers not importable: AnyText would use the fp32-upcast CrossAttention"
    block = attn.BasicTransformerBlock(320, 8, 40, context_dim=768).cuda().half().eval()
    assert isinstance(block.attn1, attn.MemoryEfficientCrossAttention), type(block.attn1).__name__
    with torch.no_grad():
        out = block(torch.randn(2, 64, 320, device="cuda", dtype=torch.half),
                    context=torch.randn(2, 77, 768, device="cuda", dtype=torch.half))
    assert out.dtype == torch.half, out.dtype
    print("[PASS] fp16 BasicTransformerBlock on CUDA uses MemoryEfficientCrossAttention (dtype path only)")
    print("[NOTE] Smoke test does not prove full AnyText inference; that happens in the run step.")
    return 0


# ---- concise console -------------------------------------------------------------------------------------------

_KEEP = re.compile(r"^\s*(\[(PASS|FAIL|OK|INFO|WARN|WARNING|DONE|RUNNER FAILURE|SKIP)|---|===|Traceback|\w*Error|Prompt:)")
_DDIM = re.compile(r"DDIM Sampler:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")


def filter_console_line(line: str) -> Optional[str]:
    """Returns the line to show on the console, or None to keep it in the full log only."""
    line = line.rstrip()
    m = _DDIM.search(line)
    if m:
        step, total = int(m.group(1)), int(m.group(2))
        return f"    DDIM step {step}/{total}" if step % 5 == 0 or step == total else None
    if _KEEP.match(line):
        return line
    return None


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)
    from datetime import datetime

    log_path = log_dir / f"runner_{args.label}_{datetime.now():%Y%m%d_%H%M%S}.log"
    repo_root = Path(args.repo_root).resolve()
    cmd = [
        str(ENV_DIR / "bin" / "python"), "-u", str(repo_root / "scripts" / "research" / "run_anytext_colab.py"),
        "--plan-path", str(repo_root / "research" / "evidence" / "document_pilot" / "anytext_feasibility_plan.json"),
        "--inputs-zip", args.archive, "--output-root", str(out), "--cache-dir", args.cache_dir,
        "--custom-font", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "--run-label", args.label,
    ]
    if args.preflight_only:
        cmd.append("--preflight-only")
    env = {**os.environ, "PYTHONPATH": str(ANYTEXT_DIR), "PYTHONUNBUFFERED": "1"}
    env.pop("ATTN_PRECISION", None)  # the precision policy must not be overridden from the environment
    print(f"Full log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ANYTEXT_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for raw in proc.stdout:
            log.write(raw)
            log.flush()
            shown = filter_console_line(raw)
            if shown:
                print(shown, flush=True)
        proc.wait()
    print(f"Runner exit code: {proc.returncode}")
    if args.preflight_only:
        return proc.returncode
    return cmd_result(argparse.Namespace(output_root=str(out)), runner_rc=proc.returncode)


def cmd_result(args: argparse.Namespace, runner_rc: int = 0) -> int:
    out = Path(args.output_root)
    pointer_path = out / "latest_run.json"
    if not pointer_path.is_file():
        print("[FAIL] No latest_run.json: the runner did not reach packaging. See the newest file in", out / "logs")
        for rc in sorted(out.glob("failure_receipt_init_*.json"))[-1:]:
            print(json.loads(rc.read_text(encoding="utf-8")).get("traceback", ""))
        return runner_rc or 1
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    run_dir = out / pointer["run_dir"]
    print(f"Run {pointer['run_id']}: {pointer['status']}")
    print(f"Archive: {out / pointer['archive']} (sha256 {pointer['archive_sha256'][:16]}...)")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    for att_id, st in manifest["attempts"].items():
        print(f"  {att_id}: {st['status']}")
        rcpt = run_dir / att_id / "failure_receipt.json"
        if rcpt.is_file():
            r = json.loads(rcpt.read_text(encoding="utf-8"))
            frame = r.get("failure_site", {}).get("innermost_project_frame", {})
            print(f"    {r.get('error_type')}: {r.get('error')}")
            print(f"    at {frame.get('file')}:{frame.get('line')} in {frame.get('function')}")
            print(f"    tensors: {json.dumps(frame.get('tensors', {}))}")
            print(f"    modules: {json.dumps(frame.get('module_parameter_dtypes', {}))}")
    return 0 if pointer["status"] == "COMPLETE" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("env"); p.add_argument("--repo-root", default=str(REPO_ROOT)); p.set_defaults(fn=cmd_env)
    p = sub.add_parser("smoke"); p.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"); p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("run")
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    p.add_argument("--archive", required=True)
    p.add_argument("--output-root", default="/content/outputs")
    p.add_argument("--cache-dir", default="/content/modelscope_cache")
    p.add_argument("--label", default="repair_fp16_xformers")
    p.add_argument("--preflight-only", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("result"); p.add_argument("--output-root", default="/content/outputs")
    p.set_defaults(fn=lambda a: cmd_result(a))
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
