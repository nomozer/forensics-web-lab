#!/usr/bin/env python3
"""Canonical AnyText Colab Notebook Builder.

Generates `notebooks/independent_anytext_feasibility_colab.ipynb` embedding
the canonical plan and verified runner for executing Attempt A and B on Colab T4.
Uses a dedicated Python 3.10 environment to match AnyText upstream environment.yaml.
Inputs are loaded strictly from the verified SDXL pilot archive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "independent_anytext_feasibility_colab.ipynb"
PLAN_PATH = REPO_ROOT / "research" / "evidence" / "document_pilot" / "anytext_feasibility_plan.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "research" / "run_anytext_colab.py"
ADAPTER_PATH = REPO_ROOT / "ml" / "evaluation" / "anytext_document_adapter.py"
GEOMETRY_PATH = REPO_ROOT / "ml" / "evaluation" / "document_feasibility_geometry.py"


def build_notebook() -> None:
    """Builds canonical Colab notebook for AnyText."""
    # Read strict UTF-8 bytes to ensure byte-exact SHA-256 invariance
    plan_bytes = PLAN_PATH.read_bytes()
    runner_bytes = RUNNER_PATH.read_bytes()
    adapter_bytes = ADAPTER_PATH.read_bytes()
    geom_bytes = GEOMETRY_PATH.read_bytes()

    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    runner_sha = hashlib.sha256(runner_bytes).hexdigest()
    adapter_sha = hashlib.sha256(adapter_bytes).hexdigest()
    geom_sha = hashlib.sha256(geom_bytes).hexdigest()

    plan_text = plan_bytes.decode("utf-8")
    runner_text = runner_bytes.decode("utf-8")
    adapter_text = adapter_bytes.decode("utf-8")
    geom_text = geom_bytes.decode("utf-8")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AnyText Document Generation Feasibility Study (Colab GPU)\n",
                "\n",
                "**Mục tiêu**: Đánh giá thực nghiệm editor chuyên biệt văn bản **AnyText** (ICLR 2024 Spotlight) trên đúng 2 nguồn thay thế chữ/số (Attempt A & B) từ pilot SDXL đã đóng.\n",
                "- **Checkpoint**: `damo/cv_anytext_text_generation_editing` (ModelScope v1.1.3, ~6.93 GiB FP16; tắt và loại bỏ translator để bảo toàn ngân sách 12 GiB).\n",
                "- **Môi trường Python**: Thiết lập môi trường Python 3.10 riêng biệt (`/content/anytext_env`) khớp hoàn toàn `environment.yaml` upstream; không can thiệp Python hệ thống của Colab (Python 3.13) và không nâng packages lên latest.\n",
                "- **Dữ liệu đầu vào**: Tải trực tiếp từ archive SDXL đã có (`document_feasibility_results.zip`, SHA-256 `deadd9205207...`). Không tải lại shard CORD parquet.\n",
                "- **Bảo toàn hình học**: Native dimensions, context crop 512x512, strict composite inside target bbox, kiểm tra bitwise outside-mask equality.\n",
                "- **Yêu cầu môi trường**: Google Colab GPU (Tesla T4 miễn phí, >= 15 GB disk, zero-token, zero-login)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Hardware Preflight & Disk Inspection\n",
                "import torch, shutil, os\n",
                "\n",
                "assert torch.cuda.is_available(), 'ERROR: GPU runtime required! Please enable GPU via Runtime -> Change runtime type.'\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)\n",
                "disk_free_gb = round(shutil.disk_usage('/').free / (1024**3), 2)\n",
                "\n",
                "print(f'[PASS] GPU Detected: {gpu_name} (VRAM: {vram_gb} GB)')\n",
                "print(f'[PASS] Disk Free: {disk_free_gb} GB (Requires >= 15 GB for ModelScope download)')\n",
                "assert vram_gb >= 7.0, f'VRAM {vram_gb} GB is insufficient for AnyText (requires >= 7 GB).'\n",
                "assert disk_free_gb >= 12.0, f'Disk space {disk_free_gb} GB is insufficient for 6.93 GiB checkpoint.'"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Setup Dedicated Python 3.10 Runtime, Clone AnyText & Verify Dependencies\n",
                "import os, pathlib, shutil, subprocess, sys\n",
                "\n",
                "# 1. Install system font\n",
                "subprocess.run(['apt-get', 'update', '-qq'], check=False, stdout=subprocess.DEVNULL)\n",
                "subprocess.run(['apt-get', 'install', '-y', '-qq', 'fonts-dejavu-core'], check=False)\n",
                "\n",
                "# 2. Clone AnyText official repository to absolute path /content/AnyText and checkout locked commit\n",
                "anytext_repo_dir = pathlib.Path('/content/AnyText')\n",
                "locked_commit = '3d4a260c67d92c0e27d6e18ed69990e19a40de6d'\n",
                "if not (anytext_repo_dir / '.git').is_dir():\n",
                "    print(f'[INFO] Cloning AnyText repository into {anytext_repo_dir}...')\n",
                "    subprocess.run(['git', 'clone', 'https://github.com/tyxsspa/AnyText.git', str(anytext_repo_dir)], check=True)\n",
                "\n",
                "subprocess.run(['git', 'checkout', locked_commit], cwd=str(anytext_repo_dir), check=True)\n",
                "head_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(anytext_repo_dir)).decode('utf-8').strip()\n",
                "assert head_commit == locked_commit, f'Git commit mismatch! Expected {locked_commit}, got {head_commit}'\n",
                "os.chdir(str(anytext_repo_dir))\n",
                "print(f'[PASS] AnyText codebase verified at absolute path {anytext_repo_dir} (commit: {head_commit})')\n",
                "\n",
                "# 3. Create isolated Python 3.10 environment (/content/anytext_env)\n",
                "# Host Python on Colab may be 3.13; we explicitly target Python 3.10\n",
                "env_dir = pathlib.Path('/content/anytext_env')\n",
                "py_bin = env_dir / 'bin' / 'python'\n",
                "pip_bin = env_dir / 'bin' / 'pip'\n",
                "\n",
                "if not py_bin.is_file():\n",
                "    py310_sys = shutil.which('python3.10')\n",
                "    if not py310_sys:\n",
                "        print('[INFO] Installing python3.10 and python3.10-venv from deadsnakes PPA...')\n",
                "        subprocess.run(['apt-get', 'install', '-y', '-qq', 'software-properties-common'], check=False, stdout=subprocess.DEVNULL)\n",
                "        subprocess.run(['add-apt-repository', '-y', 'ppa:deadsnakes/ppa'], check=False, stdout=subprocess.DEVNULL)\n",
                "        subprocess.run(['apt-get', 'update', '-qq'], check=False, stdout=subprocess.DEVNULL)\n",
                "        subprocess.run(['apt-get', 'install', '-y', '-qq', 'python3.10', 'python3.10-venv', 'python3.10-dev'], check=False)\n",
                "        py310_sys = shutil.which('python3.10')\n",
                "\n",
                "    if py310_sys:\n",
                "        print(f'[INFO] Creating virtualenv using system {py310_sys}...')\n",
                "        subprocess.run([py310_sys, '-m', 'venv', str(env_dir)], check=True)\n",
                "    else:\n",
                "        print('[INFO] System python3.10 not found. Bootstrapping standalone Python 3.10 via uv...')\n",
                "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)\n",
                "        subprocess.run(['uv', 'venv', str(env_dir), '--python', '3.10', '--seed'], check=True)\n",
                "\n",
                "assert py_bin.is_file(), f'Dedicated Python 3.10 binary failed to create at {py_bin}'\n",
                "py_ver = subprocess.check_output([str(py_bin), '--version']).decode('utf-8').strip()\n",
                "print(f'[PASS] Dedicated Python environment verified: {py_ver} at {py_bin}')\n",
                "assert '3.10' in py_ver, f'Expected Python 3.10 environment, but found {py_ver}! Refusing to proceed.'\n",
                "\n",
                "# 4. Lock installer toolchain versions (avoid unverified latest upgrades)\n",
                "print('[INFO] Locking installer versions (pip==23.3.2, setuptools==68.2.2, wheel==0.42.0)...')\n",
                "subprocess.run([str(pip_bin), 'install', 'pip==23.3.2', 'setuptools==68.2.2', 'wheel==0.42.0'], check=True)\n",
                "\n",
                "# 5. Install PyTorch 2.0.1 (CUDA 11.8) matching upstream environment.yaml\n",
                "print('[INFO] Installing PyTorch 2.0.1 (CUDA 11.8) matching upstream environment.yaml...')\n",
                "subprocess.run([\n",
                "    str(pip_bin), 'install',\n",
                "    'torch==2.0.1', 'torchvision==0.15.2',\n",
                "    '--index-url', 'https://download.pytorch.org/whl/cu118'\n",
                "], check=True)\n",
                "\n",
                "# 6. Install locked AnyText inference dependencies from upstream environment.yaml & import graph\n",
                "print('[INFO] Installing AnyText locked inference dependencies...')\n",
                "deps = [\n",
                "    'numpy==1.23.3',\n",
                "    'Pillow==9.5.0',\n",
                "    'torchmetrics==0.11.4',\n",
                "    'pytorch-lightning==1.5.0',\n",
                "    'omegaconf==2.2.3',\n",
                "    'einops==0.4.1',\n",
                "    'opencv-python==4.7.0.72',\n",
                "    'transformers==4.30.2',\n",
                "    'open_clip_torch==2.7.0',\n",
                "    'modelscope==1.10.0',\n",
                "    'addict==2.4.0',\n",
                "    'yapf==0.32.0',\n",
                "    'timm==0.6.7',\n",
                "    'safetensors==0.4.0',\n",
                "    'easydict==1.10',\n",
                "    'scikit-image==0.20.0',\n",
                "    'scipy==1.10.1',\n",
                "    'six==1.16.0',\n",
                "    'tqdm==4.66.1'\n",
                "]\n",
                "subprocess.run([str(pip_bin), 'install'] + deps, check=True)\n",
                "\n",
                "# 7. pip check verification\n",
                "print('[INFO] Executing pip check on dedicated environment...')\n",
                "chk_res = subprocess.run([str(pip_bin), 'check'], capture_output=True, text=True)\n",
                "print(chk_res.stdout)\n",
                "if chk_res.stderr:\n",
                "    print(chk_res.stderr)\n",
                "assert chk_res.returncode == 0, f'pip check failed with dependency conflict:\\n{chk_res.stdout}\\n{chk_res.stderr}'\n",
                "print('[PASS] pip check passed with 0 broken requirements.')\n",
                "\n",
                "# 8. Import Smoke Test and Functional Glyph/Position Guidance Test\n",
                "print('[INFO] Executing import and functional smoke test in Python 3.10...')\n",
                "smoke_code = (\n",
                "    'import sys\\n'\n",
                "    'print(f\"[SMOKE] Python interpreter: {sys.executable} ({sys.version})\")\\n'\n",
                "    'assert \"3.10\" in sys.version, f\"Wrong python version: {sys.version}\"\\n'\n",
                "    '\\n'\n",
                "    'import torch, torchvision, numpy as np, PIL, pytorch_lightning as pl, torchmetrics\\n'\n",
                "    'print(f\"[SMOKE] Package versions: torch={torch.__version__}, torchvision={torchvision.__version__}, numpy={np.__version__}, PIL={PIL.__version__}, pl={pl.__version__}, torchmetrics={torchmetrics.__version__}\")\\n'\n",
                "    'assert np.__version__ == \"1.23.3\", f\"numpy version mismatch: {np.__version__}\"\\n'\n",
                "    'assert PIL.__version__ == \"9.5.0\", f\"PIL version mismatch: {PIL.__version__}\"\\n'\n",
                "    'assert pl.__version__ == \"1.5.0\", f\"pl version mismatch: {pl.__version__}\"\\n'\n",
                "    'assert torchmetrics.__version__ == \"0.11.4\", f\"torchmetrics version mismatch: {torchmetrics.__version__}\"\\n'\n",
                "    'assert torch.__version__.startswith(\"2.0.1\"), f\"torch version mismatch: {torch.__version__}\"\\n'\n",
                "    'assert torchvision.__version__.startswith(\"0.15.2\"), f\"torchvision version mismatch: {torchvision.__version__}\"\\n'\n",
                "    '\\n'\n",
                "    'assert torch.cuda.is_available(), \"CUDA is not available in PyTorch! Check Colab GPU runtime.\"\\n'\n",
                "    'print(f\"[SMOKE CUDA] Device: {torch.cuda.get_device_name(0)}, CUDA version: {torch.version.cuda}\")\\n'\n",
                "    '\\n'\n",
                "    'import cv2, einops, omegaconf, transformers, open_clip, addict, yapf, timm, safetensors, easydict, skimage, six, tqdm, modelscope\\n'\n",
                "    'from modelscope.pipelines import pipeline\\n'\n",
                "    'import cldm.model, cldm.cldm, cldm.ddim_hacked, cldm.recognizer, t3_dataset, util, bert_tokenizer\\n'\n",
                "    'print(\"[PASS] Import smoke test: 100% of required inference modules and AnyText submodules imported cleanly!\")\\n'\n",
                "    '\\n'\n",
                "    '# Functional Smoke Test: Generate glyph and position guidance with DejaVuSans for both target texts without loading checkpoint\\n'\n",
                "    'from PIL import ImageFont\\n'\n",
                "    'font_path = \"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf\"\\n'\n",
                "    'font = ImageFont.truetype(font_path, size=60)\\n'\n",
                "    '\\n'\n",
                "    '# Target 1: \"350,000\"\\n'\n",
                "    't1 = \"350,000\"\\n'\n",
                "    'g1 = t3_dataset.draw_glyph(font, t1)\\n'\n",
                "    'assert g1.shape == (80, 512, 1), f\"Unexpected glyph shape: {g1.shape}\"\\n'\n",
                "    'assert g1.sum() > 0, \"Target 1 glyph rasterization has 0 pixels!\"\\n'\n",
                "    'poly1 = np.array([[100, 200], [300, 200], [300, 280], [100, 280]])\\n'\n",
                "    'gp1 = t3_dataset.draw_glyph2(font, t1, poly1, width=512, height=512)\\n'\n",
                "    'assert gp1.shape == (512, 512, 1), f\"Unexpected pos shape: {gp1.shape}\"\\n'\n",
                "    'assert gp1.sum() > 0, \"Target 1 position guidance has 0 pixels!\"\\n'\n",
                "    '\\n'\n",
                "    '# Target 2: \"Magnum Gold Almond\"\\n'\n",
                "    't2 = \"Magnum Gold Almond\"\\n'\n",
                "    'g2 = t3_dataset.draw_glyph(font, t2)\\n'\n",
                "    'assert g2.shape == (80, 512, 1), f\"Unexpected glyph shape: {g2.shape}\"\\n'\n",
                "    'assert g2.sum() > 0, \"Target 2 glyph rasterization has 0 pixels!\"\\n'\n",
                "    'poly2 = np.array([[50, 150], [450, 150], [450, 250], [50, 250]])\\n'\n",
                "    'gp2 = t3_dataset.draw_glyph2(font, t2, poly2, width=512, height=512)\\n'\n",
                "    'assert gp2.shape == (512, 512, 1), f\"Unexpected pos shape: {gp2.shape}\"\\n'\n",
                "    'assert gp2.sum() > 0, \"Target 2 position guidance has 0 pixels!\"\\n'\n",
                "    '\\n'\n",
                "    'print(f\"[PASS] Functional smoke test passed: glyphs & position guidance generated for \\'{t1}\\' (glyph: {int(g1.sum())}, pos: {int(gp1.sum())}) and \\'{t2}\\' (glyph: {int(g2.sum())}, pos: {int(gp2.sum())}).\")\\n'\n",
                ")\n",
                "smoke_env = os.environ.copy()\n",
                "smoke_env['PYTHONPATH'] = str(anytext_repo_dir)\n",
                "smoke_res = subprocess.run([str(py_bin), '-c', smoke_code], capture_output=True, text=True, cwd=str(anytext_repo_dir), env=smoke_env)\n",
                "print(smoke_res.stdout)\n",
                "if smoke_res.stderr:\n",
                "    print(smoke_res.stderr)\n",
                "assert smoke_res.returncode == 0, f'Smoke test failed:\\n{smoke_res.stderr}'"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Staging Geometry Modules, Runner & Plan with Bit-Exact Verification\n",
                "import os, pathlib, hashlib\n",
                "\n",
                "repo_dir = pathlib.Path('/content/AnyText')\n",
                "os.chdir(str(repo_dir))\n",
                "\n",
                "(repo_dir / 'ml/evaluation').mkdir(parents=True, exist_ok=True)\n",
                "(repo_dir / 'research/evidence/document_pilot').mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "p_plan = repo_dir / 'research/evidence/document_pilot/anytext_feasibility_plan.json'\n",
                "p_plan.write_bytes(" + repr(plan_text) + ".encode('utf-8'))\n",
                "assert hashlib.sha256(p_plan.read_bytes()).hexdigest() == '" + plan_sha + "', 'Plan SHA mismatch after staging!'\n",
                "\n",
                "p_geom = repo_dir / 'ml/evaluation/document_feasibility_geometry.py'\n",
                "p_geom.write_bytes(" + repr(geom_text) + ".encode('utf-8'))\n",
                "assert hashlib.sha256(p_geom.read_bytes()).hexdigest() == '" + geom_sha + "', 'Geometry SHA mismatch after staging!'\n",
                "\n",
                "p_adapt = repo_dir / 'ml/evaluation/anytext_document_adapter.py'\n",
                "p_adapt.write_bytes(" + repr(adapter_text) + ".encode('utf-8'))\n",
                "assert hashlib.sha256(p_adapt.read_bytes()).hexdigest() == '" + adapter_sha + "', 'Adapter SHA mismatch after staging!'\n",
                "\n",
                "p_runner = repo_dir / 'run_anytext_colab.py'\n",
                "p_runner.write_bytes(" + repr(runner_text) + ".encode('utf-8'))\n",
                "assert hashlib.sha256(p_runner.read_bytes()).hexdigest() == '" + runner_sha + "', 'Runner SHA mismatch after staging!'\n",
                "\n",
                "print('[PASS] Staged plan, geometry, adapter, and runner at /content/AnyText with bit-exact SHA-256 verification.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Upload and Verify SDXL Feasibility Archive\n",
                "# Archive contains authentic controls and SDXL pilot outputs from the closed pilot.\n",
                "import hashlib, os, pathlib\n",
                "from google.colab import files\n",
                "\n",
                "archive_name = 'document_feasibility_results.zip'\n",
                "archive_path = pathlib.Path(f'/content/{archive_name}')\n",
                "expected_archive_sha = 'deadd9205207eaaa296a8d9eeb493ee48eeea76ee7afd210684187e4026ed684'\n",
                "\n",
                "if not archive_path.is_file():\n",
                "    print(f'Vui lòng tải lên tệp: {archive_name}')\n",
                "    uploaded = files.upload()\n",
                "    for fname, data in uploaded.items():\n",
                "        if fname.endswith('.zip'):\n",
                "            archive_path.write_bytes(data)\n",
                "            break\n",
                "\n",
                "assert archive_path.is_file(), f'Không tìm thấy {archive_path}! Vui lòng upload archive SDXL.'\n",
                "observed_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()\n",
                "assert observed_sha == expected_archive_sha, f'Archive SHA mismatch! Expected {expected_archive_sha}, got {observed_sha}'\n",
                "print(f'[PASS] Archive SDXL đã xác nhận: {archive_path} (SHA-256: {observed_sha[:16]}...)')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Execute AnyText Feasibility Runner using Dedicated Python 3.10 Environment\n",
                "import os, pathlib, subprocess, sys\n",
                "\n",
                "repo_dir = pathlib.Path('/content/AnyText')\n",
                "os.chdir(str(repo_dir))\n",
                "\n",
                "py_bin = '/content/anytext_env/bin/python'\n",
                "cmd = [\n",
                "    py_bin, str(repo_dir / 'run_anytext_colab.py'),\n",
                "    '--plan-path', str(repo_dir / 'research/evidence/document_pilot/anytext_feasibility_plan.json'),\n",
                "    '--inputs-zip', '/content/document_feasibility_results.zip',\n",
                "    '--output-root', '/content/outputs',\n",
                "    '--cache-dir', '/content/modelscope_cache',\n",
                "    '--custom-font', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'\n",
                "]\n",
                "print('Bắt đầu thực thi AnyText Runner trên GPU bằng Python 3.10...')\n",
                "subprocess.run(cmd, check=True, cwd=str(repo_dir))\n",
                "print('[PASS] Quá trình thực thi runner hoàn thành.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Visual Inspection & Download Whitelisted Results Archive\n",
                "import glob, os\n",
                "from IPython.display import Image as IPImage, display\n",
                "from google.colab import files\n",
                "\n",
                "# Hiển thị contact sheet phóng đại\n",
                "cs_files = glob.glob('/content/outputs/*/anytext_contact_sheet.png')\n",
                "if cs_files:\n",
                "    print('Bảng đối chiếu phóng đại (Zoomed ROI Contact Sheet):')\n",
                "    display(IPImage(cs_files[0]))\n",
                "\n",
                "# Kiểm tra và tải về kết quả đã đóng gói (<= 100 MiB)\n",
                "out_zip = '/content/outputs/anytext_feasibility_results.zip'\n",
                "if os.path.isfile(out_zip):\n",
                "    size_mib = os.path.getsize(out_zip) / (1024**2)\n",
                "    assert size_mib <= 100.0, f'Archive size {size_mib:.2f} MiB vượt quá trần 100 MiB!'\n",
                "    print(f'[PASS] Kết quả đóng gói hợp lệ ({size_mib:.2f} MiB). Bắt đầu download...')\n",
                "    files.download(out_zip)\n",
                "else:\n",
                "    print('[WARNING] Không tìm thấy archive kết quả.')"
            ]
        }
    ]

    nb_data = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write notebook with strict UTF-8
    NOTEBOOK_PATH.write_text(json.dumps(nb_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] AnyText canonical Colab notebook generated at: {NOTEBOOK_PATH}")
    print(f"     SHA-256: {hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    build_notebook()
