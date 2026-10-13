#!/usr/bin/env python3
"""Canonical AnyText Colab Notebook Builder.

Generates `notebooks/independent_anytext_feasibility_colab.ipynb` embedding
the canonical plan and verified runner for executing Attempt A and B on Colab T4.
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
                "# Cell 2: Install AnyText Dependencies & Clone Repository at Locked Commit\n",
                "import os, sys, subprocess\n",
                "\n",
                "# Install system font and python dependencies\n",
                "subprocess.run(['apt-get', 'install', '-y', '-qq', 'fonts-dejavu-core'], check=False)\n",
                "subprocess.run([\n",
                "    'pip', 'install', '-q',\n",
                "    'modelscope==1.10.0', 'einops==0.4.1', 'omegaconf==2.2.3',\n",
                "    'pytorch-lightning==1.5.0', 'open_clip_torch==2.7.0',\n",
                "    'transformers==4.30.2', 'addict', 'yapf', 'timm'\n",
                "], check=True)\n",
                "\n",
                "# Clone AnyText official repository and checkout locked commit\n",
                "locked_commit = '3d4a260c67d92c0e27d6e18ed69990e19a40de6d'\n",
                "if not os.path.isdir('AnyText'):\n",
                "    subprocess.run(['git', 'clone', 'https://github.com/tyxsspa/AnyText.git'], check=True)\n",
                "\n",
                "os.chdir('AnyText')\n",
                "subprocess.run(['git', 'checkout', locked_commit], check=True)\n",
                "head_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()\n",
                "assert head_commit == locked_commit, f'Git commit mismatch! Expected {locked_commit}, got {head_commit}'\n",
                "\n",
                "if os.getcwd() not in sys.path:\n",
                "    sys.path.insert(0, os.getcwd())\n",
                "\n",
                "print(f'[PASS] AnyText codebase installed and verified at commit: {head_commit}')"
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
                "os.makedirs('ml/evaluation', exist_ok=True)\n",
                "os.makedirs('research/evidence/document_pilot', exist_ok=True)\n",
                "\n",
                f"p_plan = pathlib.Path('research/evidence/document_pilot/anytext_feasibility_plan.json')\n",
                f"p_plan.write_bytes({repr(plan_text)}.encode('utf-8'))\n",
                f"assert hashlib.sha256(p_plan.read_bytes()).hexdigest() == '{plan_sha}', 'Plan SHA mismatch after staging!'\n",
                "\n",
                f"p_geom = pathlib.Path('ml/evaluation/document_feasibility_geometry.py')\n",
                f"p_geom.write_bytes({repr(geom_text)}.encode('utf-8'))\n",
                f"assert hashlib.sha256(p_geom.read_bytes()).hexdigest() == '{geom_sha}', 'Geometry SHA mismatch after staging!'\n",
                "\n",
                f"p_adapt = pathlib.Path('ml/evaluation/anytext_document_adapter.py')\n",
                f"p_adapt.write_bytes({repr(adapter_text)}.encode('utf-8'))\n",
                f"assert hashlib.sha256(p_adapt.read_bytes()).hexdigest() == '{adapter_sha}', 'Adapter SHA mismatch after staging!'\n",
                "\n",
                f"p_runner = pathlib.Path('run_anytext_colab.py')\n",
                f"p_runner.write_bytes({repr(runner_text)}.encode('utf-8'))\n",
                f"assert hashlib.sha256(p_runner.read_bytes()).hexdigest() == '{runner_sha}', 'Runner SHA mismatch after staging!'\n",
                "\n",
                "print('[PASS] Staged plan, geometry, adapter, and runner with bit-exact SHA-256 verification.')"
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
                "# Cell 5: Execute AnyText Feasibility Runner (Safe Isolation, Filtered Download, Zoomed ROI)\n",
                "import subprocess, sys\n",
                "\n",
                "cmd = [\n",
                "    sys.executable, 'run_anytext_colab.py',\n",
                "    '--plan-path', 'research/evidence/document_pilot/anytext_feasibility_plan.json',\n",
                "    '--inputs-zip', '/content/document_feasibility_results.zip',\n",
                "    '--output-root', '/content/outputs',\n",
                "    '--cache-dir', '/content/modelscope_cache',\n",
                "    '--custom-font', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'\n",
                "]\n",
                "print('Bắt đầu thực thi AnyText Runner trên GPU...')\n",
                "subprocess.run(cmd, check=True)\n",
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
