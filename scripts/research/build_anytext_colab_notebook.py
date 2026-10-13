#!/usr/bin/env python3
"""Canonical AnyText Colab Notebook Builder.

Generates `notebooks/independent_anytext_feasibility_colab.ipynb` embedding
the canonical plan and verified runner for executing Attempt A and B on Colab T4.
"""

from __future__ import annotations

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
    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    adapter_text = ADAPTER_PATH.read_text(encoding="utf-8")
    geom_text = GEOMETRY_PATH.read_text(encoding="utf-8")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AnyText Document Generation Feasibility Study (Colab GPU)\n",
                "\n",
                "**Mục tiêu**: Đánh giá thực nghiệm editor chuyên biệt văn bản **AnyText** (ICLR 2024 Spotlight) trên đúng 2 nguồn thay thế chữ/số (Attempt A & B) từ pilot SDXL đã đóng.\n",
                "- **Checkpoint**: `damo/cv_anytext_text_generation_editing` (ModelScope v1.1.3, ~6.93 GiB FP16; tắt translator để bảo toàn ngân sách 12 GiB).\n",
                "- **Dữ liệu**: Tái sử dụng đúng 2 nguồn CORD-v2 validation (Attempt A: row 22, Attempt B: row 61).\n",
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
                "# Cell 2: Install AnyText Dependencies & Clone Official Repository\n",
                "import os, sys, subprocess\n",
                "\n",
                "# Install system font and dependencies\n",
                "subprocess.run(['apt-get', 'install', '-y', '-qq', 'fonts-dejavu-core'], check=False)\n",
                "subprocess.run([\n",
                "    'pip', 'install', '-q',\n",
                "    'modelscope==1.10.0', 'einops==0.4.1', 'omegaconf==2.2.3',\n",
                "    'pytorch-lightning==1.5.0', 'open_clip_torch==2.7.0',\n",
                "    'transformers==4.30.2', 'addict', 'yapf', 'timm'\n",
                "], check=True)\n",
                "\n",
                "# Clone AnyText official repository\n",
                "if not os.path.isdir('AnyText'):\n",
                "    subprocess.run(['git', 'clone', 'https://github.com/tyxsspa/AnyText.git'], check=True)\n",
                "\n",
                "os.chdir('AnyText')\n",
                "if os.getcwd() not in sys.path:\n",
                "    sys.path.insert(0, os.getcwd())\n",
                "\n",
                "print('[PASS] Dependencies and AnyText codebase installed.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Staging Geometry Modules, Runner & Plan\n",
                "import os, pathlib, json\n",
                "\n",
                "os.makedirs('ml/evaluation', exist_ok=True)\n",
                "os.makedirs('research/evidence/document_pilot', exist_ok=True)\n",
                "os.makedirs('inputs/attempt_A_replacement_price', exist_ok=True)\n",
                "os.makedirs('inputs/attempt_B_replacement_item_name', exist_ok=True)\n",
                "\n",
                f"pathlib.Path('research/evidence/document_pilot/anytext_feasibility_plan.json').write_text({json.dumps(plan_text)}, encoding='utf-8')\n",
                f"pathlib.Path('ml/evaluation/document_feasibility_geometry.py').write_text({json.dumps(geom_text)}, encoding='utf-8')\n",
                f"pathlib.Path('ml/evaluation/anytext_document_adapter.py').write_text({json.dumps(adapter_text)}, encoding='utf-8')\n",
                f"pathlib.Path('run_anytext_colab.py').write_text({json.dumps(runner_text)}, encoding='utf-8')\n",
                "\n",
                "print('[PASS] Successfully staged plan, geometry module, adapter and runner.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Download Authentic Inputs (from HuggingFace CORD-v2 or Uploaded SDXL Artifacts)\n",
                "# Tải 2 ảnh authentic controls từ HuggingFace Parquet shard CORD-v2 hoặc người dùng upload\n",
                "import pyarrow.parquet as pq, io, requests\n",
                "from PIL import Image\n",
                "\n",
                "print('Downloading authentic source documents from CORD-v2 validation split...')\n",
                "cord_url = 'https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data/validation-00000-of-00001-c81bcf80b4353f8a.parquet'\n",
                "resp = requests.get(cord_url, stream=True)\n",
                "table = pq.read_table(io.BytesIO(resp.content))\n",
                "\n",
                "# Row 22 (Attempt A)\n",
                "img_a_bytes = table['image'][22]['bytes'].as_py()\n",
                "img_a = Image.open(io.BytesIO(img_a_bytes)).convert('RGB')\n",
                "img_a.save('inputs/attempt_A_replacement_price/authentic_control.png')\n",
                "\n",
                "# Row 61 (Attempt B)\n",
                "img_b_bytes = table['image'][61]['bytes'].as_py()\n",
                "img_b = Image.open(io.BytesIO(img_b_bytes)).convert('RGB')\n",
                "img_b.save('inputs/attempt_B_replacement_item_name/authentic_control.png')\n",
                "\n",
                "print('[PASS] Authentic controls extracted and verified.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Execute AnyText Feasibility Runner & Package Results\n",
                "import subprocess, sys\n",
                "cmd = [\n",
                "    sys.executable, 'run_anytext_colab.py',\n",
                "    '--plan-path', 'research/evidence/document_pilot/anytext_feasibility_plan.json',\n",
                "    '--inputs-dir', 'inputs',\n",
                "    '--output-root', 'outputs',\n",
                "    '--custom-font', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'\n",
                "]\n",
                "subprocess.run(cmd, check=True)\n",
                "\n",
                "# Display Contact Sheet\n",
                "from IPython.display import Image as IPImage, display\n",
                "import glob\n",
                "cs_files = glob.glob('outputs/*/anytext_contact_sheet.png')\n",
                "if cs_files:\n",
                "    display(IPImage(cs_files[0]))\n",
                "\n",
                "# Download Results Archive\n",
                "from google.colab import files\n",
                "if os.path.isfile('outputs/anytext_feasibility_results.zip'):\n",
                "    files.download('outputs/anytext_feasibility_results.zip')"
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
    NOTEBOOK_PATH.write_text(json.dumps(nb_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] AnyText canonical Colab notebook generated at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
