#!/usr/bin/env python3
"""Programmatically build independent_document_feasibility_colab.ipynb from canonical plan and runner.

Guarantees 100% bitwise parity between repository canonical artifacts and Colab launcher payload,
eliminating manual copy-paste errors, indentation bugs, and seed drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_plan.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "research" / "run_document_feasibility_colab.py"
NOTEBOOK_OUTPUT_PATH = REPO_ROOT / "notebooks" / "independent_document_feasibility_colab.ipynb"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_python_code(code_str: str, label: str) -> None:
    """Validate python syntax, ignoring bash lines starting with !."""
    python_lines = []
    for line in code_str.split("\n"):
        stripped = line.strip()
        if stripped.startswith("!") or stripped.startswith("%"):
            python_lines.append(f"# {line}")
        else:
            python_lines.append(line)
    sanitized_code = "\n".join(python_lines)
    compile(sanitized_code, filename=label, mode="exec")


def build_notebook() -> Dict[str, Any]:
    with open(PLAN_PATH, "r", encoding="utf-8") as pf:
        plan_json_str = pf.read()
    plan_data = json.loads(plan_json_str)
    plan_sha = compute_file_sha256(PLAN_PATH)

    with open(RUNNER_PATH, "r", encoding="utf-8") as rf:
        runner_source = rf.read()
    runner_sha = compute_file_sha256(RUNNER_PATH)

    cells: List[Dict[str, Any]] = []

    # Cell 0: Markdown documentation
    doc_markdown = [
        "# Controlled AI-Document Generation Feasibility Study (Colab GPU Launcher)\n",
        "\n",
        "> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>\n",
        f"> **Protocol**: Version {plan_data['metadata']['protocol_version']} (Amendment v1.2.0)<br>\n",
        "> **Mục tiêu**: Thử nghiệm khả thi tạo dữ liệu trên đúng **bốn tài liệu CORD v2** (2 replacement, 2 removal) bằng `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`.<br>\n",
        "> **Ranh giới**: Đây là generation feasibility study, KHÔNG phải detector training, KHÔNG phải cohort đánh giá độc lập.<br>\n",
        "> **Không yêu cầu token/đăng nhập**: Chạy hoàn toàn bằng tài khoản Google Colab miễn phí thông thường (GPU T4). Không yêu cầu Hugging Face token.\n",
        "\n",
        "### Quy Trình 4 Bước Thực Hiện Trên Colab:\n",
        "1. **Cell 1**: Kiểm tra phần cứng GPU và cài đặt thư viện cần thiết (`diffusers`, `transformers`, `accelerate`, `safetensors`, `pyarrow`, `pillow`).\n",
        "2. **Cell 2**: Tái tạo tự động kế hoạch máy tiền đăng ký `document_feasibility_plan.json` và mã nguồn thực thi `run_document_feasibility_colab.py` (đối soát SHA-256 bit-exact).\n",
        "3. **Cell 3**: Thực thi 4 attempts đã khóa cứng: tải checkpoint FP16 (6.46 GiB), nạp 4 ảnh CORD-v2 từ cache, inpainting suy luận, composite và kiểm tra bitwise equality ngoài mask.\n",
        "4. **Cell 4**: Tải gói kết quả `document_feasibility_results.zip` về máy LOCAL để kiểm chứng kỹ thuật và thẩm định nội dung.\n",
    ]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": doc_markdown})

    # Cell 1: Environment & GPU setup
    cell1_code = [
        "# Cell 1: Kiểm tra phần cứng GPU và cài đặt dependencies\n",
        "!nvidia-smi\n",
        "\n",
        "print(\"Installing dependencies (diffusers, transformers, accelerate, safetensors, pyarrow, pillow)...\")\n",
        "!pip install -q \"diffusers>=0.21.0\" \"transformers>=4.34.0\" \"accelerate>=0.23.0\" safetensors pyarrow pillow\n",
        "\n",
        "import torch\n",
        "assert torch.cuda.is_available(), \"LỖI: Chưa kích hoạt GPU! Vui lòng vào Runtime -> Change runtime type -> Chọn T4 GPU!\"\n",
        "gpu_name = torch.cuda.get_device_name(0)\n",
        "vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
        "print(f\"[OK] GPU sẵn sàng: {gpu_name} ({vram_gb:.2f} GB VRAM)\")\n",
        "if vram_gb < 12.0:\n",
        "    print(f\"CẢNH BÁO: VRAM ({vram_gb:.2f} GB) thấp hơn 12 GB. Attention slicing sẽ được kích hoạt để tránh OOM.\")\n",
    ]
    cell1_str = "".join(cell1_code)
    validate_python_code(cell1_str, "Cell 1")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1_code})

    # Cell 2: Staging plan and runner
    # Note: plan_json_str is preserved exactly as read from PLAN_PATH (with original indent and newline)
    cell2_code = [
        "# Cell 2: Staging kế hoạch máy tiền đăng ký và runner script\n",
        "import json\n",
        "import hashlib\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "if hasattr(sys.stdout, 'reconfigure'):\n",
        "    try:\n",
        "        sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        f"EXPECTED_PLAN_SHA = \"{plan_sha}\"\n",
        f"EXPECTED_RUNNER_SHA = \"{runner_sha}\"\n",
        "\n",
        f"PLAN_JSON = {json.dumps(plan_json_str)}\n",
        f"RUNNER_SOURCE = {json.dumps(runner_source)}\n",
        "\n",
        "plan_file = Path('document_feasibility_plan.json')\n",
        "with open(plan_file, 'w', encoding='utf-8', newline='\\n') as f:\n",
        "    f.write(PLAN_JSON)\n",
        "\n",
        "runner_file = Path('run_document_feasibility_colab.py')\n",
        "with open(runner_file, 'w', encoding='utf-8', newline='\\n') as f:\n",
        "    f.write(RUNNER_SOURCE)\n",
        "\n",
        "def compute_sha(p: Path) -> str:\n",
        "    h = hashlib.sha256()\n",
        "    with open(p, 'rb') as f:\n",
        "        while chunk := f.read(65536): h.update(chunk)\n",
        "    return h.hexdigest()\n",
        "\n",
        "actual_plan_sha = compute_sha(plan_file)\n",
        "actual_runner_sha = compute_sha(runner_file)\n",
        "\n",
        "assert actual_plan_sha == EXPECTED_PLAN_SHA, f'Plan SHA mismatch: {actual_plan_sha} != {EXPECTED_PLAN_SHA}'\n",
        "assert actual_runner_sha == EXPECTED_RUNNER_SHA, f'Runner SHA mismatch: {actual_runner_sha} != {EXPECTED_RUNNER_SHA}'\n",
        "\n",
        "plan_data = json.loads(PLAN_JSON)\n",
        "print(f\"[OK] Đã ghi và kiểm toán kế hoạch tiền đăng ký (SHA-256: {actual_plan_sha[:12]}...)\")\n",
        "print(f\"  - Nghiên cứu: {plan_data['metadata']['study_name']}\")\n",
        "print(f\"  - Protocol: v{plan_data['metadata']['protocol_version']}\")\n",
        "print(f\"  - Model: {plan_data['metadata']['model']['repository']} @ {plan_data['metadata']['model']['revision'][:8]}\")\n",
        "print(f\"  - Attempts: {len(plan_data['attempts'])} ({[a['attempt_id'] for a in plan_data['attempts']]})\")\n",
        "print(f\"  - Seeds: {[a['seed'] for a in plan_data['attempts']]}\")\n",
        "print(f\"[OK] Đã ghi và kiểm toán runner script (SHA-256: {actual_runner_sha[:12]}...)\")\n",
    ]
    cell2_str = "".join(cell2_code)
    validate_python_code(cell2_str, "Cell 2")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell2_code})

    # Cell 3: Execute generation runner
    cell3_code = [
        "# Cell 3: Thực thi chạy 4 attempts tạo dữ liệu khả thi\n",
        "!python run_document_feasibility_colab.py --plan document_feasibility_plan.json --output-dir ./results\n",
    ]
    cell3_str = "".join(cell3_code)
    validate_python_code(cell3_str, "Cell 3")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell3_code})

    # Cell 4: Download results archive
    cell4_code = [
        "# Cell 4: Kiểm tra kết quả và tải file archive về LOCAL\n",
        "from pathlib import Path\n",
        "try:\n",
        "    from google.colab import files\n",
        "    is_colab = True\n",
        "except ImportError:\n",
        "    is_colab = False\n",
        "\n",
        "zip_path = Path('./results/document_feasibility_results.zip')\n",
        "contact_sheet = Path('./results/feasibility_contact_sheet.png')\n",
        "review_csv = Path('./results/human_review.csv')\n",
        "\n",
        "if zip_path.exists():\n",
        "    size_mb = zip_path.stat().st_size / (1024**2)\n",
        "    print(f\"[OK] Archive hoàn tất: {zip_path.name} ({size_mb:.2f} MB)\")\n",
        "    if contact_sheet.exists():\n",
        "        print(f\"[OK] Contact sheet sẵn sàng: {contact_sheet.name}\")\n",
        "    if review_csv.exists():\n",
        "        print(f\"[OK] Human review CSV sẵn sàng: {review_csv.name}\")\n",
        "    if is_colab:\n",
        "        print(\"Đang tải archive về máy LOCAL...\")\n",
        "        files.download(str(zip_path))\n",
        "    else:\n",
        "        print(f\"Môi trường ngoài Colab: file lưu tại {zip_path.resolve()}\")\n",
        "    print(\"\\nSau khi tải xong, hãy đặt file vào thư mục dự án và chạy:\")\n",
        "    print(\"  python scripts/research/verify_document_feasibility_artifacts.py --zip document_feasibility_results.zip\")\n",
        "else:\n",
        "    print(\"LỖI: Không tìm thấy file archive kết quả! Hãy kiểm tra log Cell 3.\")\n",
    ]
    cell4_str = "".join(cell4_code)
    validate_python_code(cell4_str, "Cell 4")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell4_code})

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "independent_document_feasibility_colab.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    return notebook


def main():
    notebook = build_notebook()
    NOTEBOOK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Successfully generated validated turnkey notebook: {NOTEBOOK_OUTPUT_PATH}")
    print(f"Number of cells: {len(notebook['cells'])}")


if __name__ == "__main__":
    main()
