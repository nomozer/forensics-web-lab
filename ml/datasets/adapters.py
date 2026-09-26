from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from .manifest import DatasetRecord

class BaseDatasetAdapter(ABC):
    """Abstract base adapter for normalizing external datasets into unified manifest records."""

    @abstractmethod
    def get_dataset_name(self) -> str:
        pass

    @abstractmethod
    def get_license(self) -> str:
        pass

    @abstractmethod
    def parse_records(self, root_dir: Path) -> List[DatasetRecord]:
        pass

class GenImageAdapter(BaseDatasetAdapter):
    def get_dataset_name(self) -> str:
        return "GenImage"

    def get_license(self) -> str:
        return "CC-BY-NC 4.0"

    def parse_records(self, root_dir: Path) -> List[DatasetRecord]:
        records: List[DatasetRecord] = []
        if not root_dir.exists():
            return records

        for img_file in root_dir.glob("**/*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            rel_str = str(img_file.relative_to(root_dir))
            is_ai = "ai" in rel_str.lower() or "synthetic" in rel_str.lower()
            generator = "stable_diffusion" if "sd" in rel_str.lower() else "unknown_generator"

            records.append(
                DatasetRecord(
                    sample_id=f"genimage_{img_file.stem}",
                    source_id=img_file.stem.split("_")[0],
                    image_path=str(img_file),
                    label="fully_generated" if is_ai else "authentic",
                    generator=generator if is_ai else "camera",
                    generator_version="1.5" if is_ai else "none",
                    edit_type="full_synthesis" if is_ai else "none",
                    mask_path="",
                    dataset_name=self.get_dataset_name(),
                    dataset_version="1.0",
                    split="train",
                    license=self.get_license(),
                    width=256,
                    height=256,
                    sha256="",
                )
            )
        return records

class SagiDAdapter(BaseDatasetAdapter):
    def get_dataset_name(self) -> str:
        return "SAGI-D"

    def get_license(self) -> str:
        return "CC-BY 4.0"

    def parse_records(self, root_dir: Path) -> List[DatasetRecord]:
        records: List[DatasetRecord] = []
        if not root_dir.exists():
            return records

        for img_file in (root_dir / "images").glob("**/*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            mask_candidate = root_dir / "masks" / f"{img_file.stem}_mask.png"
            has_mask = mask_candidate.exists()

            records.append(
                DatasetRecord(
                    sample_id=f"sagid_{img_file.stem}",
                    source_id=img_file.stem.split("_")[0],
                    image_path=str(img_file),
                    label="ai_edited" if has_mask else "authentic",
                    generator="sd_inpaint" if has_mask else "camera",
                    generator_version="2.0" if has_mask else "none",
                    edit_type="inpainting" if has_mask else "none",
                    mask_path=str(mask_candidate) if has_mask else "",
                    dataset_name=self.get_dataset_name(),
                    dataset_version="1.0",
                    split="train",
                    license=self.get_license(),
                    width=512,
                    height=512,
                    sha256="",
                )
            )
        return records
