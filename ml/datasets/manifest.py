import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional

LabelType = Literal["authentic", "fully_generated", "ai_edited"]
SplitType = Literal["train", "val", "test_indomain", "test_unseen_generator", "test_degraded", "test_localization", "test_hard_negatives"]

@dataclass
class DatasetRecord:
    sample_id: str
    source_id: str
    image_path: str
    label: LabelType
    generator: str
    generator_version: str
    edit_type: str
    mask_path: str
    dataset_name: str
    dataset_version: str
    split: SplitType
    license: str
    width: int
    height: int
    sha256: str

    @classmethod
    def compute_sha256(cls, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

class ManifestManager:
    FIELDNAMES = [
        "sample_id",
        "source_id",
        "image_path",
        "label",
        "generator",
        "generator_version",
        "edit_type",
        "mask_path",
        "dataset_name",
        "dataset_version",
        "split",
        "license",
        "width",
        "height",
        "sha256",
    ]

    @classmethod
    def save_csv(cls, records: List[DatasetRecord], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cls.FIELDNAMES)
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))

    @classmethod
    def load_csv(cls, input_path: Path) -> List[DatasetRecord]:
        records = []
        with open(input_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    DatasetRecord(
                        sample_id=row["sample_id"],
                        source_id=row["source_id"],
                        image_path=row["image_path"],
                        label=row["label"],  # type: ignore
                        generator=row["generator"],
                        generator_version=row["generator_version"],
                        edit_type=row["edit_type"],
                        mask_path=row["mask_path"],
                        dataset_name=row["dataset_name"],
                        dataset_version=row["dataset_version"],
                        split=row["split"],  # type: ignore
                        license=row["license"],
                        width=int(row["width"]),
                        height=int(row["height"]),
                        sha256=row["sha256"],
                    )
                )
        return records
