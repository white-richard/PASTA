"""Helpers for RWTH-PHOENIX-Weather 2014T metadata."""

from __future__ import annotations

import csv
from pathlib import Path


def infer_phoenix_split(path: str | Path) -> str:
    """Infer train/dev/test from a Phoenix annotation filename."""
    name = Path(path).name.lower()
    for split in ("train", "dev", "test"):
        if split in name:
            return split
    raise ValueError(f"Could not infer Phoenix split from filename: {path}")


def load_phoenix_csv(path: str | Path) -> dict[str, dict[str, object]]:
    """Load a Phoenix pipe-delimited corpus file into the project's sample format.

    The returned keys are <split>/<video_name> and each value contains the
    annotation name, German translation, and frame paths relative to the Phoenix
    image root. Rows whose frame directory contains no PNG files are skipped.
    """
    path = Path(path)
    split = infer_phoenix_split(path)
    img_root = (path.resolve().parent / ".." / ".." / "features" / "fullFrame-210x260px").resolve()

    data: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        required = {"name", "video", "translation"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Phoenix annotation file is missing columns: {sorted(missing)}")

        for row in reader:
            annotation_name = row["name"].strip()
            translation = row["translation"].strip()
            video_name = row["video"].strip().split("/")[0]
            frame_dir = img_root / split / video_name
            frames = sorted(frame_dir.glob("*.png"))
            if not frames:
                continue
            rel_frames = ["/" + frame.relative_to(img_root).as_posix() for frame in frames]
            data[f"{split}/{video_name}"] = {
                "name": annotation_name,
                "text": translation,
                "imgs_path": rel_frames,
            }
    return data


def load_translations(path: str | Path) -> dict[str, str]:
    """Return annotation_name -> German translation for a Phoenix split."""
    result: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for row in reader:
            result[row["name"].strip()] = row["translation"].strip()
    return result
