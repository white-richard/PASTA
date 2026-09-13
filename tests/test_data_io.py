from pathlib import Path

import pytest

from pasta.data_io import infer_phoenix_split, load_phoenix_csv, load_translations


def _write_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "PHOENIX-2014-T"
    annotations = root / "annotations" / "manual"
    frames = root / "features" / "fullFrame-210x260px" / "train" / "video-1"
    annotations.mkdir(parents=True)
    frames.mkdir(parents=True)
    (frames / "0001.png").write_bytes(b"fixture")
    csv_path = annotations / "PHOENIX-2014-T.train.corpus.csv"
    csv_path.write_text(
        "name|video|start|end|speaker|orth|translation\n"
        "sample-1|video-1/1/*.png|0|1|speaker|GLOSS|guten morgen\n",
        encoding="utf-8",
    )
    return csv_path


def test_infer_phoenix_split() -> None:
    assert infer_phoenix_split("PHOENIX-2014-T.dev.corpus.csv") == "dev"
    with pytest.raises(ValueError):
        infer_phoenix_split("annotations.csv")


def test_load_phoenix_csv(tmp_path: Path) -> None:
    csv_path = _write_fixture(tmp_path)
    data = load_phoenix_csv(csv_path)
    assert list(data) == ["train/video-1"]
    assert data["train/video-1"]["text"] == "guten morgen"
    assert data["train/video-1"]["imgs_path"] == ["/train/video-1/0001.png"]


def test_load_translations(tmp_path: Path) -> None:
    csv_path = _write_fixture(tmp_path)
    assert load_translations(csv_path) == {"sample-1": "guten morgen"}
