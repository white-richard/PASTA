from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE = "datasets/phoenix-vision_feats_pooled/A4B_features"


def test_config_does_not_require_description_features() -> None:
    config = yaml.safe_load((ROOT / "configs/phoenix2014t.yaml").read_text())
    assert config["name"] == "PASTA"
    assert "descript_feat_path" not in config["data"]


def test_supported_scripts_share_feature_cache_path() -> None:
    for script in ("extract_vision_feats.bash", "train_ppasta.bash", "train_pasta.bash"):
        text = (ROOT / "scripts" / script).read_text()
        assert FEATURE_CACHE in text, script


def test_supported_surface_has_no_stale_gmmlp_reference() -> None:
    paths = list((ROOT / "src/pasta").rglob("*.py")) + list((ROOT / "scripts").glob("*.bash"))
    offenders = [str(path.relative_to(ROOT)) for path in paths if "gmmlp" in path.read_text().lower()]
    assert offenders == []
