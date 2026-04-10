import pathlib

import torch


def load_descript_features(
    features_dir: str | pathlib.Path,
    splits: list[str] | None = None,
    hidden_layer: int = 20,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load LLaVA hidden-state features extracted by generate_descript.py.

    Args:
        features_dir: Directory containing ``phoenix_hidden_layer{N}_{split}.pt`` files.
        splits: Which splits to load. Defaults to all three: train, dev, test.
        hidden_layer: Layer index used during extraction (default 20).
        device: Device to map tensors onto.

    Returns:
        Dict mapping video name to a ``(num_frames, feature_dim)`` tensor.

    """
    if splits is None:
        splits = ["train", "dev", "test"]

    features_dir = pathlib.Path(features_dir)
    combined: dict[str, torch.Tensor] = {}

    for split in splits:
        path = features_dir / f"phoenix_hidden_layer{hidden_layer}_{split}.pt"
        data: dict = torch.load(path, map_location=device, weights_only=False)
        for vid_name, entry in data.items():
            combined[vid_name] = torch.stack(entry["features"])  # (T, D)

    return combined


if __name__ == "__main__":
    feats = load_descript_features("out/descript/features/")
    sample_vid, sample_tensor = next(iter(feats.items()))
    print(f"Videos loaded : {len(feats)}")
    print(f"Feature dim   : {sample_tensor.shape[-1]}")
    print(
        f"Frame counts  : min={min(v.shape[0] for v in feats.values())}  "
        f"max={max(v.shape[0] for v in feats.values())}  "
        f"mean={sum(v.shape[0] for v in feats.values()) / len(feats):.1f}"
    )
    print(f"Example       : '{sample_vid}'  shape={tuple(sample_tensor.shape)}")
