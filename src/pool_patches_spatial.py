"""Spatially pool patch-level vision features (T, P, D) → (T, H_out*W_out, D).

Treats the P patches as a sqrt(P) × sqrt(P) spatial grid and applies 2-D
average pooling.  The target token count is factorised into the nearest
H_out × W_out rectangle (e.g. 64 → 8×8, 128 → 8×16).  Operates on merged
feature files produced by extract_vision_feats.py (feature_mode all_patches
or subsample).  Gap-mode files are skipped with a warning.

Usage
-----
# single token count
python src/pool_patches_spatial.py \
    --input datasets/phoenix-vision_feats/A4B_features/features_train.pt \
    --n-tokens 64

# multiple token counts in one pass
python src/pool_patches_spatial.py \
    --input datasets/phoenix-vision_feats/A4B_features/features_train.pt \
    --n-tokens 64 128
"""

from __future__ import annotations

import argparse
import math
import pathlib

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _best_factors(n: int) -> tuple[int, int]:
    """Return (H, W) with H <= W, H*W == n, minimising W - H (nearest to square)."""
    best = (1, n)
    for h in range(1, math.isqrt(n) + 1):
        if n % h == 0:
            best = (h, n // h)
    return best


def spatial_pool(vis: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Pool (T, P, D) patch tensor to (T, H_out*W_out, D).

    P is factorised into the nearest-to-square H_in × W_in grid (handles
    non-square patch counts like 63 = 7×9).  n_tokens is similarly factorised
    into H_out × W_out.  Uses adaptive_avg_pool2d so fractional strides are
    handled correctly.
    """
    T, P, D = vis.shape
    H_in, W_in = _best_factors(P)

    H_out, W_out = _best_factors(n_tokens)
    if H_out > H_in or W_out > W_in:
        msg = (
            f"n_tokens={n_tokens} ({H_out}×{W_out}) exceeds input grid {H_in}×{W_in} "
            f"(P={P}). Choose a smaller n_tokens."
        )
        raise ValueError(msg)

    # (T, P, D) → (T, D, H_in, W_in) for pooling
    x = vis.permute(0, 2, 1).reshape(T, D, H_in, W_in).float()
    x = F.adaptive_avg_pool2d(x, (H_out, W_out))  # (T, D, H_out, W_out)

    actual = x.shape[2] * x.shape[3]
    x = x.reshape(T, D, actual).permute(0, 2, 1)  # (T, actual, D)
    return x.to(vis.dtype)


def pool_file(input_path: pathlib.Path, n_tokens: int) -> None:
    if input_path.is_dir():
        _pool_dir(input_path, n_tokens)
    else:
        _pool_single_file(input_path, n_tokens)


def _pool_dir(src_dir: pathlib.Path, n_tokens: int) -> None:
    """Pool all per-video files in a per-video directory (output of extract_vision_feats.py)."""
    meta = torch.load(src_dir / "_meta.pt", map_location="cpu", weights_only=False)
    feature_mode = meta.get("feature_mode", "gap")
    if feature_mode == "gap":
        print(f"  [skip] feature_mode=gap — no patch dimension to pool in {src_dir.name}")
        return

    actual_tokens: int | None = None
    out_dir: pathlib.Path | None = None

    for vid in tqdm(meta["vids"], desc=f"{n_tokens}tok"):
        data = torch.load(src_dir / f"{vid}.pt", map_location="cpu", weights_only=False)
        vis = data["vis"]
        if vis.dim() == 2:
            print(f"  [warn] {vid}: 2-D tensor in non-gap dir, skipping")
            continue
        pooled = spatial_pool(vis, n_tokens)
        if actual_tokens is None:
            actual_tokens = pooled.shape[1]
            H_out, W_out = _best_factors(actual_tokens)
            print(f"  grid: {vis.shape[1]} → {actual_tokens} patches ({H_out}×{W_out})")
            out_dir = src_dir.parent / f"{src_dir.name}_spatial{actual_tokens}tok"
            out_dir.mkdir(exist_ok=True)
        torch.save({"vis": pooled}, out_dir / f"{vid}.pt")

    n = actual_tokens if actual_tokens is not None else n_tokens
    if out_dir is None:
        out_dir = src_dir.parent / f"{src_dir.name}_spatial{n}tok"
        out_dir.mkdir(exist_ok=True)
    torch.save({**meta, "feature_mode": f"spatial_{n}tok", "n_tokens": n}, out_dir / "_meta.pt")
    print(f"Saved → {out_dir}  (tokens/frame={n}, D={meta.get('d_vit')})")


def _pool_single_file(input_path: pathlib.Path, n_tokens: int) -> None:
    print(f"Loading {input_path} (target {n_tokens} tokens/frame)...")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    meta = data.get("_meta", {})
    feature_mode = meta.get("feature_mode", "gap")

    if feature_mode == "gap":
        print(f"  [skip] feature_mode=gap — no patch dimension to pool in {input_path.name}")
        return

    actual_tokens: int | None = None
    out: dict = {}

    for vid, payload in tqdm(data.items(), desc=f"{n_tokens}tok"):
        if vid == "_meta":
            continue
        vis = payload["vis"]  # (T, P, D)
        if vis.dim() == 2:
            print(f"  [warn] {vid}: 2-D tensor in non-gap file, copying as-is")
            out[vid] = payload
            continue
        pooled = spatial_pool(vis, n_tokens)
        if actual_tokens is None:
            actual_tokens = pooled.shape[1]
            H_out, W_out = _best_factors(actual_tokens)
            print(f"  grid: {vis.shape[1]} → {actual_tokens} patches ({H_out}×{W_out})")
        out[vid] = {"vis": pooled}

    n = actual_tokens if actual_tokens is not None else n_tokens
    out["_meta"] = {**meta, "feature_mode": f"spatial_{n}tok", "n_tokens": n}

    out_path = input_path.parent / f"{input_path.stem}_spatial{n}tok.pt"
    torch.save(out, out_path)
    print(f"Saved → {out_path}  (tokens/frame={n}, D={meta.get('d_vit')})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more features_* .pt files or per-video directories to pool.",
    )
    p.add_argument(
        "--n-tokens",
        type=int,
        nargs="+",
        default=[64],
        help="Target token counts per frame (space-separated). e.g. --n-tokens 64 128",
    )
    args = p.parse_args()

    for path_str in args.input:
        for n in args.n_tokens:
            pool_file(pathlib.Path(path_str), n)


if __name__ == "__main__":
    main()
