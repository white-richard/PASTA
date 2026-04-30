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
    for h in range(1, int(math.isqrt(n)) + 1):
        if n % h == 0:
            best = (h, n // h)
    return best


def spatial_pool(vis: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Pool (T, P, D) patch tensor to (T, H_out*W_out, D).

    P must be a perfect square (H_in = W_in = sqrt(P)).
    n_tokens is factorised into the nearest-to-square H_out × W_out.
    """
    T, P, D = vis.shape
    H_in = int(math.isqrt(P))
    if H_in * H_in != P:
        msg = f"Patch count {P} is not a perfect square; cannot form spatial grid."
        raise ValueError(msg)

    H_out, W_out = _best_factors(n_tokens)
    kH, kW = H_in // H_out, H_in // W_out
    if kH < 1 or kW < 1:
        msg = f"n_tokens={n_tokens} ({H_out}×{W_out}) is larger than input grid {H_in}×{H_in}."
        raise ValueError(msg)

    # (T, P, D) → (T, D, H_in, H_in) for avg_pool2d
    x = vis.permute(0, 2, 1).reshape(T, D, H_in, H_in).float()
    x = F.avg_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))  # (T, D, H_out, W_out)

    actual = x.shape[2] * x.shape[3]
    x = x.reshape(T, D, actual).permute(0, 2, 1)  # (T, actual, D)
    return x.to(vis.dtype)


def pool_file(input_path: pathlib.Path, n_tokens: int) -> None:
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
        help="One or more features_*.pt files to pool.",
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
