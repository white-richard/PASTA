from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import struct
from collections import defaultdict

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration

from datasets import VideoDataset

_ST_DTYPE = {
    "F64": (torch.float64, 8),
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "BOOL": (torch.bool, 1),
}


def _read_safetensors_header(path: pathlib.Path) -> tuple[dict, int]:
    """Return (header_dict, data_section_offset). Uses a small read only."""
    with open(path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_size))
    return header, 8 + header_size


def _read_tensor(f, data_offset: int, meta: dict) -> torch.Tensor:
    dtype, _itemsize = _ST_DTYPE[meta["dtype"]]
    start, end = meta["data_offsets"]
    nbytes = end - start
    f.seek(data_offset + start)
    buf = bytearray(f.read(nbytes))
    if len(buf) != nbytes:
        msg = f"Short read: got {len(buf)} of {nbytes} bytes"
        raise RuntimeError(msg)
    # Read as uint8 then bit-cast to the target dtype (handles bf16 safely).
    u8 = torch.frombuffer(buf, dtype=torch.uint8)
    return u8.view(dtype).reshape(meta["shape"]).clone()


def load_vision_tower(hf_model_id: str, device: torch.device) -> torch.nn.Module:
    """Instantiate the full model on `meta`, then selectively materialise only
    the vision tower's weights on `device`. Avoids mmap'ing the ~50GB LLM
    weight shard (which may not fit in VMA space).
    """
    config = AutoConfig.from_pretrained(hf_model_id)
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration(config)
    vision_tower = model.model.vision_tower
    del model

    repo_dir = snapshot_download(
        hf_model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
    )
    index_path = pathlib.Path(repo_dir) / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        # Single-file checkpoint: parse header of the lone safetensors file.
        (only_file,) = list(pathlib.Path(repo_dir).glob("*.safetensors"))
        header, _ = _read_safetensors_header(only_file)
        weight_map = {k: only_file.name for k in header if k != "__metadata__"}

    prefix = "model.vision_tower."
    wanted = [k for k in weight_map if k.startswith(prefix)]
    if not wanted:
        msg = f"No keys with prefix {prefix!r} in {hf_model_id}"
        raise RuntimeError(msg)

    by_shard: dict[str, list[str]] = defaultdict(list)
    for k in wanted:
        by_shard[weight_map[k]].append(k)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_file, keys in by_shard.items():
        path = pathlib.Path(repo_dir) / shard_file
        header, data_offset = _read_safetensors_header(path)
        print(f"  reading {len(keys)} tensors from {shard_file}")
        with open(path, "rb") as f:
            for k in keys:
                t = _read_tensor(f, data_offset, header[k])
                state_dict[k[len(prefix) :]] = t.to(torch.bfloat16)

    vision_tower.to_empty(device=device)
    missing, unexpected = vision_tower.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:3]}... ({len(unexpected)} total)")
    if missing:
        print(f"[warn] missing keys: {missing[:3]}... ({len(missing)} total)")
    return vision_tower.eval()


class FrameDataset(Dataset):
    """Workers do full image preprocessing so the GPU never waits on CPU."""

    def __init__(self, entries, image_processor, max_soft_tokens: int = 70) -> None:
        self.entries = entries
        self.image_processor = image_processor
        self.max_soft_tokens = max_soft_tokens

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i):
        vid, idx, path = self.entries[i]
        img = Image.open(path).convert("RGB")
        kwargs: dict = {"images": img, "return_tensors": "pt"}
        if self.max_soft_tokens:
            kwargs["max_soft_tokens"] = self.max_soft_tokens
        out = self.image_processor(**kwargs)
        pv = out["pixel_values"][0]  # (patches, patch_pixels) or (C, H, W)
        pos = out["image_position_ids"][0] if "image_position_ids" in out else None
        return pv, pos, vid, idx


def collate(batch):
    pvs = torch.stack([b[0] for b in batch])
    # pos_ids may be None for non-Gemma4 processors
    pos = torch.stack([b[1] for b in batch]) if batch[0][1] is not None else None
    vids = [b[2] for b in batch]
    idxs = [b[3] for b in batch]
    return pvs, pos, vids, idxs


def shard_dir(save_path: pathlib.Path, split: str, shard_id: int, num_shards: int) -> pathlib.Path:
    return save_path / f"_shard{shard_id}of{num_shards}_{split}"


def final_dir(save_path: pathlib.Path, split: str) -> pathlib.Path:
    return save_path / f"features_{split}"


def _save_per_video(
    out_dir: pathlib.Path,
    frames: dict[str, dict[int, torch.Tensor]],
    d_vit: int,
    feature_mode: str,
) -> None:
    """Write a {vid: {frame_idx: tensor}} dict to per-video .pt files in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vids = []
    for vid, idx_map in frames.items():
        order = sorted(idx_map.keys())
        vis = torch.stack([idx_map[k] for k in order])
        entry: dict = {"vis": vis}
        if vis.dim() == 3:  # patch features (T, P, D) — save spatial mean as global repr
            entry["vis_global"] = vis.mean(dim=1)
        torch.save(entry, out_dir / f"{vid}.pt")
        vids.append(vid)
    torch.save(
        {"d_vit": d_vit, "feature_mode": feature_mode, "vids": sorted(vids)},
        out_dir / "_meta.pt",
    )


def stream_merge_to_dir(
    checkpoint_files: list[pathlib.Path],
    out_dir: pathlib.Path,
    d_vit: int,
    feature_mode: str,
) -> None:
    """Stream-merge checkpoint .pt files into per-video files without full-RAM accumulation.

    Each checkpoint contains {vid: {frame_idx: tensor}}.  Because entries are sorted by
    (vid, frame_idx) before batching, at most 1 video per checkpoint straddles a boundary;
    all others are fully contained in a single checkpoint.  The partial-file read-modify-write
    is therefore almost always a no-op (O(1) boundary videos per checkpoint).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / "_partial"
    partial_dir.mkdir(exist_ok=True)

    print(f"  streaming merge of {len(checkpoint_files)} checkpoints → {out_dir}")
    for ckpt in tqdm(checkpoint_files, desc="stream merge"):
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        for vid, idx_map in data["frames"].items():
            vpath = partial_dir / f"{vid}.pt"
            if vpath.exists():
                existing: dict[int, torch.Tensor] = torch.load(
                    vpath,
                    map_location="cpu",
                    weights_only=False,
                )
                existing.update(idx_map)
                torch.save(existing, vpath)
            else:
                torch.save(dict(idx_map), vpath)
        del data
        ckpt.unlink()

    vids: list[str] = []
    print("  stacking per-video frames…")
    for vpath in tqdm(sorted(partial_dir.iterdir()), desc="finalize"):
        vid = vpath.stem
        idx_map = torch.load(vpath, map_location="cpu", weights_only=False)
        order = sorted(idx_map.keys())
        vis = torch.stack([idx_map[k] for k in order])
        entry: dict = {"vis": vis}
        if vis.dim() == 3:  # patch features (T, P, D) — save spatial mean as global repr
            entry["vis_global"] = vis.mean(dim=1)
        torch.save(entry, out_dir / f"{vid}.pt")
        vpath.unlink()
        vids.append(vid)

    partial_dir.rmdir()
    torch.save(
        {"d_vit": d_vit, "feature_mode": feature_mode, "vids": sorted(vids)},
        out_dir / "_meta.pt",
    )
    print(f"  wrote {len(vids)} per-video files")


def run_extract(args) -> None:
    save_path = pathlib.Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading vision tower from {args.hf_model_id}...")
    processor = AutoProcessor.from_pretrained(args.hf_model_id)
    vision_tower = load_vision_tower(args.hf_model_id, device)
    torch.cuda.empty_cache()

    dataset = VideoDataset(vars(args), args.split)
    print("Collecting frame paths...")
    entries = []
    for i in range(len(dataset)):
        vid, frames = dataset[i]
        for idx, path in enumerate(frames):
            entries.append((vid, idx, path))
    entries.sort()

    entries = entries[args.shard_id :: args.num_shards]
    if args.debug:
        entries = entries[: 4 * args.batch_size]
    print(f"[shard {args.shard_id}/{args.num_shards}] frames: {len(entries)}")

    image_processor = processor.image_processor
    for attr in ("do_image_splitting", "do_pan_and_scan"):
        if hasattr(image_processor, attr):
            setattr(image_processor, attr, False)

    fd = FrameDataset(entries, image_processor, max_soft_tokens=args.max_soft_tokens)
    loader = DataLoader(
        fd,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    # {vid: {frame_idx: tensor}} — shape (D,) for gap, (P, D) for patches.
    # Flushed to disk every --checkpoint-every batches to avoid OOM when
    # storing patch-level features (e.g. all_patches mode with 63 patches/frame
    # × 827k train frames ≈ 120 GB if kept fully in RAM).
    frame_feats: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    d_vit: int | None = None
    checkpoint_files: list[pathlib.Path] = []

    def _flush(batch_idx: int) -> None:
        if not frame_feats:
            return
        ckpt = (
            save_path / f"_shard{args.shard_id}of{args.num_shards}_{args.split}_ckpt{batch_idx}.pt"
        )
        torch.save(
            {"d_vit": d_vit, "feature_mode": args.feature_mode, "frames": dict(frame_feats)},
            ckpt,
        )
        checkpoint_files.append(ckpt)
        frame_feats.clear()
        print(f"  [flush] {ckpt.name}")

    with torch.inference_mode():
        for batch_idx, (pvs, pos_ids, vids, idxs) in enumerate(
            tqdm(loader, desc=f"shard {args.shard_id}"),
        ):
            pv = pvs.to(device, dtype=torch.bfloat16, non_blocking=True)
            if pos_ids is not None:
                pos_ids = pos_ids.to(device, non_blocking=True)
            out = vision_tower(pv, pixel_position_ids=pos_ids)
            hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            # Gemma4VisionModel strips padding and flattens to (total_valid_patches, D).
            if hs.dim() == 2:
                c = pvs.shape[0]
                hs = hs.view(c, hs.shape[0] // c, hs.shape[-1])
            # hs: (B, P, D)
            if args.feature_mode == "gap":
                feat = hs.mean(dim=1).bfloat16().cpu()  # (B, D)
            elif args.feature_mode == "all_patches":
                feat = hs.bfloat16().cpu()  # (B, P, D)
            else:  # subsample
                P = hs.shape[1]
                n = min(args.n_patches, P)
                idx = torch.linspace(0, P - 1, n, device=hs.device).long()
                feat = hs[:, idx, :].bfloat16().cpu()  # (B, n_patches, D)
            if d_vit is None:
                d_vit = feat.shape[-1]
            for f, vid, frame_idx in zip(feat, vids, idxs, strict=True):
                frame_feats[vid][frame_idx] = f

            if args.checkpoint_every and (batch_idx + 1) % args.checkpoint_every == 0:
                _flush(batch_idx + 1)

    # Final flush of any remaining frames.
    _flush(len(loader))

    out_dir = shard_dir(save_path, args.split, args.shard_id, args.num_shards)
    if checkpoint_files:
        stream_merge_to_dir(checkpoint_files, out_dir, d_vit, args.feature_mode)
    else:
        _save_per_video(out_dir, dict(frame_feats), d_vit, args.feature_mode)

    print(f"[shard {args.shard_id}] saved → {out_dir}")


def run_merge(args) -> None:
    save_path = pathlib.Path(args.save_path)
    out_dir = final_dir(save_path, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    d_vit: int | None = None
    feature_mode: str | None = None
    all_vids: list[str] = []
    missing = []

    for k in range(args.num_shards):
        src = shard_dir(save_path, args.split, k, args.num_shards)
        if not src.exists():
            missing.append(str(src))
            continue
        print(f"Merging shard {src.name} …")
        meta = torch.load(src / "_meta.pt", map_location="cpu", weights_only=False)
        if d_vit is None:
            d_vit = meta["d_vit"]
        if feature_mode is None:
            feature_mode = meta.get("feature_mode", "gap")
        for vid in meta["vids"]:
            shutil.move(str(src / f"{vid}.pt"), str(out_dir / f"{vid}.pt"))
            all_vids.append(vid)
        shutil.rmtree(str(src))

    if missing:
        msg = "Missing shard directories:\n  " + "\n  ".join(missing)
        raise FileNotFoundError(msg)

    torch.save(
        {"d_vit": d_vit, "feature_mode": feature_mode, "vids": sorted(all_vids)},
        out_dir / "_meta.pt",
    )
    print(
        f"Saved → {out_dir}  (videos={len(all_vids)}, D_vit={d_vit}, feature_mode={feature_mode})",
    )


def run_finalize_checkpoints(args) -> None:
    """Stream-merge checkpoint files left behind after an OOM-killed extraction run."""
    save_path = pathlib.Path(args.save_path)
    pattern = f"_shard{args.shard_id}of{args.num_shards}_{args.split}_ckpt*.pt"
    ckpts = sorted(save_path.glob(pattern))
    if not ckpts:
        print(f"No checkpoint files found matching {pattern!r} in {save_path}")
        return
    print(f"Found {len(ckpts)} checkpoint files:")
    for c in ckpts:
        print(f"  {c.name}  ({c.stat().st_size / 1e9:.1f} GB)")

    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    d_vit = first["d_vit"]
    feature_mode = first.get("feature_mode", args.feature_mode)
    del first

    out_dir = shard_dir(save_path, args.split, args.shard_id, args.num_shards)
    stream_merge_to_dir(ckpts, out_dir, d_vit, feature_mode)
    print(f"[shard {args.shard_id}] merged → {out_dir}")
    print("Run with --merge to combine shards into the final features directory.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--img_path",
        default="datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--hf-model-id", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save_path", default="out/gmmlp_features/")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument(
        "--merge",
        action="store_true",
        help="stitch shard files into features_{split}.pt",
    )
    p.add_argument(
        "--feature-mode",
        choices=["gap", "all_patches", "subsample"],
        default="gap",
        help="gap: mean-pool over patches (D,); all_patches: keep every patch (P, D); subsample: spatially-uniform subsample (n_patches, D)",
    )
    p.add_argument(
        "--n-patches",
        type=int,
        default=64,
        help="number of patches to keep when --feature-mode=subsample",
    )
    p.add_argument(
        "--max-soft-tokens",
        type=int,
        default=70,
        choices=[0, 70, 140, 280, 560, 1120],
        help=(
            "Gemma4 image processor max_soft_tokens. Controls input patch resolution: "
            "70→63 ViT tokens (7×9), 140→~126, 280→~252. Use 0 for non-Gemma4 processors."
        ),
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help=(
            "Flush frame_feats to a temp checkpoint file every N batches and clear RAM. "
            "Checkpoints are merged into the shard file at the end. "
            "Prevents OOM when storing patch-level features over large datasets. "
            "Set to 0 to disable (original behaviour, keeps everything in RAM)."
        ),
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--finalize-checkpoints",
        action="store_true",
        help=(
            "Stream-merge checkpoint .pt files left behind by an OOM-killed extraction run "
            "into a per-video shard directory.  Run --merge afterwards to combine shards."
        ),
    )
    args = p.parse_args()

    if args.merge:
        run_merge(args)
    elif args.finalize_checkpoints:
        run_finalize_checkpoints(args)
    else:
        assert 0 <= args.shard_id < args.num_shards
        run_extract(args)


if __name__ == "__main__":
    main()
