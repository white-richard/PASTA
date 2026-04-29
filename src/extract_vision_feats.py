from __future__ import annotations

import argparse
import json
import pathlib
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

    def __init__(self, entries, image_processor) -> None:
        self.entries = entries
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i):
        vid, idx, path = self.entries[i]
        img = Image.open(path).convert("RGB")
        out = self.image_processor(images=img, return_tensors="pt", max_soft_tokens=70)
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


def shard_path(save_path: pathlib.Path, split: str, shard_id: int, num_shards: int) -> pathlib.Path:
    return save_path / f"_shard{shard_id}of{num_shards}_{split}.pt"


def final_path(save_path: pathlib.Path, split: str) -> pathlib.Path:
    return save_path / f"features_{split}.pt"


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

    fd = FrameDataset(entries, image_processor)
    loader = DataLoader(
        fd,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    # {vid: {frame_idx: tensor}} — shape (D,) for gap, (P, D) for patches.
    frame_feats: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    d_vit: int | None = None

    with torch.inference_mode():
        for pvs, pos_ids, vids, idxs in tqdm(loader, desc=f"shard {args.shard_id}"):
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

    out_file = shard_path(pathlib.Path(args.save_path), args.split, args.shard_id, args.num_shards)
    torch.save(
        {"d_vit": d_vit, "feature_mode": args.feature_mode, "frames": dict(frame_feats)},
        out_file,
    )
    print(f"[shard {args.shard_id}] saved → {out_file}")


def run_merge(args) -> None:
    save_path = pathlib.Path(args.save_path)
    merged: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    d_vit: int | None = None
    feature_mode: str | None = None
    missing = []

    for k in range(args.num_shards):
        p = shard_path(save_path, args.split, k, args.num_shards)
        if not p.exists():
            missing.append(str(p))
            continue
        print(f"Loading {p}...")
        data = torch.load(p, map_location="cpu", weights_only=False)
        if d_vit is None:
            d_vit = data["d_vit"]
        if feature_mode is None:
            feature_mode = data.get("feature_mode", "gap")
        for vid, idx_map in data["frames"].items():
            merged[vid].update(idx_map)

    if missing:
        msg = "Missing shard files:\n  " + "\n  ".join(missing)
        raise FileNotFoundError(msg)

    print(f"Stacking {len(merged)} videos...")
    out: dict = {"_meta": {"d_vit": d_vit, "feature_mode": feature_mode}}
    for vid, idx_map in merged.items():
        order = sorted(idx_map.keys())
        vis = torch.stack([idx_map[i] for i in order], dim=0)  # (T, D) or (T, P, D)
        out[vid] = {"vis": vis}

    final = final_path(save_path, args.split)
    torch.save(out, final)
    print(f"Saved → {final}  (videos={len(merged)}, D_vit={d_vit}, feature_mode={feature_mode})")


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
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.merge:
        run_merge(args)
    else:
        assert 0 <= args.shard_id < args.num_shards
        run_extract(args)


if __name__ == "__main__":
    main()
