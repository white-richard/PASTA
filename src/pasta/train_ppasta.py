import os

os.environ["USE_TF"] = "0"

import argparse
import datetime
import gc
import json
import random
import resource
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import hpargparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from flamingo_pytorch import PerceiverResampler
from grad_cache.context_managers import RandContext
from grad_cache.grad_cache import GradCache
from hpman.m import _
from loguru import logger
from peft import LoraConfig, get_peft_model
from PIL import Image
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor

from . import utils
from .datasets import load_dataset_file
from .grad_cache_util import filip_loss_fn, split_tgt_input


class LazyFeatureDir:
    """Lazy per-video loader that mimics a {vid: {"vis": tensor}} dict interface.

    Backed by a directory of per-video .pt files produced by extract_vision_feats.py
    or pool_patches_spatial.py. Keeps a set of known video IDs for fast membership
    tests; the actual tensor is only read from disk when __getitem__ is called.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        meta = torch.load(self.root / "_meta.pt", map_location="cpu", weights_only=False)
        self._vids: set[str] = set(meta.get("vids", []))
        self.d_vit: int | None = meta.get("d_vit")
        self.feature_mode: str | None = meta.get("feature_mode")

    def __contains__(self, vid: object) -> bool:
        return vid in self._vids

    def __getitem__(self, vid: str) -> dict:
        return torch.load(self.root / f"{vid}.pt", map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        return len(self._vids)


def load_siglip_features(
    path: str | Path,
    representation: str = "tokens",
) -> dict[str, torch.Tensor]:
    """Load precomputed SigLIP2 translation features.

    translation_embed.py stores a leading translation dimension even when a
    Phoenix sample has one reference. tokens removes that dimension and
    returns (L, D) features for FILIP. pooled returns the model's pooled
    text representation as (D,).
    """
    if representation not in {"tokens", "pooled"}:
        raise ValueError("representation must be 'tokens' or 'pooled'")

    data = torch.load(path, weights_only=False)
    out: dict[str, torch.Tensor] = {}
    key = "siglip2_token_feat" if representation == "tokens" else "siglip2_feat"
    for vid, entry in data.items():
        if key not in entry:
            raise KeyError(f"Video {vid} missing {key!r} in {path}.")
        out[vid] = entry[key].float().mean(dim=0)
    return out


class PPASTADataset(Dataset):
    """Loads PIL frames or pre-extracted ViT features + SigLIP text feats.

    When vis_proj_feats is provided (pre-extracted ViT features from
    extract_vision_feats.py), PIL image loading is skipped entirely and the
    stored (T, D_vit) tensors are returned instead.
    """

    def __init__(
        self,
        path,
        config,
        args,
        phase: str,
        siglip_feats: dict[str, torch.Tensor],
        vis_proj_feats: dict[str, dict] | None = None,
    ) -> None:
        self.config = config
        self.args = args
        self.phase = phase
        self.img_path = config["data"]["img_path"]
        self.max_length = config["data"]["max_length"]
        self.vis_proj_feats = (
            vis_proj_feats  # {vid_name: {"vis": (T, D_vit), "vis_global": (T, D_vit)}}
        )
        self.use_global_repr = getattr(args, "use_global_repr", False)
        self.global_feat_key = getattr(args, "global_feat_key", "vis_global")

        self.raw_data = load_dataset_file(path[phase])
        self.siglip_feats = siglip_feats

        required = [self.siglip_feats]
        if vis_proj_feats is not None:
            required.append(vis_proj_feats)

        self.list = [key for key in self.raw_data if all(key.split("/")[1] in d for d in required)]
        dropped = len(self.raw_data) - len(self.list)
        if dropped:
            logger.warning(
                f"[{phase}] Dropped {dropped} videos missing SigLIP/ViT features.",
            )

    def __len__(self) -> int:
        return len(self.list)

    def __getitem__(self, index: int):
        key = self.list[index]
        sample = self.raw_data[key]
        vid_name = key.split("/")[1]

        siglip_feat = self.siglip_feats[vid_name]

        if self.vis_proj_feats is not None:
            entry = self.vis_proj_feats[vid_name]
            if self.use_global_repr:
                vis = entry.get(self.global_feat_key)
                if vis is None:
                    vis = entry["vis"]
                    if vis.dim() == 3:  # (T, P, D) patches → spatial mean → (T, D)
                        vis = vis.mean(dim=1)
            else:
                vis = entry["vis"]  # (T, D_vit) or (T, P, D_vit)
            if len(vis) > self.max_length:
                idxs = sorted(random.sample(range(len(vis)), self.max_length))
                vis = vis[idxs]
            return vid_name, siglip_feat, vis

        pil_frames = self._load_pil_frames(
            [self.img_path + x for x in sample["imgs_path"]],
        )
        return vid_name, siglip_feat, pil_frames

    def _load_pil_frames(self, paths: list[str]) -> list[Image.Image]:
        if len(paths) > self.max_length:
            indices = sorted(random.sample(range(len(paths)), self.max_length))
            paths = [paths[i] for i in indices]
        frames = []
        for p in paths:
            frames.append(Image.open(p).convert("RGB"))
        return frames

    def collate_fn(self, batch):
        if self.vis_proj_feats is not None:
            _names, siglip_feats, vis_list = zip(*batch, strict=False)
            src_input = {
                "vis_feats": list(vis_list),
                "src_length_batch": torch.tensor([len(v) for v in vis_list]),
            }
        else:
            _names, siglip_feats, images = zip(*batch, strict=False)
            src_input = {
                "images": list(images),
                "src_length_batch": torch.tensor([len(imgs) for imgs in images]),
            }
        # siglip_feats: list of (L_i, D) token tensors with variable L, or (D,) gap tensors.
        # Pad variable-length token sequences to the batch max L with zeros.
        feats_list = list(siglip_feats)
        if feats_list[0].dim() == 2:
            max_L = max(f.shape[0] for f in feats_list)
            D = feats_list[0].shape[1]
            padded = torch.zeros(len(feats_list), max_L, D)
            for i, f in enumerate(feats_list):
                padded[i, : f.shape[0]] = f
            siglip_tensor = padded  # (B, max_L, D)
        else:
            siglip_tensor = torch.stack(feats_list)  # (B, D) gap fallback
        tgt_input = {"siglip_feat": siglip_tensor}
        return src_input, tgt_input

    def __str__(self) -> str:
        mode = "pre-extracted" if self.vis_proj_feats is not None else "PIL"
        return f"#total {self.phase} set: {len(self.list)} ({mode})."


def split_ppasta_src_input(model_input: dict, chunk_size: int) -> list[dict]:
    """Split src_input along the video (batch) dimension."""
    seq = model_input.get("images") or model_input["vis_feats"]
    B = len(seq)
    chunks = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk: dict = {"src_length_batch": model_input["src_length_batch"][start:end]}
        if "images" in model_input:
            chunk["images"] = model_input["images"][start:end]
        else:
            chunk["vis_feats"] = model_input["vis_feats"][start:end]
        chunks.append(chunk)
    return chunks


def split_ppasta_input(model_input: dict, chunk_size: int) -> list[dict]:
    if "images" in model_input or "vis_feats" in model_input:
        return split_ppasta_src_input(model_input, chunk_size)
    return split_tgt_input(model_input, chunk_size)


class FrameChunkedEncoder:
    """Two-phase gradient accumulation over ViT frame chunks.

    Applies GradCache's technique at the frame level inside a single encoder
    call so only one frame chunk's ViT activations live on GPU at a time.

    Not an nn.Module — holds plain references to modules owned by the calling
    PPASTAImageEncoder to avoid double-registering parameters.

    Phase 1 (forward): ViT runs per-chunk under no_grad; all_vis is detached
    and promoted to a grad leaf; Perceiver runs with full grad on all_vis.
    Per-chunk pixel_values and RNG states are saved for phase 2.

    Phase 2 (chunked_backward): backward through Perceiver gives all_vis.grad;
    each ViT chunk is re-run independently with grad to accumulate ViT param
    gradients, then freed before the next chunk.

    ~1.5x ViT forward passes; peak VRAM for ViT
    activations drops from O(total_frame_chunks) to O(1).
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        perceiver: nn.Module,
        align_proj: nn.Module | None,
        image_processor,
        vision_chunk_size: int,
        gemma4_max_soft_tokens: int | None,
        max_frames_with_grad: int,
    ) -> None:
        self.vision_tower = vision_tower
        self.perceiver = perceiver
        self.align_proj = align_proj
        self._image_processor = image_processor
        self.vision_chunk_size = vision_chunk_size
        self.gemma4_max_soft_tokens = gemma4_max_soft_tokens
        self.max_frames_with_grad = max_frames_with_grad
        self._saved: dict | None = None

    def _build_pv(
        self,
        chunk: list,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        proc_kwargs: dict = {"images": chunk, "return_tensors": "pt"}
        if self.gemma4_max_soft_tokens is not None:
            proc_kwargs["max_soft_tokens"] = self.gemma4_max_soft_tokens
        inputs = self._image_processor(**proc_kwargs).to(device)
        return inputs["pixel_values"].to(dtype), inputs.get("image_position_ids")

    def _vit_pool(
        self,
        pv: torch.Tensor,
        pos_ids: torch.Tensor | None,
        n_frames: int,
    ) -> torch.Tensor:
        """ViT forward + GAP over patches → (n_frames, D_vit)."""
        vis = self.vision_tower(pv, pixel_position_ids=pos_ids).last_hidden_state
        if vis.dim() == 2:
            vis = vis.view(n_frames, vis.shape[0] // n_frames, vis.shape[-1])
        return vis.mean(dim=1)

    def _perceiver_forward(
        self,
        all_vis_leaf: torch.Tensor,
        video_lengths: list[int],
    ) -> torch.Tensor:
        """Perceiver on all_vis_leaf → (B, K, D) normalised tokens."""
        vid_tokens: list[torch.Tensor] = []
        start = 0
        for length in video_lengths:
            vis_frames = all_vis_leaf[start : start + length]
            out = self.perceiver(vis_frames.unsqueeze(0).unsqueeze(2))  # (1, T, K, D)
            pooled = out.mean(dim=1)  # (1, K, D)
            if self.align_proj is not None:
                pooled = self.align_proj(pooled)
            vid_tokens.append(F.normalize(pooled, dim=-1))
            start += length
        return torch.cat(vid_tokens, dim=0)  # (B, K, D)

    def forward(self, src_input: dict) -> torch.Tensor:
        """Phase 1: ViT no_grad per frame chunk; Perceiver with grad. Saves chunk state."""
        images_batch = src_input["images"]
        device = next(self.perceiver.parameters()).device
        dtype = next(self.perceiver.parameters()).dtype

        capped: list[list] = []
        for vid_frames in images_batch:
            if len(vid_frames) > self.max_frames_with_grad:
                step = len(vid_frames) / self.max_frames_with_grad
                vid_frames = [vid_frames[int(i * step)] for i in range(self.max_frames_with_grad)]
            capped.append(vid_frames)

        video_lengths = [len(f) for f in capped]
        flat_frames = [f for frames in capped for f in frames]

        saved_chunks: list[tuple] = []
        vis_pooled_list: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(flat_frames), self.vision_chunk_size):
                chunk = flat_frames[i : i + self.vision_chunk_size]
                pv, pos_ids = self._build_pv(chunk, device, dtype)
                rng = RandContext(pv)
                vis_pooled_list.append(self._vit_pool(pv, pos_ids, len(chunk)))
                saved_chunks.append((pv, pos_ids, len(chunk), rng))

        all_vis = torch.cat(vis_pooled_list, dim=0)  # (total_frames, D_vit)
        all_vis_leaf = all_vis.detach().requires_grad_(True)

        output = self._perceiver_forward(all_vis_leaf, video_lengths)

        self._saved = {
            "chunks": saved_chunks,
            "all_vis_leaf": all_vis_leaf,
            "device": device,
            "dtype": dtype,
        }
        return output

    def chunked_backward(self, surrogate: torch.Tensor) -> None:
        """Phase 2: backward through Perceiver, then independently through each ViT chunk."""
        assert self._saved is not None, "chunked_backward() must be called after forward()"
        saved = self._saved
        self._saved = None

        all_vis_leaf: torch.Tensor = saved["all_vis_leaf"]
        device: torch.device = saved["device"]
        dtype: torch.dtype = saved["dtype"]

        # Step 1: propagate surrogate through Perceiver → all_vis_leaf.grad.
        surrogate.backward()
        all_vis_grad = all_vis_leaf.grad  # (total_frames, D_vit)

        # Step 2: re-run each ViT chunk with grad, accumulate param grads, then free.
        amp_enabled = device.type == "cuda"
        frame_idx = 0
        for pv, pos_ids, n_frames, rng in saved["chunks"]:
            grad_slice = all_vis_grad[frame_idx : frame_idx + n_frames]
            with rng:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                    vis_pooled = self._vit_pool(pv, pos_ids, n_frames)
            chunk_surrogate = torch.dot(
                vis_pooled.flatten().float(),
                grad_slice.flatten().float(),
            )
            chunk_surrogate.backward()
            frame_idx += n_frames


class PPASTAImageEncoder(nn.Module):
    """SigLIP2 ViT (LoRA) + Perceiver Resampler.

    forward() → (B, K, D) L2-normalised Perceiver latents for FILIP.
    """

    _DEFAULT_PROCESSOR: dict[str, str] = {
        "gemma4": "google/gemma-4-E2B-it",
    }

    def __init__(
        self,
        model_id: str,
        model_family: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        num_latents: int,
        num_media_embeds: int,
        vision_chunk_size: int,
        image_processor_id: str | None = None,
        align_dim: int | None = None,
        preextracted_vit_dim: int | None = None,
        use_global_repr: bool = False,
        use_frame_grad_cache: bool = False,
        max_frames_with_grad: int = 128,
    ) -> None:
        super().__init__()
        self.vision_chunk_size = vision_chunk_size
        self.use_global_repr = use_global_repr
        self.gemma4_max_soft_tokens = 70 if model_family == "gemma4" else None
        self.max_frames_with_grad = max_frames_with_grad
        self.model_family = model_family

        if preextracted_vit_dim is not None:
            # Pre-extracted mode: skip loading the ViT entirely.
            # Only the Perceiver (and optional align_proj) are created.
            self.vision_tower = None
            self._image_processor = None
            vit_hidden = preextracted_vit_dim
        else:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                bias="none",
            )

            if model_family == "gemma4":
                from transformers import Gemma4ForConditionalGeneration

                base = Gemma4ForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )

                def _unwrap_clippable(m: torch.nn.Module) -> None:
                    for name, child in list(m.named_children()):
                        if type(child).__name__ == "Gemma4ClippableLinear":
                            setattr(m, name, child.linear)
                        else:
                            _unwrap_clippable(child)

                _unwrap_clippable(base.model.vision_tower)
                base.model.vision_tower = get_peft_model(base.model.vision_tower, lora_cfg)
                # base.model.vision_tower.gradient_checkpointing_enable()
                self.vision_tower = base.model.vision_tower
                vit_hidden = base.config.vision_config.hidden_size
                del base
            else:
                msg = f"Unknown model_family: {model_family!r}. Choose 'gemma4'."
                raise ValueError(msg)

            gc.collect()

            proc_id = image_processor_id or self._DEFAULT_PROCESSOR[model_family]
            self._image_processor = AutoImageProcessor.from_pretrained(proc_id)
            ip = getattr(self._image_processor, "image_processor", self._image_processor)
            for attr in ("do_image_splitting", "do_pan_and_scan"):
                if hasattr(ip, attr):
                    setattr(ip, attr, False)

        self.vit_hidden = vit_hidden
        self.num_latents = num_latents

        self.perceiver = PerceiverResampler(
            dim=vit_hidden,
            depth=4,
            dim_head=128,
            heads=8,
            num_latents=num_latents,
            num_media_embeds=num_media_embeds,
        )

        if align_dim is not None and align_dim != vit_hidden:
            self.align_proj: nn.Linear | None = nn.Linear(vit_hidden, align_dim)
        else:
            self.align_proj = None

        self._frame_enc: FrameChunkedEncoder | None = None
        if use_frame_grad_cache and preextracted_vit_dim is None:
            self._frame_enc = FrameChunkedEncoder(
                vision_tower=self.vision_tower,
                perceiver=self.perceiver,
                align_proj=self.align_proj,
                image_processor=self._image_processor,
                vision_chunk_size=vision_chunk_size,
                gemma4_max_soft_tokens=self.gemma4_max_soft_tokens,
                max_frames_with_grad=self.max_frames_with_grad,
            )

    def chunked_backward(self, surrogate: torch.Tensor) -> None:
        assert self._frame_enc is not None, "chunked_backward requires use_frame_grad_cache=True"
        self._frame_enc.chunked_backward(surrogate)

    def _forward_preextracted(self, src_input: dict) -> torch.Tensor:
        """Bypass ViT using pre-extracted features; run only the Perceiver.

        Returns (B, K, D) L2-normalised Perceiver latents for FILIP.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        vid_tokens_list: list[torch.Tensor] = []
        for vis_frames in src_input["vis_feats"]:
            vis_frames = vis_frames.to(device, dtype=dtype, non_blocking=True)
            if self.use_global_repr:
                # (T, D) → (1, 1, T, D): Perceiver attends across all frames at once
                x = vis_frames.unsqueeze(0).unsqueeze(0)
                out = self.perceiver(x)  # (1, 1, K, D)
                pooled = out.squeeze(1)  # (1, K, D)
            else:
                # (T, D) gap → (1, T, 1, D); (T, P, D) patches → (1, T, P, D)
                x = (
                    vis_frames.unsqueeze(0)
                    if vis_frames.dim() == 3
                    else vis_frames.unsqueeze(0).unsqueeze(2)
                )
                out = self.perceiver(x)  # (1, T, K, D)
                pooled = out.mean(dim=1)  # (1, K, D) — collapse temporal dim
            if self.align_proj is not None:
                pooled = self.align_proj(pooled)  # (1, K, align_dim)
            vid_tokens_list.append(F.normalize(pooled, dim=-1))  # (1, K, D)
        return torch.cat(vid_tokens_list, dim=0)  # (B, K, D)

    def forward_latents(self, src_input: dict) -> torch.Tensor:
        """Return (B, K, D_vit) Perceiver latents (before CLS pooling).

        Used by downstream translation models (e.g. PASTA) that need a
        sequence of latent tokens rather than a single pooled embedding.
        Supports both PIL image mode (src_input["images"]) and pre-extracted
        mode (src_input["vis_feats"]).
        """
        if "vis_feats" in src_input:
            return self._forward_latents_preextracted(src_input)

        images_batch = src_input["images"]
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        capped_batch: list[list] = []
        for vid_frames in images_batch:
            if len(vid_frames) > self.max_frames_with_grad:
                step = len(vid_frames) / self.max_frames_with_grad
                vid_frames = [vid_frames[int(i * step)] for i in range(self.max_frames_with_grad)]
            capped_batch.append(vid_frames)

        video_lengths = [len(f) for f in capped_batch]
        flat_frames = [f for frames in capped_batch for f in frames]

        vis_pooled_list: list[torch.Tensor] = []
        for i in range(0, len(flat_frames), self.vision_chunk_size):
            chunk = flat_frames[i : i + self.vision_chunk_size]
            proc_kwargs: dict = {"images": chunk, "return_tensors": "pt"}
            if self.gemma4_max_soft_tokens is not None:
                proc_kwargs["max_soft_tokens"] = self.gemma4_max_soft_tokens
            inputs = self._image_processor(**proc_kwargs).to(device)
            pv = inputs["pixel_values"].to(dtype)
            pos_ids = inputs.get("image_position_ids")
            vis = self.vision_tower(pv, pixel_position_ids=pos_ids).last_hidden_state
            if vis.dim() == 2:
                c = len(chunk)
                vis = vis.view(c, vis.shape[0] // c, vis.shape[-1])
            vis_pooled_list.append(vis.mean(dim=1))

        all_vis = torch.cat(vis_pooled_list, dim=0)

        latents_list: list[torch.Tensor] = []
        start = 0
        for length in video_lengths:
            vis_frames = all_vis[start : start + length]
            out = self.perceiver(vis_frames.unsqueeze(0).unsqueeze(2))  # (1, T, K, D)
            pooled = out.mean(dim=1)  # (1, K, D)
            latents_list.append(pooled)
            start += length

        return torch.cat(latents_list, dim=0)  # (B, K, D_vit)

    def _forward_latents_preextracted(self, src_input: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        feats = src_input["vis_feats"]  # list of (T_i, D_vit) CPU tensors

        if self.use_global_repr:
            # Process each video individually: (T, D) → (1, 1, T, D)
            latents_list: list[torch.Tensor] = []
            for vis_frames in feats:
                vis_frames = vis_frames.to(device, dtype=dtype, non_blocking=True)
                out = self.perceiver(vis_frames.unsqueeze(0).unsqueeze(0))  # (1, 1, K, D)
                latents_list.append(out.squeeze(1))  # (1, K, D)
            return torch.cat(latents_list, dim=0)  # (B, K, D)

        lengths = [len(f) for f in feats]
        max_T = max(lengths)
        B = len(feats)
        D = feats[0].shape[-1]

        # Single padded transfer: (B, max_T, P, D_vit)
        # feats[0] is (T, D) for gap or (T, P, D) for patches
        if feats[0].dim() == 3:
            n_patches = feats[0].shape[1]
            padded = torch.zeros(B, max_T, n_patches, D, dtype=dtype, device=device)
            for i, (f, t) in enumerate(zip(feats, lengths, strict=False)):
                padded[i, :t] = f[:t].to(dtype=dtype, non_blocking=True)
        else:
            padded = torch.zeros(B, max_T, 1, D, dtype=dtype, device=device)
            for i, (f, t) in enumerate(zip(feats, lengths, strict=False)):
                padded[i, :t, 0, :] = f[:t].to(dtype=dtype, non_blocking=True)

        # One Perceiver call for the whole batch: (B, max_T, K, D)
        out = self.perceiver(padded)

        # Masked mean over T to ignore padding
        mask = torch.arange(max_T, device=device).unsqueeze(0) < torch.tensor(
            lengths,
            device=device,
        ).unsqueeze(1)
        mask = mask[:, :, None, None].to(dtype)  # (B, max_T, 1, 1)
        return (out * mask).sum(dim=1) / mask.sum(dim=1)  # (B, K, D)

    def forward(self, src_input: dict) -> torch.Tensor:
        """Return (B, K, D) L2-normalised Perceiver latents for FILIP."""
        if "vis_feats" in src_input:
            return self._forward_preextracted(src_input)

        if self._frame_enc is not None:
            return self._frame_enc.forward(src_input)

        images_batch = src_input["images"]
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Cap frames per video in both grad and no-grad passes: keeps the no-grad
        # representation consistent with the grad pass and bounds ViT call count.
        capped_batch: list[list[Image.Image]] = []
        for vid_frames in images_batch:
            if len(vid_frames) > self.max_frames_with_grad:
                step = len(vid_frames) / self.max_frames_with_grad
                vid_frames = [vid_frames[int(i * step)] for i in range(self.max_frames_with_grad)]
            capped_batch.append(vid_frames)

        video_lengths = [len(f) for f in capped_batch]
        flat_frames = [f for frames in capped_batch for f in frames]

        # Encode all frames across all videos in a single chunk loop.
        # GAP-pool patches immediately so we never hold (T*B, P, D_vit) in VRAM.
        vis_pooled_list: list[torch.Tensor] = []
        for i in range(0, len(flat_frames), self.vision_chunk_size):
            chunk = flat_frames[i : i + self.vision_chunk_size]
            proc_kwargs: dict = {"images": chunk, "return_tensors": "pt"}
            if self.gemma4_max_soft_tokens is not None:
                proc_kwargs["max_soft_tokens"] = self.gemma4_max_soft_tokens
            inputs = self._image_processor(**proc_kwargs).to(device)
            pv = inputs["pixel_values"].to(dtype)
            # (c, num_patches, 2) for Gemma4; None for SigLIP
            pos_ids = inputs.get("image_position_ids")
            vis = self.vision_tower(pv, pixel_position_ids=pos_ids).last_hidden_state
            # Gemma4VisionModel strips padding and flattens the batch dimension,
            # returning (total_valid_patches, D). Reshape back to (c, p, D).
            if vis.dim() == 2:
                c = len(chunk)
                vis = vis.view(c, vis.shape[0] // c, vis.shape[-1])
            vis_pooled_list.append(vis.mean(dim=1))  # (c, D_vit) — GAP over patches

        all_vis = torch.cat(vis_pooled_list, dim=0)  # (total_frames, D_vit)

        # Split by video, run Perceiver inline so patch tensors are freed per video.
        vid_tokens_list: list[torch.Tensor] = []
        start = 0
        for length in video_lengths:
            vis_frames = all_vis[start : start + length]  # (T, D_vit)
            # (1, T, 1, D_vit): media=T frames, 1 token each — uses temporal pos embs.
            out = self.perceiver(vis_frames.unsqueeze(0).unsqueeze(2))  # (1, T, K, D)
            pooled = out.mean(dim=1)  # (1, K, D) — collapse temporal dim
            if self.align_proj is not None:
                pooled = self.align_proj(pooled)  # (1, K, align_dim)
            vid_tokens_list.append(F.normalize(pooled, dim=-1))  # (1, K, D)
            start += length

        return torch.cat(vid_tokens_list, dim=0)  # (B, K, D)


class PPASTATextEncoder(nn.Module):
    """Pass-through for pre-extracted SigLIP token embeddings.

    Returns (B, L, D) L2-normalised token tensors for FILIP, or (B, D) for
    gap. Zero-padded positions (from variable-length sequences)
    normalise to zero vectors, which are detected as padding in filip_loss_fn.
    """

    def forward(self, tgt_input: dict) -> torch.Tensor:
        feat = tgt_input["siglip_feat"].float()
        if torch.cuda.is_available():
            feat = feat.cuda()
        # GradCache needs a grad_fn on the output to call surrogate.backward().
        # This encoder has no parameters, so we attach requires_grad to the leaf
        # input; backward computes a gradient for feat but no parameter is updated.
        feat = feat.requires_grad_(True)
        return F.normalize(feat, dim=-1)  # (B, L, D) or (B, D)


class PPASTA(nn.Module):
    def __init__(
        self,
        args,
        config,
        align_dim: int | None = None,
        preextracted_vit_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.model_image = PPASTAImageEncoder(
            model_id=args.model_id,
            model_family=args.model_family,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_latents=args.num_latents,
            num_media_embeds=args.num_media_embeds,
            vision_chunk_size=args.vision_chunk_size,
            image_processor_id=args.image_processor_id or None,
            align_dim=align_dim,
            preextracted_vit_dim=preextracted_vit_dim,
            use_global_repr=getattr(args, "use_global_repr", False),
            use_frame_grad_cache=getattr(args, "frame_grad_cache", False),
            max_frames_with_grad=args.max_frames,
        )
        self.model_text = PPASTATextEncoder()
        self.temperature = args.temperature

    def forward(self, src_input: dict, tgt_input: dict) -> torch.Tensor:
        """Direct forward used during eval (no GradCache)."""
        sentence_emb = self.model_image(src_input)
        text_feat = self.model_text(tgt_input)
        # sentence_emb: (B, K, D) video tokens; text_feat: (B, L, D) or (B, D)
        if sentence_emb.dim() == 3 and text_feat.dim() == 3:
            return filip_loss_fn(sentence_emb, text_feat, self.temperature)
        return info_nce(sentence_emb, text_feat, self.temperature)


def get_args_parser():
    parser = argparse.ArgumentParser("PPASTA Pretraining", add_help=False)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--finetune", default="", help="load weights from checkpoint")

    # Optimizer
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER")
    parser.add_argument("--opt-eps", default=1e-9, type=float, metavar="EPSILON")
    parser.add_argument("--opt-betas", default=[0.9, 0.98], type=float, nargs="+")
    parser.add_argument("--clip-grad", type=float, default=1.0, metavar="NORM")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M")
    parser.add_argument("--weight-decay", type=float, default=0.05)

    # LR schedule
    parser.add_argument("--sched", default="cosine", type=str, metavar="SCHEDULER")
    parser.add_argument(
        "--use-lr-scheduler",
        action="store_true",
        help="Enable learning-rate scheduler.",
    )
    parser.add_argument(
        "--no-lr-scheduler",
        action="store_false",
        dest="use_lr_scheduler",
        help="Disable learning-rate scheduler.",
    )
    parser.set_defaults(use_lr_scheduler=True)
    parser.add_argument("--lr", type=float, default=1e-4, metavar="LR")
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None)
    parser.add_argument("--lr-noise-pct", type=float, default=0.67)
    parser.add_argument("--lr-noise-std", type=float, default=1.0)
    parser.add_argument("--warmup-lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--min-lr", type=float, default=4e-5, metavar="LR")
    parser.add_argument("--decay-epochs", type=float, default=30, metavar="N")
    parser.add_argument("--warmup-epochs", type=int, default=2, metavar="N")
    parser.add_argument("--cooldown-epochs", type=int, default=5, metavar="N")
    parser.add_argument("--patience-epochs", type=int, default=10, metavar="N")
    parser.add_argument("--decay-rate", "--dr", type=float, default=0.1, metavar="RATE")

    # I/O
    parser.add_argument("--output_dir", default="out/ppasta")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the validation loop.")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_num_workers", default=2, type=int)
    parser.add_argument("--prefetch_factor", default=1, type=int)
    parser.add_argument("--eval_prefetch_factor", default=1, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)  # PIL lists can't be pinned
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument("--config", type=str, default="configs/phoenix2014t.yaml")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="1 epoch, 2 batches (smoke test).",
    )

    # Model
    parser.add_argument("--model_id", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument(
        "--model_family",
        choices=["llava", "gemma4"],
        default="gemma4",
        help=(
            "Model family to use as the visual backbone. "
            "'gemma4': Gemma 4 — SigLIP2 ViT + Gemma MLP projector. "
            "Set --model_id to the matching HuggingFace repo "
            "(e.g. 'google/gemma-4-4b-it' for gemma4)."
        ),
    )
    parser.add_argument(
        "--image_processor_id",
        type=str,
        default="",
        help=(
            "Override the image processor used to preprocess frames. "
            "Defaults to the plain ViT processor for each family "
            "('google/siglip-so400m-patch14-384' for llava, "
            "'google/siglip2-so400m-patch14-384' for gemma4). "
            "Override when using a different SigLIP2 resolution variant."
        ),
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--num_latents",
        type=int,
        default=64,
        help="Perceiver Resampler K (number of output tokens).",
    )
    parser.add_argument(
        "--num_media_embeds",
        type=int,
        default=512,
        help="Perceiver max temporal positions (>= max frames per video).",
    )
    parser.add_argument(
        "--vision_chunk_size",
        type=int,
        default=8,
        help="Frames sent through ViT at once.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=128,
        help="Max frames per video sent through the ViT. Longer videos are uniformly subsampled.",
    )

    # Loss
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--grad_chunk_size",
        type=int,
        default=None,
        help="GradCache chunk size. Defaults to batch-size.",
    )
    parser.add_argument(
        "--frame-grad-cache",
        action="store_true",
        help=(
            "Apply GradCache at the frame level inside the image encoder (live ViT only). "
            "Only one frame chunk's ViT activations live on GPU at a time. "
            "Tradeoff: ~1.5x ViT forward passes vs. the naive path."
        ),
    )

    # Features
    parser.add_argument(
        "--siglip_feat_train",
        default="datasets/phoenix-translations/phoenix_translations_siglip2_train.pt",
        help="Path to siglip2 text features .pt for train split (default: translation embeddings).",
    )
    parser.add_argument(
        "--siglip_feat_dev",
        default="datasets/phoenix-translations/phoenix_translations_siglip2_dev.pt",
        help="Path to siglip2 text features .pt for dev split (default: translation embeddings).",
    )
    parser.add_argument(
        "--siglip_feat_test",
        default="datasets/phoenix-translations/phoenix_translations_siglip2_test.pt",
        help="Path to siglip2 text features .pt for test split (default: translation embeddings).",
    )

    # Pre-extracted ViT features
    parser.add_argument(
        "--preextracted_feat_dir",
        type=str,
        default=None,
        help=(
            "Directory containing features_{split}[_spatial{n}tok]/ per-video subdirectories. "
            "When set, the ViT is bypassed during training and only the Perceiver is trained."
        ),
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=0,
        help=(
            "Number of spatial patch tokens per frame. "
            "Loads features_{split}_spatial{n}tok/. Set to 0 to load raw features_{split}/."
        ),
    )
    parser.add_argument(
        "--use-global-repr",
        action="store_true",
        help=(
            "Use the global per-frame representation (key: --global-feat-key) from "
            "pre-extracted .pt files so the Perceiver attends across all video frames "
            "at once rather than per-frame with temporal pooling. "
            "Requires 'vis_global' (or --global-feat-key) in each per-video .pt file."
        ),
    )
    parser.add_argument(
        "--global-feat-key",
        type=str,
        default="vis_global",
        help="Key in the per-video .pt file for the global per-frame representation (default: vis_global).",
    )

    return parser


def train_one_epoch(
    args,
    model: PPASTA,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    set_training_mode: bool = True,
) -> dict:
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}/{args.epochs}]"
    print_freq = 10

    chunk_size = args.grad_chunk_size if args.grad_chunk_size is not None else args.batch_size
    temperature = args.temperature

    def _loss_fn(v_reps, t_reps):
        if v_reps.dim() == 3 and t_reps.dim() == 3:
            return filip_loss_fn(v_reps, t_reps, temperature)
        return info_nce(v_reps, t_reps, temperature)

    amp_enabled = device.type == "cuda"

    class _GradCache(GradCache):
        def model_call(self, enc, model_input):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                return enc(model_input)

        def forward_backward(
            self,
            model,
            model_inputs,
            cached_gradients,
            random_states,
            no_sync_except_last=False,
        ):
            if not hasattr(model, "chunked_backward") or getattr(model, "_frame_enc", None) is None:
                return super().forward_backward(
                    model,
                    model_inputs,
                    cached_gradients,
                    random_states,
                    no_sync_except_last,
                )
            for x, state, gradient in zip(
                model_inputs,
                random_states,
                cached_gradients,
                strict=False,
            ):
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)
                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                model.chunked_backward(surrogate)
            return None

    grad_cache = _GradCache(
        models=[model.model_image, model.model_text],
        chunk_sizes=[chunk_size, chunk_size],
        loss_fn=_loss_fn,
        split_input_fn=split_ppasta_input,
        fp16=False,
        device=device,
    )

    optimizer.zero_grad()
    loss_value = 0.0

    for step, (src_input, tgt_input) in enumerate(
        tqdm(
            metric_logger.log_every(data_loader, print_freq, header),
            total=len(data_loader),
            disable=not utils.is_main_process(),
        ),
    ):
        if args.debug and step >= 10:
            print("DEBUG MODE: stopping after 2 batches.")
            break

        loss = grad_cache(src_input, tgt_input)
        optimizer.step()
        optimizer.zero_grad()

        loss_value = loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model: PPASTA, epoch: int, device: torch.device) -> dict:
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 10
    loss_val = 0.0

    amp_enabled = device.type == "cuda"
    for step, (src_input, tgt_input) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header),
    ):
        if args.debug and step >= 2:
            print("DEBUG MODE: stopping after 2 batches.")
            break

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            loss = model(src_input, tgt_input)

        loss_val = loss.item()
        metric_logger.update(loss=loss_val)

    metric_logger.synchronize_between_processes()
    print("* Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def log_memory(args, tag: str) -> None:
    if not getattr(args, "log_memory", False):
        return
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / (1024 * 1024) if sys.platform == "darwin" else rss_kb / 1024
    msg = f"[memory] {tag} rss_max_mb={rss_mb:.2f}"
    if torch.cuda.is_available():
        msg += (
            f" cuda_allocated_mb={torch.cuda.memory_allocated() / 1e6:.2f}"
            f" cuda_reserved_mb={torch.cuda.memory_reserved() / 1e6:.2f}"
            f" cuda_max_allocated_mb={torch.cuda.max_memory_allocated() / 1e6:.2f}"
        )
    print(msg)


def main(args, config) -> None:
    args.distributed = False
    args.gpu = None
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    print("Loading SigLIP text features …")
    siglip_by_split = {
        "train": load_siglip_features(args.siglip_feat_train),
        "dev": load_siglip_features(args.siglip_feat_dev),
        "test": load_siglip_features(args.siglip_feat_test),
    }

    vis_proj_by_split: dict[str, dict] | dict[str, None] = {
        "train": None,
        "dev": None,
        "test": None,
    }
    if args.preextracted_feat_dir:
        feat_dir = Path(args.preextracted_feat_dir)
        print(f"Using pre-extracted ViT features from {feat_dir} …")
        for split in ["train", "dev", "test"]:
            if args.n_tokens:
                p = feat_dir / f"features_{split}_spatial{args.n_tokens}tok"
            else:
                p = feat_dir / f"features_{split}"
            if p.is_dir():
                vis_proj_by_split[split] = LazyFeatureDir(p)
            else:
                msg = f"Expected directory {p} for pre-extracted features, but it does not exist."
                raise FileNotFoundError(msg)
            print(f"  [{split}] {len(vis_proj_by_split[split])} videos")

    print("Creating datasets …")

    def make_dataset(phase: str) -> PPASTADataset:
        return PPASTADataset(
            path=config["data"]["label_path"],
            config=config,
            args=args,
            phase=phase,
            siglip_feats=siglip_by_split[phase],
            vis_proj_feats=vis_proj_by_split[phase],
        )

    train_data = make_dataset("train")
    dev_data = make_dataset("dev")
    test_data = make_dataset("test")
    print(train_data)
    print(dev_data)
    print(test_data)

    train_sampler = torch.utils.data.RandomSampler(train_data)
    dev_sampler = torch.utils.data.SequentialSampler(dev_data)
    test_sampler = torch.utils.data.SequentialSampler(test_data)

    def make_train_loader():
        kwargs = {
            "dataset": train_data,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "collate_fn": train_data.collate_fn,
            "sampler": train_sampler,
            "pin_memory": False,  # PIL lists can't be pinned
            "drop_last": False,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            kwargs["prefetch_factor"] = args.prefetch_factor
        return DataLoader(**kwargs)

    def make_eval_loader(dataset, sampler):
        kwargs = {
            "dataset": dataset,
            "batch_size": args.batch_size,
            "num_workers": args.eval_num_workers,
            "collate_fn": dataset.collate_fn,
            "sampler": sampler,
            "pin_memory": False,
            "persistent_workers": False,
        }
        if args.eval_num_workers > 0:
            kwargs["prefetch_factor"] = args.eval_prefetch_factor
        return DataLoader(**kwargs)

    dev_loader = make_eval_loader(dev_data, dev_sampler)
    test_loader = make_eval_loader(test_data, test_sampler)

    # --- Model ---
    print("Creating PPASTA model …")
    siglip_dim = next(iter(siglip_by_split["train"].values())).shape[-1]

    # When using pre-extracted features, read D_vit from file metadata so the
    # Perceiver is built with the correct dimension and the ViT is not loaded.
    preextracted_vit_dim: int | None = None
    if args.preextracted_feat_dir:
        train_feats = vis_proj_by_split["train"]
        if isinstance(train_feats, LazyFeatureDir):
            preextracted_vit_dim = train_feats.d_vit
        else:
            msg = "Expected vis_proj_by_split[split] to be LazyFeatureDir when using pre-extracted features."
            raise ValueError(msg)
        print(f"Pre-extracted mode: D_vit={preextracted_vit_dim}, ViT not loaded.")

    model = PPASTA(
        args=args,
        config=config,
        align_dim=siglip_dim,
        preextracted_vit_dim=preextracted_vit_dim,
    ).to(device)

    if args.finetune:
        ckpt = torch.load(args.finetune, map_location="cpu", weights_only=False)
        ret = model.load_state_dict(ckpt["model"], strict=False)
        print("Missing keys:\n", "\n".join(ret.missing_keys))
        print("Unexpected keys:\n", "\n".join(ret.unexpected_keys))

    n_params = utils.count_parameters_in_MB(model)
    print(f"Trainable parameters: {n_params:.1f}M")

    optimizer = create_optimizer(args, model)
    lr_scheduler = None
    if args.use_lr_scheduler:
        lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if not args.eval and "optimizer" in ckpt and "epoch" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            if (
                lr_scheduler is not None
                and "lr_scheduler" in ckpt
                and ckpt["lr_scheduler"] is not None
            ):
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            args.start_epoch = ckpt["epoch"] + 1

    if args.eval:
        if not args.resume:
            logger.warning("Specify --resume /path/to/checkpoint.pth for eval.")
        for split, loader in [("dev", dev_loader), ("test", test_loader)]:
            stats = evaluate(args, loader, model, args.start_epoch, device)
            print(f"{split} loss: {stats['loss']:.4f}")
        return

    if args.debug:
        args.epochs = args.start_epoch + 1

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = np.inf

    for epoch in tqdm(
        range(args.start_epoch, args.epochs),
        total=args.epochs - args.start_epoch,
        disable=not utils.is_main_process(),
    ):
        log_memory(args, f"epoch_{epoch}_start")

        train_loader = make_train_loader()
        train_stats = train_one_epoch(args, model, train_loader, optimizer, device, epoch)
        del train_loader
        log_memory(args, f"epoch_{epoch}_after_train")
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            if lr_scheduler is not None:
                checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
            utils.save_on_master(
                checkpoint,
                output_dir / "checkpoint.pth",
            )

        dev_stats = None
        if not args.skip_validation:
            dev_stats = evaluate(args, dev_loader, model, epoch, device)

        if dev_stats is not None and min_loss > dev_stats["loss"]:
            min_loss = dev_stats["loss"]
            if args.output_dir:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_stats": train_stats,
                    "dev_stats": dev_stats,
                    "min_loss": min_loss,
                    "n_parameters": n_params,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if lr_scheduler is not None:
                    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
                utils.save_on_master(
                    checkpoint,
                    output_dir / "best_checkpoint.pth",
                )

        if dev_stats is not None and args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_stats": train_stats,
                "dev_stats": dev_stats,
                "min_loss": min_loss,
                "n_parameters": n_params,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if lr_scheduler is not None:
                checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
            loss_tag = f"{dev_stats['loss']:.4f}".replace(".", "p")
            utils.save_on_master(
                checkpoint,
                output_dir / f"checkpoint_epoch_{epoch}_devloss_{loss_tag}.pth",
            )

        if dev_stats is not None:
            print(f"* DEV loss {dev_stats['loss']:.4f}  (best {min_loss:.4f})")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_params,
        }
        if dev_stats is not None:
            log_stats.update({f"dev_{k}": v for k, v in dev_stats.items()})
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        log_memory(args, f"epoch_{epoch}_end")

    # Final eval on best checkpoint
    if args.output_dir and not args.skip_validation:
        ckpt = torch.load(
            str(output_dir / "best_checkpoint.pth"),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(ckpt["model"], strict=True)
        for split, loader in [("dev", dev_loader), ("test", test_loader)]:
            stats = evaluate(args, loader, model, epoch, device)
            print(f"[best ckpt] {split} loss {stats['loss']:.4f}")

    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


def cli() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser("PPASTA", parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)


if __name__ == "__main__":
    cli()
