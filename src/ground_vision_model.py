from __future__ import annotations

import argparse
import pathlib
import random

import torch
import torch.nn.functional as F
from flamingo_pytorch import PerceiverResampler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from load_descript_features import load_descript_features


class FramePairDataset(Dataset):
    """Flat list of (frame_path, teacher_feature) pairs across all videos/splits.

    Each item is one frame.  The teacher feature is the pre-extracted LLaVA
    hidden-state vector for that same frame (same position in the video).

    When the number of actual frames on disk differs from the number of stored
    teacher features, we take the minimum so indices always correspond.
    """

    def __init__(
        self,
        features: dict[str, torch.Tensor],
        img_root: str | pathlib.Path,
        split: str,
        max_frames_per_video: int | None = None,
    ) -> None:
        self.img_root = pathlib.Path(img_root) / split
        self.pairs: list[tuple[pathlib.Path, torch.Tensor]] = []

        for vid_name, teacher_feats in features.items():
            frame_dir = self.img_root / vid_name
            if not frame_dir.is_dir():
                continue

            frame_paths = sorted(p for p in frame_dir.iterdir() if p.suffix in {".png", ".jpg"})
            T = min(len(frame_paths), len(teacher_feats))
            frame_paths = frame_paths[:T]
            teacher_feats = teacher_feats[:T]

            if max_frames_per_video is not None and max_frames_per_video < T:
                indices = sorted(random.sample(range(T), max_frames_per_video))
                frame_paths = [frame_paths[i] for i in indices]
                teacher_feats = teacher_feats[indices]
                T = max_frames_per_video

            for i in range(T):
                self.pairs.append((frame_paths[i], teacher_feats[i]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Image.Image, torch.Tensor]:
        frame_path, teacher_feat = self.pairs[idx]
        img = Image.open(frame_path).convert("RGB")
        return img, teacher_feat


def collate_fn(batch: list[tuple[Image.Image, torch.Tensor]]):
    imgs, teacher_feats = zip(*batch, strict=False)
    return list(imgs), torch.stack(teacher_feats)


class VisionGrounder(nn.Module):
    """LLaVA-OV vision tower + projection head, LoRA-adapted, outputting GAP embeddings."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()

        base = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        # Apply LoRA to the vision tower (SigLIP attention projections)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
        )
        base.vision_tower = get_peft_model(base.vision_tower, lora_cfg)

        self.vision_tower = base.vision_tower
        self.projector = base.multi_modal_projector  # trained end-to-end (no LoRA needed)
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.processor.image_processor.do_image_splitting = False

        del base

    def encode_frames(self, images: list[Image.Image]) -> torch.Tensor:
        """Run images through vision tower + projector, return GAP over patch dimension.

        Args:
            images: List of PIL images (one per frame).

        Returns:
            Tensor of shape (N, D_lm) — one embedding per frame.

        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(device)
        pixel_values = inputs["pixel_values"].to(dtype)  # (N, C, H, W)

        # Vision tower → (N, num_patches, D_vis)
        vis_out = self.vision_tower(pixel_values, output_hidden_states=False).last_hidden_state

        # Projection head → (N, num_patches, D_lm)
        projected = self.projector(vis_out)

        # Global average pool over patches → (N, D_lm)
        return projected.mean(dim=1)

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        return self.encode_frames(images)


def info_nce_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between student and teacher embeddings.

    Args:
        student: (N, D) — vision tower + projector output, L2-normalised inside.
        teacher: (N, D) — pre-extracted LLaVA hidden states, L2-normalised inside.
        temperature: Logit scale divisor.

    Returns:
        Scalar loss.

    """
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    logits = student @ teacher.T / temperature  # (N, N)
    labels = torch.arange(len(student), device=student.device)
    loss_s2t = F.cross_entropy(logits, labels)
    loss_t2s = F.cross_entropy(logits.T, labels)
    return (loss_s2t + loss_t2s) / 2


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Teacher features ---
    print(f"Loading teacher features from {args.features_dir} ...")
    features = load_descript_features(
        args.features_dir,
        splits=["train"],
        hidden_layer=args.hidden_layer,
        device="cpu",
    )
    print(f"  {len(features)} videos loaded.")

    # --- Dataset / DataLoader ---
    dataset = FramePairDataset(
        features,
        img_root=args.img_root,
        split="train",
        max_frames_per_video=args.max_frames_per_video,
    )
    print(f"  {len(dataset)} frame pairs total.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # --- Model ---
    print(f"Loading LLaVA-OV from {args.model_id} ...")
    model = VisionGrounder(
        model_id=args.model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    PerceiverResampler(
        dim=1024,  # D_vit from SigLIP
        depth=2,  # number of cross-attention layers
        dim_head=64,
        heads=8,
        num_latents=64,  # your K
        num_time_embeds=16,  # your T
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training ---
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (imgs, teacher_feats) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):
            if args.debug and step >= 2:
                break

            teacher_feats = teacher_feats.to(device, dtype=torch.bfloat16)

            # Chunk large batches through the vision tower to avoid OOM,
            # accumulate gradients across chunks before stepping.
            chunk_size = args.vision_chunk_size
            student_chunks: list[torch.Tensor] = []
            for i in range(0, len(imgs), chunk_size):
                chunk_imgs = imgs[i : i + chunk_size]
                student_chunks.append(model.encode_frames(chunk_imgs))
            student_feats = torch.cat(student_chunks, dim=0)  # (N, D)

            loss = info_nce_loss(student_feats, teacher_feats, args.temperature)
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        avg_loss = running_loss / max(step + 1, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}  avg_loss={avg_loss:.4f}")

        ckpt_path = output_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "vision_tower_state": model.vision_tower.state_dict(),
                "projector_state": model.projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint → {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Data
    p.add_argument(
        "--features_dir",
        required=True,
        help="Dir containing phoenix_hidden_layerN_*.pt files",
    )
    p.add_argument(
        "--img_root",
        required=True,
        help="Root of Phoenix frame folders (parent of train/dev/test)",
    )
    p.add_argument(
        "--hidden_layer",
        type=int,
        default=20,
        help="LLaVA hidden layer used during feature extraction (default: 20)",
    )
    p.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Cap frames sampled per video per epoch (None = all)",
    )

    # Model
    p.add_argument("--model_id", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of (frame, teacher_feat) pairs per batch",
    )
    p.add_argument(
        "--vision_chunk_size",
        type=int,
        default=8,
        help="Frames processed through vision tower at once (GPU memory knob)",
    )
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="out/grounded_vision")

    # Misc
    p.add_argument("--debug", action="store_true", help="Run 1 epoch, 2 batches (smoke test)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1
    train(args)
