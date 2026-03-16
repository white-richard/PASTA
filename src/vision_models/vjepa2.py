# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#
# python vjepa2.py --video-root path/to/your/frame_folders --device cuda


import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Number of frames to sample from each video clip
NUM_FRAMES = 16
# Spatial size expected by the model
IMG_SIZE = 224
# Batch size for the demo dataloader
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Simplified video dataset  (frame-folder layout, inspired by MMSLT/datasets.py)
# ---------------------------------------------------------------------------
# Expected directory layout:
#   <root>/
#       video_001/
#           frame_000.jpg
#           frame_001.jpg
#           ...
#       video_002/
#           ...


class VideoFrameDataset(Dataset):
    """Loads videos stored as folders of JPEG/PNG frames.
    Each item returns a tensor of shape (T, C, H, W).
    """

    def __init__(self, root: str, num_frames: int = NUM_FRAMES, img_size: int = IMG_SIZE) -> None:
        self.root = root
        self.num_frames = num_frames

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ],
        )

        # Collect all video sub-directories
        self.video_dirs = sorted(
            [
                os.path.join(root, name)
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            ],
        )

        if not self.video_dirs:
            msg = f"No video sub-directories found under: {root}"
            raise RuntimeError(msg)

    def __len__(self) -> int:
        return len(self.video_dirs)

    def _load_frames(self, video_dir: str) -> torch.Tensor:
        """Read image files from *video_dir* and return (T, C, H, W) tensor."""
        frame_paths = sorted(
            [
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ],
        )

        if not frame_paths:
            msg = f"No frame images found in: {video_dir}"
            raise RuntimeError(msg)

        # Uniformly sample self.num_frames indices
        total = len(frame_paths)
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            img = cv2.imread(frame_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            frames.append(self.transform(img))  # (C, H, W)

        # Stack -> (T, C, H, W)
        return torch.stack(frames, dim=0)

    def __getitem__(self, index: int):
        video_dir = self.video_dirs[index]
        video_name = os.path.basename(video_dir)
        frames = self._load_frames(video_dir)  # (T, C, H, W)
        return video_name, frames


def collate_fn(batch):
    """Collate a list of (name, frames) tuples.
    frames shape per item: (T, C, H, W).

    Returns:
        names  – list of str
        videos – (B, C, T, H, W)  float32 tensor ready for V-JEPA-2.

    """
    names, frames_list = zip(*batch, strict=False)
    # Stack along batch dim: (B, T, C, H, W)
    videos = torch.stack(frames_list, dim=0)
    # V-JEPA-2 expects (B, C, T, H, W)
    videos = videos.permute(0, 2, 1, 3, 4).contiguous()
    return list(names), videos


# ---------------------------------------------------------------------------
# Model loading via torch.hub
# ---------------------------------------------------------------------------


def load_model(device: torch.device):
    """Load the V-JEPA-2 preprocessor and ViT-Large encoder from torch.hub.
    Returns (processor, model).
    """
    print("Loading V-JEPA-2 preprocessor from torch.hub ...")
    processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")

    print("Loading V-JEPA-2 ViT-Large encoder from torch.hub ...")
    model = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
    model.to(device).eval()

    return processor, model


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def run_forward_pass(video_root: str, device: torch.device):
    """Build a dataloader over *video_root*, run one forward batch through
    V-JEPA-2, and print the output shape.
    """
    # -- dataset & loader --
    dataset = VideoFrameDataset(root=video_root, num_frames=NUM_FRAMES, img_size=IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Dataset: {len(dataset)} video(s) found in '{video_root}'")

    # -- model --
    processor, model = load_model(device)

    # -- single forward batch --
    names, videos = next(iter(loader))
    print(f"\nBatch video names : {names}")
    print(f"Input tensor shape: {videos.shape}  (B, C, T, H, W)")

    videos = videos.to(device)

    # The processor handles any additional normalisation / patching required
    # by the hub model (returns a dict with at least 'pixel_values').
    processed = processor(videos)
    pixel_values = processed["pixel_values"].to(device)

    with torch.inference_mode():
        # model() returns patch-wise feature embeddings: (B, N_patches, embed_dim)
        patch_features = model(pixel_values)

    print(f"Output patch features shape: {patch_features.shape}  (B, N_patches, embed_dim)")
    print("\nForward pass complete.")
    return patch_features


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V-JEPA-2 forward-pass demo")
    parser.add_argument(
        "--video-root",
        type=str,
        default="sample_videos",
        help="Path to directory containing per-video frame folders.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    run_forward_pass(video_root=args.video_root, device=device)
