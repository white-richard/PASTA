from __future__ import annotations

import gzip
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from vidaug import augmentors as va

from .data_io import load_phoenix_csv
from .definition import PAD_IDX
from .utils import data_augmentation


class VideoDataset(Dataset):
    """Frame-directory dataset used by the vision-feature extraction stage."""

    def __init__(self, arg_dict: dict, phase: str) -> None:
        root = Path(arg_dict["img_path"])
        self.feat_path = root / phase
        if not self.feat_path.is_dir():
            raise FileNotFoundError(f"Phoenix frame directory not found: {self.feat_path}")
        self.video_names = sorted(p.name for p in self.feat_path.iterdir() if p.is_dir())
        self.name_to_idx = {name: idx for idx, name in enumerate(self.video_names)}

    def __len__(self) -> int:
        return len(self.video_names)

    def __getitem__(self, index: int):
        vid_name = self.video_names[index]
        frames_dir = self.feat_path / vid_name
        frame_files = sorted(
            str(path)
            for path in frames_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        return vid_name, frame_files

    def get_by_name(self, vid_name: str):
        return self[self.name_to_idx[vid_name]]


def load_dataset_file(filename: str | Path):
    """Load a legacy gzip/pickle dataset or a Phoenix corpus CSV.

    PASTA uses the CSV path. The gzip/pickle branch is retained for compatibility
    with older experiment artifacts.
    """
    try:
        with gzip.open(filename, "rb") as handle:
            return pickle.load(handle)
    except (OSError, gzip.BadGzipFile, EOFError, pickle.UnpicklingError):
        return load_phoenix_csv(filename)


class S2T_Dataset(Dataset):
    """Phoenix sign-to-text dataset used by PASTA fine-tuning."""

    def __init__(
        self,
        path,
        tokenizer,
        config,
        args,
        phase,
        use_ppasta_backbone: bool = False,
        ppasta_feat_cache: str | None = None,
        ppasta_n_tokens: int = 0,
    ) -> None:
        self.config = config
        self.args = args
        self.raw_data = load_dataset_file(path[phase])
        self.tokenizer = tokenizer
        self.phase = phase
        self.max_length = config["data"]["max_length"]
        self.img_path = Path(config["data"]["img_path"])
        self.use_ppasta_backbone = use_ppasta_backbone

        self.ppasta_feat_dir: Path | None = None
        self._ppasta_feat_is_dict = False
        if ppasta_feat_cache:
            root = Path(ppasta_feat_cache)
            new_dir = root / (
                f"features_{phase}_spatial{ppasta_n_tokens}tok"
                if ppasta_n_tokens
                else f"features_{phase}"
            )
            if new_dir.exists():
                self.ppasta_feat_dir = new_dir
                self._ppasta_feat_is_dict = True
            else:
                old_dir = root / phase
                if old_dir.exists():
                    self.ppasta_feat_dir = old_dir

        self.list = list(self.raw_data)
        if use_ppasta_backbone and self.ppasta_feat_dir is not None:
            full_len = len(self.list)
            self.list = [
                key
                for key in self.list
                if (self.ppasta_feat_dir / f"{self._video_name(key)}.pt").exists()
            ]
            print(
                f"  [{phase}] PPASTA feature cache: {self.ppasta_feat_dir} "
                f"({len(self.list)}/{full_len} videos)",
            )

        def sometimes(aug):
            return va.Sometimes(0.5, aug)

        self.seq = va.Sequential(
            [
                sometimes(va.RandomRotate(30)),
                sometimes(va.RandomResize(0.2)),
                sometimes(va.RandomTranslate(x=10, y=10)),
            ],
        )

    @staticmethod
    def _video_name(key: str) -> str:
        return key.split("/", 1)[1] if "/" in key else key

    def __len__(self) -> int:
        return len(self.list)

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        tgt_sample = sample["text"]
        name_sample = sample["name"]
        img_paths = [self.img_path / str(path).lstrip("/") for path in sample["imgs_path"]]

        if self.use_ppasta_backbone:
            vid_name = self._video_name(key)
            if self.ppasta_feat_dir is not None:
                pt_path = self.ppasta_feat_dir / f"{vid_name}.pt"
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                vis_feats = data["vis"] if self._ppasta_feat_is_dict else data
                if len(vis_feats) > self.max_length:
                    indices = sorted(random.sample(range(len(vis_feats)), k=self.max_length))
                    vis_feats = vis_feats[indices]
                return name_sample, tgt_sample, None, vis_feats

            pil_frames = self.load_pil_imgs(img_paths)
            return name_sample, tgt_sample, pil_frames, None

        img_sample, _ = self.load_imgs(img_paths)
        return name_sample, tgt_sample, img_sample

    def load_pil_imgs(self, paths: list[Path]) -> list[Image.Image]:
        if len(paths) > self.max_length:
            indices = sorted(random.sample(range(len(paths)), k=self.max_length))
            paths = [paths[i] for i in indices]
        frames: list[Image.Image] = []
        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        return frames

    def load_imgs(self, paths: list[Path]):
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
        )
        selected_indices = None
        if len(paths) > self.max_length:
            selected_indices = sorted(random.sample(range(len(paths)), k=self.max_length))
            paths = [paths[i] for i in selected_indices]

        imgs = torch.zeros(len(paths), 3, self.args.input_size, self.args.input_size)
        crop_rect, resize = data_augmentation(
            resize=(self.args.resize, self.args.resize),
            crop_size=self.args.input_size,
            is_train=(self.phase == "train"),
        )

        batch_image = []
        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                img = np.zeros((self.args.input_size, self.args.input_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_image.append(Image.fromarray(img))

        if self.phase == "train":
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):
            img_resized = img.resize(resize)
            img_tensor = data_transform(img_resized).unsqueeze(0)
            imgs[i] = img_tensor[
                :,
                :,
                crop_rect[1] : crop_rect[3],
                crop_rect[0] : crop_rect[2],
            ]
        return imgs, selected_indices

    def collate_fn(self, batch):
        if self.use_ppasta_backbone:
            name_batch = [item[0] for item in batch]
            tgt_batch = [item[1] for item in batch]
            pil_frames_batch = [item[2] for item in batch]
            vis_feats_batch = [item[3] for item in batch]
            has_vis = [features is not None for features in vis_feats_batch]
            if any(has_vis) and not all(has_vis):
                raise RuntimeError("A PPASTA batch mixed cached features and raw frames.")

            tgt_input = self.tokenizer(
                text_target=tgt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if all(has_vis):
                lengths = torch.tensor([len(features) for features in vis_feats_batch])
                return {
                    "vis_feats": vis_feats_batch,
                    "name_batch": name_batch,
                    "src_length_batch": lengths,
                }, tgt_input

            lengths = torch.tensor([len(frames) for frames in pil_frames_batch])
            return {
                "pil_frames": pil_frames_batch,
                "name_batch": name_batch,
                "src_length_batch": lengths,
            }, tgt_input

        name_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        img_tmp = [item[2] for item in batch]

        max_len = max(len(video) for video in img_tmp)
        video_length = torch.LongTensor([np.ceil(len(video) / 4.0) * 4 + 16 for video in img_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len += left_pad + right_pad

        padded_video = [
            torch.cat(
                (
                    video[0][None].expand(left_pad, -1, -1, -1),
                    video,
                    video[-1][None].expand(max_len - len(video) - left_pad, -1, -1, -1),
                ),
                dim=0,
            )
            for video in img_tmp
        ]
        img_tmp = [padded_video[i][0 : video_length[i]] for i in range(len(padded_video))]
        src_length_batch = torch.tensor([len(video) for video in img_tmp])
        img_batch = torch.cat(img_tmp, 0)

        new_src_lengths = ((((src_length_batch - 5 + 1) / 2) - 5 + 1) / 2).long()
        mask_gen = [torch.ones([length]) + 7 for length in new_src_lengths]
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()

        tgt_input = self.tokenizer(
            text_target=tgt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "input_img": img_batch,
            "attention_mask": img_padding_mask,
            "name_batch": name_batch,
            "src_length_batch": src_length_batch,
            "new_src_length_batch": new_src_lengths,
        }, tgt_input

    def __str__(self) -> str:
        return f"#total {self.phase} set: {len(self.list)}."
