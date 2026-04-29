import gzip
import os
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

from definition import *
from utils import data_augmentation


# Datasets for Generating Descriptions
class VideoDataset(Dataset):
    def __init__(self, arg_dict, phase) -> None:

        # self.image_processor = image_processor
        self.arg_dict = arg_dict
        # self.process_config = process_config
        self.feat_path = arg_dict.get("img_path") + phase
        self.video_names = [
            name
            for name in os.listdir(self.feat_path)
            if os.path.isdir(os.path.join(self.feat_path, name))
        ]
        assert self.feat_path is not None
        self.name_to_idx = {name: idx for idx, name in enumerate(self.video_names)}

    def __len__(self) -> int:

        return len(self.video_names)

    def __getitem__(self, index: int):

        vid_name = self.video_names[index]
        frames_dir = os.path.join(self.feat_path, vid_name)
        frame_files = sorted(
            [
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith((".png", ".jpg"))
            ],
        )

        return vid_name, frame_files

    def get_by_name(self, vid_name: str):
        idx = self.name_to_idx[vid_name]
        return self.__getitem__(idx)


class MissDataset(Dataset):
    def __init__(self, arg_dict, phase) -> None:

        self.arg_dict = arg_dict
        self.feat_path = arg_dict.get("img_path") + phase
        self.video_names = [
            name
            for name in os.listdir(self.feat_path)
            if os.path.isdir(os.path.join(self.feat_path, name))
        ]
        assert self.feat_path is not None
        self.videos = torch.load(f"resume_path.{phase}")
        self.keys = list(self.videos.keys())
        self.filtered = [name for name in self.video_names if name not in self.keys]

    def __len__(self) -> int:

        return len(self.filtered)

    def __getitem__(self, index: int):

        vid_name = self.filtered[index]
        frames_dir = os.path.join(self.feat_path, vid_name)
        frame_files = sorted(
            [
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith((".png", ".jpg"))
            ],
        )

        return vid_name, frame_files


def load_dataset_file(filename):
    # Try gzip+pickle first (original format), fall back to Phoenix CSV
    try:
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        pass

    # Phoenix CSV format: pipe-delimited with columns name|video|start|end|speaker|orth|translation
    import csv
    import glob as _glob

    data = {}
    # Infer split from filename
    fname = os.path.basename(filename)
    if "train" in fname:
        split = "train"
    elif "dev" in fname:
        split = "dev"
    elif "test" in fname:
        split = "test"
    else:
        split = "train"

    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            name = row["name"].strip()
            translation = row["translation"].strip()
            # video col: e.g. "VideoName/1/*.png" — extract folder name
            video_col = row["video"].strip()
            video_name = video_col.split("/")[0]
            # Expand glob relative to the image root (two levels up from annotations/manual)
            annotations_dir = os.path.dirname(os.path.abspath(filename))
            img_root = os.path.join(annotations_dir, "..", "..", "features", "fullFrame-210x260px")
            frame_dir = os.path.join(img_root, split, video_name)
            frames = sorted(_glob.glob(os.path.join(frame_dir, "*.png")))
            if not frames:
                continue
            rel_frames = ["/" + os.path.relpath(f, img_root) for f in frames]
            key = f"{split}/{video_name}"
            data[key] = {
                "name": name,
                "text": translation,
                "imgs_path": rel_frames,
            }
    return data


# Datasets for MMLP and SLT
def load_dataset_file(filename):
    # Try gzip+pickle first (original format), fall back to Phoenix CSV
    try:
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        pass

    # Phoenix CSV format: pipe-delimited with columns name|video|start|end|speaker|orth|translation
    import csv
    import glob as _glob

    data = {}
    # Infer split from filename
    fname = os.path.basename(filename)
    if "train" in fname:
        split = "train"
    elif "dev" in fname:
        split = "dev"
    elif "test" in fname:
        split = "test"
    else:
        split = "train"

    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            name = row["name"].strip()
            translation = row["translation"].strip()
            # video col: e.g. "VideoName/1/*.png" — extract folder name
            video_col = row["video"].strip()
            video_name = video_col.split("/")[0]
            # Expand glob relative to the image root (two levels up from annotations/manual)
            annotations_dir = os.path.dirname(os.path.abspath(filename))
            img_root = os.path.join(annotations_dir, "..", "..", "features", "fullFrame-210x260px")
            frame_dir = os.path.join(img_root, split, video_name)
            frames = sorted(_glob.glob(os.path.join(frame_dir, "*.png")))
            if not frames:
                continue
            rel_frames = ["/" + os.path.relpath(f, img_root) for f in frames]
            key = f"{split}/{video_name}"
            data[key] = {
                "name": name,
                "text": translation,
                "imgs_path": rel_frames,
            }
    return data


class S2T_Dataset(Dataset):
    def __init__(self, path, tokenizer, config, args, phase, use_gmmlp_backbone=False, gmmlp_feat_cache=None) -> None:
        self.config = config
        self.args = args

        self.raw_data = load_dataset_file(path[phase])
        self.tokenizer = tokenizer
        self.phase = phase
        # self.descript_feat = torch.load(config["data"]["descript_feat_path"][phase])
        self.descript_feat = torch.load(
            config["data"]["descript_feat_path"][phase],
            weights_only=False,
        )
        self.max_length = config["data"]["max_length"]
        self.img_path = config["data"]["img_path"]
        self.use_gmmlp_backbone = use_gmmlp_backbone
        self.gmmlp_feat_cache = Path(gmmlp_feat_cache) / phase if gmmlp_feat_cache else None

        self.list = [key for key, value in self.raw_data.items()]

        # Pre-load all cached ViT features into RAM so __getitem__ avoids per-sample
        # file I/O, which otherwise stalls the GPU waiting on the DataLoader.
        self._gmmlp_feats: dict[str, torch.Tensor] | None = None
        if use_gmmlp_backbone and self.gmmlp_feat_cache is not None:
            feats: dict[str, torch.Tensor] = {}
            for key in self.list:
                vid_name = key.split("/")[1] if "/" in key else key
                pt = self.gmmlp_feat_cache / f"{vid_name}.pt"
                if pt.exists():
                    feats[vid_name] = torch.load(pt, weights_only=True)
            self._gmmlp_feats = feats
            print(f"  [{phase}] pre-loaded {len(feats)}/{len(self.list)} GMMLP feature tensors")

        def sometimes(aug):
            return va.Sometimes(
                0.5,
                aug,
            )  # Used to apply augmentor with 50% probability

        self.seq = va.Sequential(
            [
                sometimes(va.RandomRotate(30)),
                sometimes(va.RandomResize(0.2)),
                sometimes(va.RandomTranslate(x=10, y=10)),
                # sometimes(va.Brightness(min=0.1, max=1.5)),
                # sometimes(va.Color(min=0.1, max=1.5)),
            ],
        )

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        descript_sample = self.descript_feat[key.split("/")[1]]["siglip2_feat"]
        tgt_sample = sample["text"]
        name_sample = sample["name"]

        img_paths = [self.img_path + x for x in sample["imgs_path"]]

        if self.use_gmmlp_backbone:
            vid_name = key.split("/")[1] if "/" in key else key
            if self._gmmlp_feats is not None and vid_name in self._gmmlp_feats:
                # Cache hit: skip frame loading entirely; apply random subsampling to feats
                vis_feats_full = self._gmmlp_feats[vid_name]  # (T_all, D_vit)
                n = len(vis_feats_full)
                if n > self.max_length:
                    selected_indices = sorted(random.sample(range(n), k=self.max_length))
                    vis_feats = vis_feats_full[selected_indices]
                    descript_sample = descript_sample[selected_indices]
                else:
                    vis_feats = vis_feats_full
                return name_sample, descript_sample, tgt_sample, None, None, vis_feats
            # No cache: fall back to PIL frames
            img_sample, selected_indices = self.load_imgs(img_paths)
            if selected_indices is not None:
                descript_sample = descript_sample[selected_indices]
            pil_frames = self.load_pil_imgs(img_paths, selected_indices)
            return name_sample, descript_sample, tgt_sample, img_sample, pil_frames, None

        img_sample, selected_indices = self.load_imgs(img_paths)
        if selected_indices is not None:
            descript_sample = descript_sample[selected_indices]
        return name_sample, descript_sample, tgt_sample, img_sample

    def load_all_pil_frames(self, paths: list) -> list:
        """Load every frame as PIL (no subsampling) for GMMLP feature extraction."""
        frames = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        return frames

    def load_pil_imgs(self, paths, selected_indices=None):
        """Load raw PIL images (no normalization) for GMMLP backbone preprocessing."""
        if selected_indices is not None:
            paths = [paths[i] for i in selected_indices]
        elif len(paths) > self.max_length:
            selected_indices = sorted(random.sample(range(len(paths)), k=self.max_length))
            paths = [paths[i] for i in selected_indices]
        frames = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        return frames

    def load_imgs(self, paths):

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
        for i, img_path in enumerate(paths):
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((self.args.input_size, self.args.input_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            batch_image.append(img)

        if self.phase == "train":
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):
            img_resized = img.resize(resize)
            img_tensor = data_transform(img_resized).unsqueeze(0)
            imgs[i, :, :, :] = img_tensor[
                :,
                :,
                crop_rect[1] : crop_rect[3],
                crop_rect[0] : crop_rect[2],
            ]

        return imgs, selected_indices

    def collate_fn(self, batch):
        # Fast path: GMMLP + full cache — skip all img/descript tensor work entirely.
        if self._gmmlp_feats is not None:
            name_batch, tgt_batch, vis_feats_batch = [], [], []
            for name_sample, _descript, tgt_sample, _img, _pil, vis_feats in batch:
                name_batch.append(name_sample)
                tgt_batch.append(tgt_sample)
                vis_feats_batch.append(vis_feats)
            tgt_input = self.tokenizer(
                text_target=tgt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            return {"vis_feats": vis_feats_batch, "name_batch": name_batch}, tgt_input

        tgt_batch, txt_tmp, src_length_batch, name_batch, img_tmp = [], [], [], [], []
        pil_frames_batch = [] if self.use_gmmlp_backbone else None
        vis_feats_batch = [] if self.use_gmmlp_backbone else None

        for item in batch:
            if self.use_gmmlp_backbone:
                name_sample, txt_sample, tgt_sample, img_sample, pil_frames, vis_feats = item
                if vis_feats is not None:
                    vis_feats_batch.append(vis_feats)
                else:
                    pil_frames_batch.append(pil_frames)
            else:
                name_sample, txt_sample, tgt_sample, img_sample = item
            name_batch.append(name_sample)

            txt_tmp.append(txt_sample)

            tgt_batch.append(tgt_sample)

            img_tmp.append(img_sample)

        max_len = max([len(vid) for vid in img_tmp])
        video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in img_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad

        padded_txt = []

        for txt in txt_tmp:
            left_padded = txt[0].unsqueeze(0).expand(left_pad, -1)
            right_pad_len = max_len - len(txt) - left_pad
            right_padded = txt[-1].unsqueeze(0).expand(right_pad_len, -1)
            padded_tmp = torch.cat((left_padded, txt, right_padded), dim=0)
            padded_txt.append(padded_tmp)

        padded_video = [
            torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                ),
                dim=0,
            )
            for vid in img_tmp
        ]

        img_tmp = [padded_video[i][0 : video_length[i], :, :, :] for i in range(len(padded_video))]
        new_txt_tmp = [padded_txt[i][0 : video_length[i], :] for i in range(len(padded_txt))]

        for i in range(len(img_tmp)):
            src_length_batch.append(len(img_tmp[i]))

        src_length_batch = torch.tensor(src_length_batch)

        txt_batch = torch.cat(new_txt_tmp, 0)
        img_batch = torch.cat(img_tmp, 0)

        new_src_lengths = (((src_length_batch - 5 + 1) / 2) - 5 + 1) / 2
        new_src_lengths = new_src_lengths.long()
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()

        tgt_input = self.tokenizer(
            text_target=tgt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        src_input = {}
        src_input["input_descript"] = txt_batch
        src_input["input_img"] = img_batch
        src_input["attention_mask"] = img_padding_mask
        src_input["name_batch"] = name_batch
        src_input["src_length_batch"] = src_length_batch
        src_input["new_src_length_batch"] = new_src_lengths
        if vis_feats_batch:
            src_input["vis_feats"] = vis_feats_batch
        elif pil_frames_batch:
            src_input["pil_frames"] = pil_frames_batch

        return src_input, tgt_input

    def __str__(self) -> str:
        return f"#total {self.phase} set: {len(self.list)}."
