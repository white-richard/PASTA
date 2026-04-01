import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

import utils
from definition import *

try:
    from vidaug import augmentors as va
except ImportError:
    va = None


class S2T_ASLDataset(Dataset):
    """
    How2Sign dataset loader for sentence-level MP4 clips.

    Keeps the same sample and batch contract as the original S2T_Dataset:
        __getitem__ -> (name_sample, descript_sample, tgt_sample, img_sample)

    collate_fn returns:
        src_input = {
            "input_descript": ...,
            "input_img": ...,
            "attention_mask": ...,
            "name_batch": ...,
            "src_length_batch": ...,
            "new_src_length_batch": ...
        }
        tgt_input = tokenizer(...)
    """

    SPLIT_MAP = {
        "train": "train",
        "dev": "val",
        "val": "val",
        "test": "test",
    }

    CSV_NAME_MAP = {
        "train": "how2sign_train.csv",
        "val": "how2sign_val.csv",
        "test": "how2sign_test.csv",
    }

    VIDEO_DIR_MAP = {
        "train": "train_rgb_front_clips",
        "val": "val_rgb_front_clips",
        "test": "test_rgb_front_clips",
    }

    def __init__(self, path, tokenizer, config, args, phase):
        self.config = config
        self.args = args
        self.tokenizer = tokenizer
        self.phase = phase

        if phase not in self.SPLIT_MAP:
            raise ValueError(f"Unsupported phase '{phase}'. Expected one of {list(self.SPLIT_MAP.keys())}")

        self.split = self.SPLIT_MAP[phase]
        self.data_root = Path(config["data"]["how2sign_root"]).expanduser().resolve()
        self.max_length = int(config["data"]["max_length"])
        self.descript_dim = int(config["data"].get("descript_dim", 768))

        self.csv_path = (
            self.data_root
            / "How2Sign"
            / "sentence_level"
            / self.split
            / "text"
            / "en"
            / "raw_text"
            / self.CSV_NAME_MAP[self.split]
        )

        self.video_root = self.data_root / self.VIDEO_DIR_MAP[self.split] / "raw_videos"

        if not self.csv_path.exists():
            raise FileNotFoundError(f"How2Sign CSV not found: {self.csv_path}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"How2Sign video folder not found: {self.video_root}")

        self.descript_feat = None
        descript_feat_path = config["data"].get("descript_feat_path", None)
        if descript_feat_path is not None:
            if isinstance(descript_feat_path, dict) and phase in descript_feat_path:
                feat_path = descript_feat_path[phase]
                if feat_path and os.path.exists(feat_path):
                    self.descript_feat = torch.load(feat_path, weights_only=False)
            elif isinstance(descript_feat_path, str) and os.path.exists(descript_feat_path):
                self.descript_feat = torch.load(descript_feat_path, weights_only=False)

        self.samples, self.missing_videos = self._load_csv(self.csv_path)

        if va is not None:
            sometimes = lambda aug: va.Sometimes(0.5, aug)
            self.seq = va.Sequential(
                [
                    sometimes(va.RandomRotate(30)),
                    sometimes(va.RandomResize(0.2)),
                    sometimes(va.RandomTranslate(x=10, y=10)),
                ]
            )
        else:
            self.seq = None

        print(
            f"[S2T_ASLDataset] phase={self.phase} split={self.split} "
            f"samples={len(self.samples)} missing_videos={self.missing_videos}"
        )

    def _load_csv(self, csv_path: Path):
        samples = []
        missing_videos = 0

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                clip_id = row["SENTENCE_NAME"].strip()
                text = row["SENTENCE"].strip()
                mp4_path = self.video_root / f"{clip_id}.mp4"

                if mp4_path.exists():
                    samples.append(
                        {
                            "name": clip_id,
                            "text": text,
                            "video_path": str(mp4_path),
                        }
                    )
                else:
                    missing_videos += 1

        return samples, missing_videos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        name_sample = sample["name"]
        tgt_sample = sample["text"]
        video_path = sample["video_path"]

        img_sample, selected_indices = self.load_video(video_path)

        if self.descript_feat is not None and name_sample in self.descript_feat:
            feat = self.descript_feat[name_sample]["bert_feat"]
            if not torch.is_tensor(feat):
                feat = torch.tensor(feat, dtype=torch.float32)

            if len(feat) >= len(selected_indices):
                feat = feat[selected_indices]
            else:
                feat = torch.zeros(len(selected_indices), self.descript_dim, dtype=torch.float32)

            descript_sample = feat.float()
        else:
            descript_sample = torch.zeros(len(selected_indices), self.descript_dim, dtype=torch.float32)

        return name_sample, descript_sample, tgt_sample, img_sample

    def load_video(self, video_path):
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        crop_rect, resize = utils.data_augmentation(
            resize=(self.args.resize, self.args.resize),
            crop_size=self.args.input_size,
            is_train=(self.phase == "train"),
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        selected_indices = list(range(len(frames)))
        if len(frames) > self.max_length:
            selected_indices = sorted(random.sample(range(len(frames)), k=self.max_length))
            frames = [frames[i] for i in selected_indices]

        if self.phase == "train" and self.seq is not None:
            frames = self.seq(frames)

        imgs = torch.zeros(len(frames), 3, self.args.input_size, self.args.input_size)

        for i, img in enumerate(frames):
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
        tgt_batch, txt_tmp, src_length_batch, name_batch, img_tmp = [], [], [], [], []

        for name_sample, txt_sample, tgt_sample, img_sample in batch:
            name_batch.append(name_sample)
            txt_tmp.append(txt_sample)
            tgt_batch.append(tgt_sample)
            img_tmp.append(img_sample)

        max_len = max(len(vid) for vid in img_tmp)
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

        img_tmp = [padded_video[i][0:video_length[i], :, :, :] for i in range(len(padded_video))]
        new_txt_tmp = [padded_txt[i][0:video_length[i], :] for i in range(len(padded_txt))]

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

        src_input = {}
        src_input["input_descript"] = txt_batch
        src_input["input_img"] = img_batch
        src_input["attention_mask"] = img_padding_mask
        src_input["name_batch"] = name_batch
        src_input["src_length_batch"] = src_length_batch
        src_input["new_src_length_batch"] = new_src_lengths

        tgt_input = self.tokenizer(
            text_target=tgt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return src_input, tgt_input

    def __str__(self):
        return (
            f"#total {self.phase} set: {len(self.samples)}. "
            f"(split={self.split}, missing_videos={self.missing_videos})"
        )