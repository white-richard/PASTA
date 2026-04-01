import argparse
import csv
import glob
import os
import signal
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from llavaov import LLaVA


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class ASLVideoDataset(Dataset):
    """
    How2Sign sentence-level clip dataset for description generation.

    Returns:
        clip_id, frames_as_PIL_list
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

    def __init__(self, args_dict, split, skip_ids=None):
        self.data_root = Path(args_dict["how2sign_root"]).expanduser().resolve()
        self.split = self.SPLIT_MAP[split]
        self.max_frames = args_dict.get("max_frames", 64)
        self.frame_stride = args_dict.get("frame_stride", 1)

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
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video folder not found: {self.video_root}")

        if skip_ids is None:
            skip_ids = set()

        self.samples = []
        missing = 0

        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                clip_id = row["SENTENCE_NAME"].strip()
                mp4_path = self.video_root / f"{clip_id}.mp4"

                if clip_id in skip_ids:
                    continue

                if mp4_path.exists():
                    self.samples.append(
                        {
                            "name": clip_id,
                            "video_path": str(mp4_path),
                        }
                    )
                else:
                    missing += 1

        print(
            f"[ASLVideoDataset] split={self.split} "
            f"samples={len(self.samples)} missing_videos={missing} skipped_existing={len(skip_ids)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        clip_id = sample["name"]
        video_path = sample["video_path"]

        frames = self.load_video_frames(video_path)
        return clip_id, frames

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)

            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        # Uniform subsampling if too many frames
        if len(frames) > self.max_frames:
            idx = torch.linspace(0, len(frames) - 1, steps=self.max_frames).long().tolist()
            frames = [frames[i] for i in idx]

        return frames


class ASLMissDataset(ASLVideoDataset):
    """
    Resume-aware dataset:
    skips clip IDs already present in saved description shards.
    """

    def __init__(self, args_dict, split):
        save_path = Path(args_dict["save_path"]).expanduser().resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        existing_ids = set()
        pattern = str(save_path / f"how2sign_SLdescript.{split}_*")
        for shard in glob.glob(pattern):
            try:
                data = torch.load(shard, weights_only=False)
                existing_ids.update(data.keys())
            except Exception:
                pass

        super().__init__(args_dict, split, skip_ids=existing_ids)


def simple_collate_fn(batch):
    """
    Keep batch as a raw list of samples because each sample contains
    a variable-length list of PIL images.
    """
    return batch


def create_feature(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    batch_size = args.video_bs
    frame_batch = args.frame_bs
    resume = args.resume
    split = args.split
    save_path = Path(args.save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    if resume:
        dataset = ASLMissDataset(vars(args), split)
    else:
        dataset = ASLVideoDataset(vars(args), split)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=simple_collate_fn,
    )

    mmlm = LLaVA()
    text_dict = defaultdict(dict)
    cnt = 0

    for batch in tqdm(dataloader):
        # batch is a list of size batch_size; current logic assumes video_bs=1
        if len(batch) != 1:
            raise ValueError(
                f"Current script expects video_bs=1 for simplicity. Got batch size {len(batch)}"
            )

        vid_name, images = batch[0]

        idx = [0]
        for i in range(0, len(images), frame_batch):
            end = min(i + frame_batch, len(images))
            idx.append(end)

        vid_texts = []
        for i in range(len(idx) - 1):
            start, end = idx[i : i + 2]
            frames = images[start:end]

            texts = mmlm(images=frames)
            vid_texts.extend(texts)

        text_dict[vid_name]["texts"] = vid_texts

        cnt += 1

        if cnt % 100 == 0:
            shard_path = save_path / f"how2sign_SLdescript.{split}_{rank}"
            torch.save(text_dict, shard_path)
            print(f"[rank {rank}] Saved partial descriptions at iteration {cnt} -> {shard_path}")

    shard_path = save_path / f"how2sign_SLdescript.{split}_{rank}"
    torch.save(text_dict, shard_path)
    print(f"[rank {rank}] Saving complete -> {shard_path}")


def cleanup():
    torch.cuda.empty_cache()
    sys.exit(0)


def signal_handler(sig, frame):
    cleanup()


signal.signal(signal.SIGINT, signal_handler)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--how2sign_root",
        type=str,
        required=True,
        help="Root folder of the uploaded how2sign-data directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "dev", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--frame_bs",
        type=int,
        default=8,
        help="Batch size of frames for each LLaVA call",
    )
    parser.add_argument(
        "--video_bs",
        type=int,
        default=1,
        help="Number of videos per dataloader batch; keep this at 1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generating features by skipping already-saved clips",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Folder to save generated description shards",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=64,
        help="Maximum number of frames to keep per clip after uniform subsampling",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Decode every Nth frame from the clip",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers",
    )

    args = parser.parse_args()
    create_feature(args)


if __name__ == "__main__":
    main()