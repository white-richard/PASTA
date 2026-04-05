import argparse
import pathlib
import signal
import sys
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

from datasets import MissDataset, VideoDataset
from llavaov import LLaVA

torch.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


def _checkpoint(frame_texts: dict, save_file: str) -> None:
    text_dict = {
        vid: {"texts": [frame_texts[vid][i] for i in sorted(frame_texts[vid])]}
        for vid in frame_texts
    }
    torch.save(text_dict, save_file)


def create_feature(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunk_size = args.chunk_size
    resume = args.resume
    split = args.split
    save_path = args.save_path
    save_file = f"{save_path}phoenix_SLdescript.{split}"
    debug_mode = args.debug_mode

    dataset = MissDataset(vars(args), split) if resume else VideoDataset(vars(args), split)

    # Phase 1: collect all frame paths upfront
    print("Collecting frame paths...")
    all_entries = []  # (vid_name, frame_idx, frame_path)
    for i in range(len(dataset)):
        vid_name, frame_files = dataset[i]
        for frame_idx, frame_path in enumerate(frame_files):
            all_entries.append((vid_name, frame_idx, frame_path))
        if debug_mode and i >= 3:
            break

    print(f"Total frames to process: {len(all_entries)}")

    # Phase 2: vLLM inference in large chunks for maximum GPU utilization
    mmlm = LLaVA()
    frame_texts: dict[str, dict[int, str]] = defaultdict(dict)

    for chunk_num, start in enumerate(
        tqdm(range(0, len(all_entries), chunk_size), desc="Processing chunks"),
    ):
        chunk = all_entries[start : start + chunk_size]
        images = [Image.open(path).convert("RGB") for _, _, path in chunk]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            texts = mmlm(images=images)

        for (vid_name, frame_idx, _), text in zip(chunk, texts, strict=False):
            frame_texts[vid_name][frame_idx] = text

        if (chunk_num + 1) % 10 == 0:
            _checkpoint(frame_texts, save_file)
            print(f"Checkpoint saved at chunk {chunk_num + 1}.")

    # Phase 3: assemble final dict and save
    text_dict = {
        vid: {"texts": [frame_texts[vid][i] for i in sorted(frame_texts[vid])]}
        for vid in frame_texts
    }
    torch.save(text_dict, save_file)
    print("Saving features complete!")


def cleanup() -> None:
    torch.cuda.empty_cache()
    sys.exit(0)


def signal_handler(sig, frame) -> None:
    cleanup()


signal.signal(signal.SIGINT, signal_handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        default="datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/",
        help="path to dataset folder",
    )
    parser.add_argument("--split", type=str, default="train", help="split") # train | dev | test
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="number of frames per vLLM batch (larger = better GPU utilization)",
    )
    parser.add_argument("--video_bs", type=int, default=1)  # kept for MissDataset compat
    parser.add_argument("--resume", action="store_true", help="resume generating features")
    parser.add_argument(
        "--save_path",
        type=str,
        default="tmp/features/",
        help="path to save features",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="stop after the second video",
    )
    args = parser.parse_args()

    save_path = pathlib.Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    create_feature(args)


if __name__ == "__main__":
    main()
