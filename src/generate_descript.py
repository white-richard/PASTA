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
torch.set_float32_matmul_precision("medium")


def _checkpoint(
    frame_data: dict,
    frame_paths: dict,
    save_file: str,
    extract_hidden_states: bool,
) -> None:
    if extract_hidden_states:
        feature_dict = {
            vid: {
                "features": [frame_data[vid][i] for i in sorted(frame_data[vid])],
                "paths": [frame_paths[vid][i] for i in sorted(frame_paths[vid])],
            }
            for vid in frame_data
        }
    else:
        feature_dict = {
            vid: {
                "texts": [frame_data[vid][i] for i in sorted(frame_data[vid])],
                "paths": [frame_paths[vid][i] for i in sorted(frame_paths[vid])],
            }
            for vid in frame_data
        }
    torch.save(feature_dict, save_file)


def create_feature(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunk_size = args.chunk_size
    resume = args.resume
    split = args.split
    save_path = args.save_path
    extract_hidden_states = args.extract_hidden_states
    hidden_state_layer = args.hidden_state_layer
    debug_mode = args.debug_mode

    suffix = f"hidden_layer{hidden_state_layer}" if extract_hidden_states else "SLdescript"
    save_file = f"{save_path}phoenix_{suffix}_{split}.pt"

    dataset = MissDataset(vars(args), split) if resume else VideoDataset(vars(args), split)

    print("Collecting frame paths...")
    all_entries = []
    for i in range(len(dataset)):
        vid_name, frame_files = dataset[i]
        for frame_idx, frame_path in enumerate(frame_files):
            all_entries.append((vid_name, frame_idx, frame_path))

    print(f"Total frames to process: {len(all_entries)}")

    mmlm = LLaVA(
        extract_hidden_states=extract_hidden_states,
        hidden_state_layer=hidden_state_layer,
    )
    frame_data: dict[str, dict[int, str | torch.Tensor]] = defaultdict(dict)
    frame_paths: dict[str, dict[int, str]] = defaultdict(dict)

    for chunk_num, start in enumerate(
        tqdm(range(0, len(all_entries), chunk_size), desc="Processing chunks"),
    ):
        chunk = all_entries[start : start + chunk_size]
        images = [Image.open(path).convert("RGB") for _, _, path in chunk]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = mmlm(images=images)

        for (vid_name, frame_idx, frame_path), output in zip(chunk, outputs, strict=False):
            frame_data[vid_name][frame_idx] = output
            frame_paths[vid_name][frame_idx] = str(frame_path)

        if (chunk_num + 1) % 500 == 0:
            _checkpoint(frame_data, frame_paths, save_file, extract_hidden_states)
            print(f"Checkpoint saved at chunk {chunk_num + 1}.")

        if debug_mode and chunk_num >= 10:
            break

    _checkpoint(frame_data, frame_paths, save_file, extract_hidden_states)
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
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--video_bs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_path", type=str, default="out/descript/features/")
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument(
        "--extract-hidden-states",
        action="store_true",
        help="extract mean-pooled Qwen2 hidden states instead of generating text",
    )
    parser.add_argument(
        "--hidden-state-layer",
        type=int,
        default=20,
        help="which Qwen2 layer to extract from (0-27 for 7B)",
    )
    args = parser.parse_args()

    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)
    create_feature(args)


if __name__ == "__main__":
    main()
