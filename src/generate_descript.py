import argparse
import pathlib
import signal
import sys
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

from datasets import MissDataset, VideoDataset

torch.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)  # keep as fallback; some models need it
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
                "features": [frame_data[vid][i][0] for i in sorted(frame_data[vid])],
                "features_last": [frame_data[vid][i][1] for i in sorted(frame_data[vid])],
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
    save_file = save_path / f"phoenix_{suffix}_{split}.pt"

    dataset = MissDataset(vars(args), split) if resume else VideoDataset(vars(args), split)

    print("Collecting frame paths...")
    all_entries = []
    for i in range(len(dataset)):
        vid_name, frame_files = dataset[i]
        for frame_idx, frame_path in enumerate(frame_files):
            all_entries.append((vid_name, frame_idx, frame_path))

    print(f"Total frames to process: {len(all_entries)}")

    _FAMILY_DEFAULT_ID = {
        "llava": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "gemma4": "google/gemma-4-4b-it",
    }
    model_id = args.model_id or _FAMILY_DEFAULT_ID[args.model_family]

    if args.model_family == "llava":
        from llavaov import LLaVA

        mmlm = LLaVA(
            model_id=model_id,
            extract_hidden_states=extract_hidden_states,
            hidden_state_layer=hidden_state_layer,
        )
    else:  # gemma4
        from gemma4 import Gemma4

        mmlm = Gemma4(
            model_id=model_id,
            hf_model_id=args.hf_model_id or model_id,
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
    parser.add_argument(
        "--model_family",
        choices=["llava", "gemma4"],
        default="llava",
        help=(
            "MLLM backbone for frame description / hidden-state extraction. "
            "'llava' (default): LLaVA-OneVision. 'gemma4': Gemma 4."
        ),
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help=(
            "HuggingFace model ID. Defaults to "
            "'llava-hf/llava-onevision-qwen2-7b-ov-hf' for llava and "
            "'google/gemma-4-E2B-it' for gemma4."
        ),
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="",
        help=(
            "HuggingFace model ID for hidden-state extraction (gemma4 only). "
            "Must be a standard HF repo (not GGUF) loadable by transformers. "
            "Defaults to --model_id. Example: 'google/gemma-4-E2B-it'."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--video_bs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_path", type=str, default="out/descript/features/")
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument(
        "--extract-hidden-states",
        action="store_true",
        help="extract mean-pooled LLM hidden states instead of generating text",
    )
    parser.add_argument(
        "--hidden-state-layer",
        type=int,
        default=20,
        help="which LLM layer to extract from (0-27 for 7B)",
    )
    args = parser.parse_args()

    args.save_path = pathlib.Path(args.save_path)
    args.save_path.mkdir(parents=True, exist_ok=True)
    print(f"Features will be saved to: {args.save_path}")
    create_feature(args)


if __name__ == "__main__":
    main()
