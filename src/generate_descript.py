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
torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision("medium")


def _checkpoint(
    frame_data: dict,
    frame_paths: dict,
    save_file: str,
    extract_hidden_states: bool,
) -> None:
    if extract_hidden_states:
        feature_dict = {}
        for vid in frame_data:
            sorted_idx = sorted(frame_data[vid])
            samples = [frame_data[vid][i] for i in sorted_idx]
            first = samples[0]
            if isinstance(first, dict):
                # multi-layer dict format: {"layers": {idx: tensor}, "text": str}
                layer_indices = list(first["layers"].keys())
                vid_dict: dict = {
                    "layers": {
                        layer_idx: [s["layers"][layer_idx] for s in samples]
                        for layer_idx in layer_indices
                    },
                    "paths": [frame_paths[vid][i] for i in sorted_idx],
                }
                if "text" in first:
                    vid_dict["texts"] = [s["text"] for s in samples]
            else:
                # tuple format (mid_tensor, last_tensor) for LLaVA.
                vid_dict = {
                    "features": [s[0] for s in samples],
                    "features_last": [s[1] for s in samples],
                    "paths": [frame_paths[vid][i] for i in sorted_idx],
                }
            feature_dict[vid] = vid_dict
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
    video_bs = args.video_bs
    resume = args.resume
    split = args.split
    save_path = args.save_path
    extract_hidden_states = args.extract_hidden_states
    hidden_state_layers = args.hidden_state_layers
    save_text = args.save_text
    debug = args.debug

    if extract_hidden_states:
        layers_str = "_".join(str(l) for l in hidden_state_layers)
        suffix = f"hidden_layers{layers_str}" + ("_text" if save_text else "")
    else:
        suffix = "SLdescript"
    save_file = save_path / f"phoenix_{suffix}_{split}.pt"

    if debug:
        video_bs = 1

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
            hidden_state_layer=hidden_state_layers[0],
        )
    else:  # gemma4
        from gemma4 import Gemma4

        mmlm = Gemma4(
            model_id=model_id,
            hf_model_id=args.hf_model_id or model_id,
            extract_hidden_states=extract_hidden_states,
            hidden_state_layers=hidden_state_layers,
            save_text=save_text,
        )
    frame_data: dict[str, dict[int, str | torch.Tensor]] = defaultdict(dict)
    frame_paths: dict[str, dict[int, str]] = defaultdict(dict)

    n_batches = -(-len(all_entries) // video_bs)  # ceil div
    for batch_num, start in enumerate(
        tqdm(range(0, len(all_entries), video_bs), total=n_batches, desc="Processing batches"),
    ):
        batch = all_entries[start : start + video_bs]
        images = [Image.open(path).convert("RGB") for _, _, path in batch]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = mmlm(images=images)

        for (vid_name, frame_idx, frame_path), output in zip(batch, outputs, strict=False):
            frame_data[vid_name][frame_idx] = output
            frame_paths[vid_name][frame_idx] = str(frame_path)

        frames_done = start + len(batch)
        if frames_done % chunk_size < video_bs:
            _checkpoint(frame_data, frame_paths, save_file, extract_hidden_states)
            print(f"Checkpoint saved at frame {frames_done}.")

        if debug and batch_num >= 1:
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
    parser.add_argument("--save_path", type=str, default="out/text_descript")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--extract-hidden-states",
        action="store_true",
        help="extract mean-pooled LLM hidden states instead of (or alongside) generating text",
    )
    parser.add_argument(
        "--hidden-state-layers",
        type=int,
        nargs="+",
        default=[20],
        help="which LLM layers to extract from; accepts multiple values (e.g. --hidden-state-layers 10 20 27)",
    )
    parser.add_argument(
        "--save-text",
        action="store_true",
        help="when extracting hidden states, also save the text descriptions (gemma4 only)",
    )
    args = parser.parse_args()

    args.save_path = pathlib.Path(args.save_path)
    args.save_path.mkdir(parents=True, exist_ok=True)
    print(f"Features will be saved to: {args.save_path}")
    create_feature(args)


if __name__ == "__main__":
    main()
