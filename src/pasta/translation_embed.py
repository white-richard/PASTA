"""Create text features for dataset translations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, SiglipTextModel

from .data_io import load_translations

SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phoenix2014t.yaml",
        help="YAML config with data.label_path entries.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/phoenix-translations",
        help="Directory for generated .pt feature files.",
    )
    parser.add_argument(
        "--features",
        choices=["pooled", "tokens", "both"],
        default="both",
        help="Which SigLIP2 text representation to save.",
    )
    parser.add_argument(
        "--model-id",
        default=SIGLIP2_MODEL_ID,
        help="Hugging Face SigLIP2 text model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    label_paths = config["data"]["label_path"]
    translations = {split: load_translations(path) for split, path in label_paths.items()}
    print("  ".join(f"{split}: {len(items)}" for split, items in translations.items()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = SiglipTextModel.from_pretrained(args.model_id).to(device)
    model.eval()

    def encode(texts: list[str]) -> dict[str, torch.Tensor]:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device))
        result: dict[str, torch.Tensor] = {}
        if args.features in ("pooled", "both"):
            result["pooled"] = outputs.pooler_output.cpu()
        if args.features in ("tokens", "both"):
            result["tokens"] = outputs.last_hidden_state.cpu()
        return result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, split_translations in translations.items():
        features = defaultdict(dict)
        for name, text in tqdm(split_translations.items(), desc=split):
            features[name]["texts"] = [text]
            encoded = encode([text])
            if "pooled" in encoded:
                features[name]["siglip2_feat"] = encoded["pooled"]
            if "tokens" in encoded:
                features[name]["siglip2_token_feat"] = encoded["tokens"]
        destination = output_dir / f"phoenix_translations_siglip2_{split}.pt"
        torch.save(features, destination)
        print(f"saved: {destination}")


if __name__ == "__main__":
    main()
