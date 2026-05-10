import argparse
import csv
import pathlib
from collections import defaultdict

import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="src/configs/config_mmslt_phoenix.yaml",
    help="Path to YAML config with data.label_path entries.",
)
parser.add_argument(
    "--output_dir",
    default="datasets/phoenix-translations",
    help="Directory to write output .pt files.",
)
parser.add_argument(
    "--features",
    choices=["pooled", "tokens", "both"],
    default="both",
    help="'pooled': store (1, D) pooler_output; 'tokens': store (1, L, D) last_hidden_state; 'both': store both.",
)
args = parser.parse_args()

import yaml

with open(args.config, encoding="utf-8") as f:
    config = yaml.safe_load(f)

label_paths = config["data"]["label_path"]


def load_translations(csv_path):
    """Return {name: translation_string} from a Phoenix CSV."""
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            result[row["name"].strip()] = row["translation"].strip()
    return result


train_translations = load_translations(label_paths["train"])
dev_translations = load_translations(label_paths["dev"])
test_translations = load_translations(label_paths["test"])

print(
    f"train: {len(train_translations)}  dev: {len(dev_translations)}  test: {len(test_translations)}",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, SiglipTextModel

SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"
tokenizer = AutoTokenizer.from_pretrained(SIGLIP2_MODEL_ID)
model = SiglipTextModel.from_pretrained(SIGLIP2_MODEL_ID).to(device)
model.eval()


def encode(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"].to(device))
    result = {}
    if args.features in ("pooled", "both"):
        result["pooled"] = outputs.pooler_output.cpu()
    if args.features in ("tokens", "both"):
        result["tokens"] = outputs.last_hidden_state.cpu()
    return result


def build_split(translations):
    result = defaultdict(dict)
    for name, text in tqdm(translations.items()):
        result[name]["texts"] = [text]
        feats = encode([text])
        if "pooled" in feats:
            result[name]["siglip2_feat"] = feats["pooled"]
        if "tokens" in feats:
            result[name]["siglip2_token_feat"] = feats["tokens"]
    return result


new_train = build_split(train_translations)
new_dev = build_split(dev_translations)
new_test = build_split(test_translations)

out_path = pathlib.Path(args.output_dir)
out_path.mkdir(parents=True, exist_ok=True)

torch.save(new_train, out_path / "phoenix_translations_siglip2_train.pt")
torch.save(new_dev, out_path / "phoenix_translations_siglip2_dev.pt")
torch.save(new_test, out_path / "phoenix_translations_siglip2_test.pt")

print(f"Saved to {out_path}")
