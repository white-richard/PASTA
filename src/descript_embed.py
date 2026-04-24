import argparse
import os
import pathlib
from collections import defaultdict

import torch
from tqdm import tqdm

os.environ["USE_TF"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--encoder",
    choices=["bert", "siglip", "siglip2"],
    default="bert",
    help=(
        "Text encoder to use: "
        "'bert' (bert-base-cased), "
        "'siglip' (google/siglip-so400m-patch14-384), "
        "'siglip2' (google/siglip2-so400m-patch14-384)."
    ),
)
args = parser.parse_args()

path = pathlib.Path("datasets/phoenix-descript")

train = torch.load(path / "phoenix_SLdescriptions.train", weights_only=False)
dev = torch.load(path / "phoenix_SLdescriptions.dev", weights_only=False)
test = torch.load(path / "phoenix_SLdescriptions.test", weights_only=False)
raise "Need to update these to the correct path. TODO later"

print(len(train))
print(len(dev))
print(len(test))

device = torch.device("cuda:0")

if args.encoder == "siglip":
    from transformers import SiglipTextModel, SiglipTokenizer

    SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
    tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_MODEL_ID)
    model = SiglipTextModel.from_pretrained(SIGLIP_MODEL_ID)
    feat_key = "siglip_feat"

    def encode(texts):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device))
        return outputs.pooler_output.cpu()

elif args.encoder == "siglip2":
    from transformers import AutoTokenizer, SiglipTextModel

    SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"
    tokenizer = AutoTokenizer.from_pretrained(SIGLIP2_MODEL_ID)
    model = SiglipTextModel.from_pretrained(SIGLIP2_MODEL_ID)
    feat_key = "siglip2_feat"

    def encode(texts):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device))
        return outputs.pooler_output.cpu()

else:
    from transformers import BertModel, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")
    feat_key = "bert_feat"

    def encode(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
        return outputs.last_hidden_state[:, 0, :].cpu()


model.to(device)

new_train = defaultdict(dict)
new_dev = defaultdict(dict)
new_test = defaultdict(dict)

for k, v in tqdm(train.items()):
    new_train[k]["texts"] = v
    new_train[k][feat_key] = encode(v)

for k, v in tqdm(dev.items()):
    new_dev[k]["texts"] = v
    new_dev[k][feat_key] = encode(v)

for k, v in tqdm(test.items()):
    new_test[k]["texts"] = v
    new_test[k][feat_key] = encode(v)

path = pathlib.Path("out") / path
path.mkdir(parents=True, exist_ok=True)
torch.save(new_train, path / f"phoenix_SLdescriptions_{args.encoder}_train.pt")
torch.save(new_dev, path / f"phoenix_SLdescriptions_{args.encoder}_dev.pt")
torch.save(new_test, path / f"phoenix_SLdescriptions_{args.encoder}_test.pt")
