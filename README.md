# Serving PASTA: Learning to Translate Sign Language in 85 Million Forward Passes

Implementation for Perceiver-Aligned Sign-to-Text Architecture (PASTA),
a video-to-text sign language translation framework which maps sign videos to
spoken language sentences.

## Setup

**Prerequisites:** `uv` ([install](https://docs.astral.sh/uv/getting-started/installation/)) and `git`.

### Download Phoenix dataset

Run this in a `tmux` session — it takes a few hours:

```bash
mkdir -p datasets
cd datasets
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz
tar xzf phoenix-2014-T.v3.tar.gz
rm phoenix-2014-T.v3.tar.gz
cd ..
```

> **Note:** Video description labels are available on [GoogleDrive](https://drive.google.com/drive/folders/1Vymg9G7io2sGMBhyWJWCCiF65iI_qik1?usp=drive_link), but are not used in the current iteration of PASTA and do not need to be downloaded.

### Download How2Sign Dataset

> **Note:** How2Sign integration is planned for future work; download is not necessary.

If you do want to prepare it, download train/validation/test splits from [how2sign.github.io](https://how2sign.github.io/) — specifically the **Green Screen RGB clips (frontal view)** and **English Translation (manually re-aligned)** — and place them under `datasets/how2sign/`.

### Install dependencies

```bash
git submodule update --init --recursive
uv sync
uv pip install -e repos/gradcache
```

---

## Reproducing PASTA

The pipeline runs in four sequential stages. Each stage saves its outputs to a known path so the next stage can pick them up.

### Step 1 — Extract vision features

```bash
bash scripts/extract_vision_feats.bash
```

Runs the SigLIP2 ViT (backed by Gemma-4) over every frame in the Phoenix train/dev/test splits.
Extraction is sharded across all available GPUs in parallel; shards are merged automatically at the end of each split.

**Output:** `datasets/phoenix-vision_feats_pooled/A4B_features/`

---

### Step 2 — Embed translations

```bash
bash scripts/translation_embed.bash
```

Encodes the ground-truth German translation sentences using SigLIP2's text encoder.
These embeddings are used as the text side of the FILIP contrastive objective during PPASTA pretraining.

**Output:** `datasets/phoenix-translations/`

---

### Step 3 — PPASTA pretraining

```bash
bash scripts/train_ppasta.bash
```

Trains the Perceiver cross-attention module and a LoRA adapter on the vision encoder using a FILIP-style contrastive loss between the pooled video features (Step 1) and the translation embeddings (Step 2).
This stage aligns the visual representation space with the language decoder's embedding space before full sequence-to-sequence training.

**Output:** `out/ppasta/best_checkpoint.pth`

---

### Step 4 — PASTA training

```bash
bash scripts/train_pasta.bash
```

Fine-tunes the full sign-to-text model end-to-end.
The pretrained PPASTA backbone (Step 3) initialises the SigLIP2 ViT and Perceiver.
A frozen Gemma-4 (or mBART) language decoder is conditioned on the Perceiver output tokens to generate German translations.
Supports single- and dual-GPU training via `NUM_GPUS` in the script.

**Output:** `out/pasta/best_checkpoint.pth`

---

## Optional: Description generation (not required)

These steps generate and embed VLM-produced visual descriptions of the signing videos.
They are **not used in the current iteration of PASTA** and can be skipped.

### Generate descriptions

```bash
bash scripts/generate_descript.bash
```

Runs a vision-language model (Gemma-4) over each video to produce natural-language descriptions of the signing content.
Descriptions and their hidden-state embeddings are saved to `tmp/`.

### Embed descriptions

```bash
bash scripts/descript_embed.bash
```

Encodes the generated descriptions with SigLIP2, producing fixed-size embeddings for downstream use.

**Output:** `out/datasets/`
