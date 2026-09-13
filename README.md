# PASTA

PASTA (Perceiver-Aligned Sign-to-Text Architecture) is a pipeline for gloss-free sign language translation on RWTH-PHOENIX-Weather 2014T. The current implementation extracts frame-level visual features, aligns video representations with German translation embeddings during PPASTA pretraining, and then finetunes a sequence-to-sequence translation model.

## Pipeline

```text
Phoenix video frames
        |
        v
Gemma 4 vision tower
        |
        +--> cached frame features ------------------+
                                                     |
German translations --> text embeddings              |
        |                                            |
        +-----------------> PPASTA <-----------------+
                              |
                              v
                    Perceiver visual encoder
                              |
                              v
                         PASTA training
                              |
                              v
                    German text translation
```

PPASTA uses a FILIP-style contrastive objective between Perceiver video tokens and precomputed SigLIP2 translation-token features. PASTA then loads the pretrained visual encoder and projects its latent tokens into the language decoder's embedding space. The supported training path uses Gemma 4 but retains an mBART decoder option.

## Repository layout

```text
configs/                  Phoenix dataset configuration
scripts/                  Supported entry points
src/pasta/                Installable PASTA package
tests/                    Data and repository tests
```

## Environment

The checked-in environment targets Python 3.12, PyTorch 2.6, and the CUDA 12.4 PyTorch wheel index. Training is intended for NVIDIA GPUs.

Install [uv](https://docs.astral.sh/uv/) and clone the repository with its GradCache submodule:

```bash
git submodule update --init --recursive
uv sync --extra dev
uv pip install -e repos/gradcache
```

### Dataset

Download RWTH-PHOENIX-Weather 2014T into `datasets/`:

```bash
mkdir -p datasets
cd datasets
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz
tar xzf phoenix-2014-T.v3.tar.gz
rm phoenix-2014-T.v3.tar.gz
cd ..
```

The expected annotation and frame paths are defined in `configs/phoenix2014t.yaml`.

## Reproduce training

Run the stages in order. Model IDs, batch sizes, GPU selection, and output paths are near the top of each shell script.

### Extract visual features

```bash
bash scripts/extract_vision_feats.bash
```

The extraction script shards each Phoenix split across the visible GPUs and merges the results after each split.

Output:

```text
datasets/phoenix-vision_feats_pooled/A4B_features/features_{train,dev,test}/
```

### Embed the reference translations

```bash
bash scripts/translation_embed.bash
```

This creates SigLIP2 pooled and token-level embeddings for the German reference translations.

Output:

```text
datasets/phoenix-translations/phoenix_translations_siglip2_{train,dev,test}.pt
```

### Pretrain PPASTA

```bash
bash scripts/train_ppasta.bash
```

PPASTA trains the Perceiver alignment stage against the translation embeddings. The default script consumes the cached visual features from [step 1](<README#Extract visual features>) instead of running the vision tower again.

Output:

```text
out/ppasta/best_checkpoint.pth
```

### Train PASTA

```bash
bash scripts/train_pasta.bash
```

The finetuning script loads the PPASTA checkpoint and cached visual features, then trains the sign-to-text translation model. Set `NUM_GPUS` in the script to select the single- or two-GPU launch path.

Output:

```text
out/pasta/best_checkpoint.pth
```

To evaluate an existing PASTA checkpoint without starting a training run:

```bash
bash scripts/train_pasta.bash --test-checkpoint out/pasta/best_checkpoint.pth
```

Evaluation reports BLEU-1 through BLEU-4, ROUGE-L, generation time per video, and real-time factor.

## Current scope

The maintained path is Phoenix-2014T only. How2Sign/ASL experiments and generated video descriptions are not part of the current pipeline.

## Background and credit

PASTA started from the MMSLT project by Jeon et al. and keeps some code adapted from that repository alongside the PASTA/PPASTA experiments:

> H. Jeon et al., **Leveraging the Power of MLLMs for Gloss-Free Sign Language Translation**, ICCV 2025.  
> https://github.com/hwjeon98/MMSLT

If you build on this repository, please cite the original MMSLT work.
