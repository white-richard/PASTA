# MMSLT

## Setup

**Prerequisites:** `uv` ([install](https://docs.astral.sh/uv/getting-started/installation/)) and `git`.

### 1. Download Phoenix dataset

Run this in a `tmux` session — it takes a few hours:

```bash
mkdir -p datasets
cd datasets
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz
tar xzf phoenix-2014-T.v3.tar.gz
cd ..
```

You'll need to download the video description labels from [GoogleDrive](https://drive.google.com/drive/folders/1Vymg9G7io2sGMBhyWJWCCiF65iI_qik1?usp=drive_link). Move them into this dir: `datasets/phoenix-descript`

```txt
datasets/phoenix-descript
├── phoenix_SLdescriptions.dev
├── phoenix_SLdescriptions.test
└── phoenix_SLdescriptions.train
```

CSL requires a formal request; use Phoenix for reproducibility verification.

### 2. Create virtual environment

```bash
uv venv --python 3.10
source .venv/bin/activate
```

### 3. Patch `nlg-eval`

`gensim 3.8.3` cannot be compiled on Python 3.10 due to a removed NumPy build flag.
Clone the dependency locally and relax its gensim version pin before installing:

```bash
cd MMSLT
git clone https://github.com/Maluuba/nlg-eval.git nlg-eval-temp
cd nlg-eval-temp
git checkout 2ab4528fad5548315cf61e40c2249fec8c8ad233
git checkout -b py310-patch
sed -i 's/gensim~=3.8.3/gensim>=4.0.1/' requirements.txt
git commit -am 'Relax gensim requirement to >=4.0.1 for Python 3.10 compatibility'
cd ../..
```

### 4. Install dependencies

```bash
uv pip install --upgrade pip setuptools wheel packaging ninja

uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

uv pip install --upgrade "wandb>=0.19"
```

The `requirements.txt` references `nlg-eval-temp` via a relative path.
All other packages install from PyPI or the PyTorch index without compilation.

## Installation

## Code Descriptions

### 1. **Generate BERT Description Embeddings**

Before training, you must convert the raw LLaVA-generated text descriptions into BERT embeddings. The `descript_embed.ipynb` notebook does the following:

1. Loads `phoenix_SLdescriptions.{train,dev,test}` from `datasets/phoenix-descript/` — each file is a dict of `{video_name: [list of text descriptions]}`
2. Runs each video's descriptions through `bert-base-cased`, taking the CLS token embedding as a 768-dim feature vector per description
3. Overwrites the same files with the enriched format: `{video_name: {'texts': [...], 'bert_feat': tensor}}`

```bash
sudo apt install jupyter-core
```

```bash
cd MMSLT && jupyter nbconvert --to notebook --execute --inplace descript_embed.ipynb
```

---

### 2. Reproduce author's results

Reproduce the authors results using the following bash script:
This trains the MMLP then MMSLT using the paper's hyperparameters

```bash
cd MMSLT
bash reproduce_author.bash
```

## Notes

- The `--nproc_per_node=4` flag specifies that the training will use 4 GPUs. Adjust this based on your available GPU resources. However, for exact reproducibility of the results, it is highly recommended to use 4 GPUs as specified.
- Text sign descriptions and weight files in our [GoogleDrive](https://drive.google.com/drive/folders/1Vymg9G7io2sGMBhyWJWCCiF65iI_qik1?usp=drive_link)

# DEPRICATED INSTRUCTIONS

### 2. **MMLP Training**

To train the MMLP (MultiModal Language Processing) model, run the following command:

We should be in the MMSLT dir for training:

```bash
cd MMSLT
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=1234 \
--use_env train_mmlp.py \
--batch-size 4 \
--epochs 80 \
--opt adamw \
--lr 1e-4 \
--output_dir pretrain_models/mmlp
```

---

### 3. **MMSLT Training**

To fine-tune the MMSLT (MultiModal Spoken Language Translation) model, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=1234 \
--use_env train_mmslt.py \
--batch-size 2 \
--epochs 200 \
--opt adamw \
--lr 1e-4 \
--finetune pretrain_models/mmlp/best_checkpoint.pth \
--output_dir out/mmslt
```
