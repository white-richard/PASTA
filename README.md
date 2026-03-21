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
git clone https://github.com/Maluuba/nlg-eval.git src/nlg-eval-temp
cd src/nlg-eval-temp
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

uv pip install -r requirements.txt

uv pip install --upgrade "wandb>=0.19"

git submodule update --init --recursive
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
jupyter nbconvert --to notebook --execute --inplace src/descript_embed.ipynb
```

---

### 2. Reproduce author's results

Reproduce the authors results using the following bash script:
This trains the MMLP then MMSLT using the paper's hyperparameters

```bash
chmod +x scripts/reproduce_author.bash
./scripts/reproduce_author.bash
```

## Notes

- Text sign descriptions and weight files from MMSLT can be found in [GoogleDrive](https://drive.google.com/drive/folders/1Vymg9G7io2sGMBhyWJWCCiF65iI_qik1?usp=drive_link)
