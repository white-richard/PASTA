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
rm phoenix-2014-T.v3.tar.gz
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

### 2. Patch `nlg-eval`

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

### 3. Install dependencies

```bash
git submodule update --init --recursive
uv sync
uv pip install -e repos/gradcache
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
python src/descript_embed.py
```

---

### 2. Reproduce author's results

Reproduce the authors results using the following bash script:
This trains the MMLP then MMSLT using the paper's hyperparameters

```bash
chmod +x scripts/reproduce_author.bash
./scripts/reproduce_author.bash
```

## Generate Descriptions with vLLM and TurboQuant

Install dependencies in a new venv:

```bash
uv venv --python 3.12 desc-venv
bash scripts/setup_descript_env.bash
source desc-venv/bin/activate
```

Run the code:

```bash 
scripts/generate_descript_author.bash
```

this runs over each split (train, dev, test).

## Notes

- Text sign descriptions and weight files from MMSLT can be found in [GoogleDrive](https://drive.google.com/drive/folders/1Vymg9G7io2sGMBhyWJWCCiF65iI_qik1?usp=drive_link)
