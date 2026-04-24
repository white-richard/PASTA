# ASL VLM Project

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

You'll need to download the video description labels from [GoogleDrive](https://drive.google.com/drive/folders/1Vymg9G7io2sGMBhyWJWCCiF65iI_qik1?usp=drive_link).

Structure the descriptions like:

```txt
datasets/text_phoenix-descript
├── phoenix_SLdescriptions.dev
├── phoenix_SLdescriptions.test
└── phoenix_SLdescriptions.train
```

### Patch `nlg-eval`

TODO: is this necessary now that we are using python 3.12?

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

### Install dependencies

TODO: resolve setup.fish into pyproject.toml

```bash
git submodule update --init --recursive
uv sync
uv pip install -e repos/gradcache
```

# Method

## Generate Description Embeddings

Before training, you must convert the raw VLM-generated text descriptions into embeddings:

```bash
bash scripts/generate_descript.bash
```

TODO: finish method

# Appendix

## Reproduce MMSLT author's results

This trains the MMLP then MMSLT using the paper's hyperparameters

Text sign descriptions and weight files from MMSLT can be found in [GoogleDrive](https://drive.google.com/drive/folders/1Vymg9G7io2sGMBhyWJWCCiF65iI_qik1?usp=drive_link)

```bash
bash scripts/author/reproduce_author.bash
```
