# MMSLT

## Setup (Python 3.10 + CUDA 12.1)

**Prerequisites:** `uv` ([install](https://docs.astral.sh/uv/getting-started/installation/)), CUDA 12.1 toolkit, and `git`.

### 1. Download Phoenix dataset

Run this in a `tmux` session — it takes a few hours:

```bash
mkdir -p datasets
cd datasets
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz
tar xzf phoenix-2014-T.v3.tar.gz
cd ..
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
cd ..
```

### 4. Install PyTorch (CUDA 12.1) and Flash Attention

```bash
uv pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 \
    packaging ninja setuptools wheel \
    --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

> **Do not upgrade `torch` or `flash-attn` after this step.**

### 5. Install remaining dependencies

```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

The `requirements.txt` references `nlg-eval-temp` via a relative path.
All other packages install from PyPI or the PyTorch index without compilation.

### Verify key package versions

```bash
uv pip freeze | grep -E 'torch==|nltk==|keras==|h11==|gensim=='
# Expected:
#   gensim==4.4.0   (or newer)
#   h11==0.16.0     (or newer)
#   keras==3.x.x    (>= 2.13.1)
#   nltk==3.9.3     (or newer)
#   torch==2.4.1+cu121
```

## Installation

## Code Descriptions

### 1. **MMLP Training**

  
To train the MMLP (MultiModal Language Processing) model, run the following command:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    
    --nproc_per_node=4 \
    --master_port=1234 \
    --use_env train_mmlp.py \
    --batch-size 4 \
    --epochs 80 \
    --opt adamw \ 
    --lr 1e-4 \ 
    --output_dir /path/to/your/path/pretrain_models/mmlp

- Replace `/path/to/your/path/pretrain_models/mmlp` with the directory path where you want to save the MMLP model checkpoints and outputs.

----------

### 2. **MMSLT Training**

To fine-tune the MMSLT (MultiModal Spoken Language Translation) model, run the following command:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=1234 \
    --use_env train_mmslt.py \
    --batch-size 2 \
    --epochs 200 \
    --opt adamw \
    --lr 1e-4 \
    --finetune /path/to/your/path/pretrain_models/mmlp/best_checkpoint.pth \
    --output_dir /path/to/your/path/out/mmslt

- Replace `/path/to/your/path/pretrain_models/mmlp/best_checkpoint.pth` with the path to the best checkpoint from the MMLP training process.

- Replace `/path/to/your/path/out/mmslt` with the directory path where you want to save the MMSLT model checkpoints and outputs.

## Notes
  
- The `--nproc_per_node=4` flag specifies that the training will use 4 GPUs. Adjust this based on your available GPU resources. However, for exact reproducibility of the results, it is highly recommended to use 4 GPUs as specified.
- Text sign descriptions and weight files in our [GoogleDrive](https://drive.google.com/drive/folders/1Vymg9G7io2sGMBhyWJWCCiF65iI_qik1?usp=drive_link)
