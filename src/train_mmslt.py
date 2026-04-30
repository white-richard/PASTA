import argparse
import datetime
import gc
import json
import os
import random
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import hpargparse
import numpy as np
import torch
import wandb
import yaml
from hpman.m import _
from loguru import logger
from rouge_score import rouge_scorer as _rouge_module
from sacrebleu.metrics import BLEU
from timm.optim import create_optimizer
from torch import nn
from torch.backends import cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    MBart50TokenizerFast,
)

import utils
from datasets import S2T_Dataset
from definition import *
from models import MMSLT

try:
    from nlgeval import compute_metrics
except Exception:
    compute_metrics = None
    logger.warning("nlgeval is not installed; extra eval metrics will be skipped.")

try:
    import psutil
except ImportError:
    psutil = None


def get_args_parser():
    parser = argparse.ArgumentParser(
        "LLaVA-guided Sign Language Translation script",
        add_help=False,
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=80, type=int)

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    # * Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1.0e-09,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1.0e-09)",
    )
    parser.add_argument(
        "--opt-betas",
        default=[0.9, 0.98],
        type=float,
        nargs="+",
        metavar="BETA",  # [0.9, 0.98]
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,  # 0.001 is original
        help="weight decay (default: 0.05)",
    )

    # * Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--no-lr-scheduler",
        action="store_true",
        help="Disable learning rate scheduler.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1.0e-08,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # * Baise params
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--eval-metrics",
        action="store_true",
        help="Compute extra BLEU-1/2/3 and ROUGE metrics during evaluation.",
    )
    parser.add_argument(
        "--skip_val",
        action="store_true",
        help="Skip validation on the dev set during training.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        metavar="N",
        help="Run dev evaluation every N epochs (default: 1).",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=80,
        metavar="N",
        help="Max tokens to generate during BLEU evaluation (default: 80; Phoenix avg is ~10 words).",
    )
    parser.add_argument(
        "--eval-num-beams",
        type=int,
        default=4,
        metavar="N",
        help="Beam width for mbart generation during evaluation (default: 4; ignored for gemma4 greedy).",
    )

    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--eval_num_workers",
        default=1,
        type=int,
        help="Number of DataLoader workers for dev/test evaluation dataloaders. ",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--config", type=str, default="src/configs/config_mmslt_phoenix.yaml")
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log process/CUDA memory periodically.",
    )
    parser.add_argument(
        "--language_decoder",
        default="mbart",
        choices=["mbart", "gemma4"],
        help="Language decoder backend: mbart (default) or gemma4 (LoRA on global MLP expert + attention)",
    )
    parser.add_argument(
        "--gemma4_model_id",
        default="google/gemma-4-E2B-it",
        help="HuggingFace model ID for Gemma4 when --language_decoder=gemma4",
    )

    # *Drop out params
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # * data process params
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default="resnet18",
        help="Vision vision_backbone name. Use 'dummy' for a tiny random-weight model.",
    )

    # * visualization
    parser.add_argument("--visualize", action="store_true")

    # * debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode: only 1 epoch and 2 batches per split.",
    )

    # * GMMLP pretrained vision encoder
    parser.add_argument(
        "--gmmlp_checkpoint",
        default="",
        help="Path to a GMMLP checkpoint (.pth) to use as the pretrained vision encoder. "
        "When set, replaces the standard vision backbone with the GMMLP SigLIP2 ViT + "
        "Perceiver encoder. The dataset will supply raw PIL frames for preprocessing.",
    )
    parser.add_argument(
        "--gmmlp_model_id",
        default="google/gemma-4-E2B-it",
        help="HuggingFace model ID used when training the GMMLP checkpoint (needed to "
        "reconstruct the vision tower architecture).",
    )
    parser.add_argument(
        "--gmmlp_model_family",
        default="gemma4",
        choices=["llava", "gemma4"],
        help="Model family of the GMMLP checkpoint ('gemma4' or 'llava').",
    )
    parser.add_argument("--gmmlp_lora_r", type=int, default=16)
    parser.add_argument("--gmmlp_lora_alpha", type=int, default=32)
    parser.add_argument("--gmmlp_lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--gmmlp_num_latents",
        type=int,
        default=64,
        help="Perceiver num_latents (K) used in the GMMLP checkpoint.",
    )
    parser.add_argument(
        "--gmmlp_num_media_embeds",
        type=int,
        default=512,
        help="Perceiver num_media_embeds used in the GMMLP checkpoint.",
    )
    parser.add_argument(
        "--gmmlp_vision_chunk_size",
        type=int,
        default=8,
        help="Frames processed per ViT chunk (matches GMMLP training value).",
    )
    parser.add_argument(
        "--gmmlp_feat_cache",
        default="",
        help="Directory of pre-extracted GMMLP ViT features produced by extract_vision_feats.py. "
        "Each video must be saved as {cache}/{split}/{video_name}.pt containing a "
        "(T, P, D_vit) patch tensor. When set, the ViT is not loaded and only the "
        "Perceiver runs during training.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Nominal video frame rate used to convert frame counts to seconds for timing metrics.",
    )

    return parser




def main(args, config) -> None:
    args.distributed = False
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset:")
    if args.language_decoder == "gemma4":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.gemma4_model_id)
    else:
        tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt",
            src_lang="de_DE",
            tgt_lang="de_DE",
            model_max_length=1024,
        )

    use_gmmlp_backbone = bool(args.gmmlp_checkpoint)
    dataset_kwargs = {
        "path": config["data"]["label_path"],
        "tokenizer": tokenizer,
        "config": config,
        "args": args,
        "use_gmmlp_backbone": use_gmmlp_backbone,
        "gmmlp_feat_cache": args.gmmlp_feat_cache or None,
    }

    train_data = S2T_Dataset(phase="train", **dataset_kwargs)
    print(train_data)

    dev_data = S2T_Dataset(phase="dev", **dataset_kwargs)
    print(dev_data)

    test_data = S2T_Dataset(phase="test", **dataset_kwargs)
    print(test_data)

    pin_mem = bool(args.pin_mem and args.device.startswith("cuda"))

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": train_data.collate_fn,
        "sampler": None,
        "shuffle": True,
        "pin_memory": pin_mem,
    }
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 1
        train_loader_kwargs["persistent_workers"] = True

    eval_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.eval_num_workers,
        "pin_memory": pin_mem,
    }
    if args.eval_num_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = 1
        eval_loader_kwargs["persistent_workers"] = False

    train_dataloader = DataLoader(train_data, **train_loader_kwargs)

    dev_dataloader = DataLoader(
        dev_data,
        collate_fn=dev_data.collate_fn,
        sampler=None,
        shuffle=False,
        **eval_loader_kwargs,
    )

    test_dataloader = DataLoader(
        test_data,
        collate_fn=test_data.collate_fn,
        sampler=None,
        shuffle=False,
        **eval_loader_kwargs,
    )

    print("Creating model:")

    gmmlp_encoder = None
    if args.gmmlp_checkpoint:
        print(f"Loading GMMLP encoder from {args.gmmlp_checkpoint} …")
        from train_gmmlp import GMMLPImageEncoder

        gmmlp_encoder = GMMLPImageEncoder(
            model_id=args.gmmlp_model_id,
            model_family=args.gmmlp_model_family,
            lora_r=args.gmmlp_lora_r,
            lora_alpha=args.gmmlp_lora_alpha,
            lora_dropout=args.gmmlp_lora_dropout,
            num_latents=args.gmmlp_num_latents,
            num_media_embeds=args.gmmlp_num_media_embeds,
            vision_chunk_size=args.gmmlp_vision_chunk_size,
            temperature=0.07,
        )
        ckpt = torch.load(args.gmmlp_checkpoint, map_location="cpu", weights_only=False)
        prefix = "model_image."
        encoder_state = {
            k[len(prefix) :]: v for k, v in ckpt["model"].items() if k.startswith(prefix)
        }
        missing, unexpected = gmmlp_encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print("GMMLP encoder missing keys:\n", "\n".join(missing))
        if unexpected:
            print("GMMLP encoder unexpected keys:\n", "\n".join(unexpected))
        print(
            f"Loaded GMMLP encoder (D_vit={gmmlp_encoder.vit_hidden}, K={gmmlp_encoder.num_latents})",
        )

        if args.gmmlp_feat_cache:
            # ViT no longer needed — swap to a Perceiver-only encoder to free GPU memory
            # before loading the Gemma4 LM.
            vit_hidden = gmmlp_encoder.vit_hidden
            light = GMMLPImageEncoder(
                model_id=args.gmmlp_model_id,
                model_family=args.gmmlp_model_family,
                lora_r=args.gmmlp_lora_r,
                lora_alpha=args.gmmlp_lora_alpha,
                lora_dropout=args.gmmlp_lora_dropout,
                num_latents=args.gmmlp_num_latents,
                num_media_embeds=args.gmmlp_num_media_embeds,
                vision_chunk_size=args.gmmlp_vision_chunk_size,
                temperature=0.07,
                preextracted_vit_dim=vit_hidden,
            )
            light.perceiver.load_state_dict(gmmlp_encoder.perceiver.state_dict())
            light.cls_token.data.copy_(gmmlp_encoder.cls_token.data)
            light.cls_attn.load_state_dict(gmmlp_encoder.cls_attn.state_dict())
            del gmmlp_encoder
            gc.collect()
            torch.cuda.empty_cache()
            gmmlp_encoder = light
            print(f"Switched to Perceiver-only encoder (D_vit={vit_hidden}), ViT freed.")

    model = MMSLT(
        config,
        args,
        vision_backbone=args.vision_backbone,
        language_decoder=args.language_decoder,
        gemma4_model_id=args.gemma4_model_id,
        gmmlp_encoder=gmmlp_encoder,
    )
    if args.language_decoder == "gemma4":
        # Gemma4 LM is loaded with device_map="auto" + 4-bit bitsandbytes quantization,
        # which can't be moved with .to(). Move every other sub-module explicitly.
        for name, module in model.named_children():
            if name != "gemma4":
                module.to(device)
    else:
        model.to(device)

    if args.finetune:
        print("***********************************")
        print("Load parameters for Visual Encoder...")
        print("***********************************")
        state_dict = torch.load(args.finetune, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in state_dict["model"].items():
            if "model_image.backbone" in k:
                k = "backbone." + ".".join(k.split(".")[2:])
                new_state_dict[k] = v
            if "trans_encoder" in k:
                k = "mbart.base_model.model.model.encoder." + ".".join(k.split(".")[4:])
                new_state_dict[k] = v
            if "model_image.conv" in k:
                k = "conv." + ".".join(k.split(".")[2:])
                new_state_dict[k] = v
            if "projector" in k:
                k = "projector." + ".".join(k.split(".")[2:])
                new_state_dict[k] = v
            if "descriptproj" in k:
                k = "descriptproj." + ".".join(k.split(".")[2:])
                new_state_dict[k] = v

        ret = model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    model_without_ddp = model
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"number of params: {n_parameters}M")

    optimizer = create_optimizer(args, model_without_ddp)
    print(optimizer)

    lr_scheduler = scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        eta_min=args.lr*0.1,
        T_max=args.epochs,
    )
    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}")
    if args.resume:
        print("Resuming Model Parameters... ")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if (
                lr_scheduler is not None
                and "lr_scheduler" in checkpoint
                and checkpoint["lr_scheduler"] is not None
            ):
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        if not args.resume:
            logger.warning(
                "Please specify the trained model: --resume /path/to/best_checkpoint.pth",
            )
        test_stats = evaluate(
            args,
            dev_dataloader,
            model,
            model_without_ddp,
            tokenizer,
            ce_criterion,
            config,
            UNK_IDX,
            SPECIAL_SYMBOLS,
            PAD_IDX,
            device,
            fps=args.fps,
        )
        print(
            f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f} ",
        )
        test_stats = evaluate(
            args,
            test_dataloader,
            model,
            model_without_ddp,
            tokenizer,
            ce_criterion,
            config,
            UNK_IDX,
            SPECIAL_SYMBOLS,
            PAD_IDX,
            device,
            fps=args.fps,
        )
        print(
            f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}",
        )
        return

    if args.debug:
        print("*** DEBUG MODE: overriding epochs to 1 ***")
        args.epochs = args.start_epoch + 1

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    train_peak_ram_gb = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            args,
            model,
            ce_criterion,
            train_dataloader,
            optimizer,
            device,
            epoch,
            config,
        )
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        train_peak_ram_gb = max(train_peak_ram_gb, train_stats.get("peak_ram_gb", 0.0))

        if args.output_dir:
            checkpoint_paths = [
                output_dir / "checkpoint.pth",
                output_dir / f"checkpoint_epoch_{epoch:04d}.pth",
            ]
            for checkpoint_path in checkpoint_paths:
                state = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                if lr_scheduler is not None:
                    state["lr_scheduler"] = lr_scheduler.state_dict()
                torch.save(
                    state,
                    checkpoint_path,
                )

        test_stats = evaluate(
            args,
            dev_dataloader,
            model,
            model_without_ddp,
            tokenizer,
            ce_criterion,
            config,
            UNK_IDX,
            SPECIAL_SYMBOLS,
            PAD_IDX,
            device,
            fps=args.fps,
        )
        print(
            f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}",
        )

        if max_accuracy < test_stats["belu4"]:
            max_accuracy = test_stats["belu4"]
            if args.output_dir:
                state = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "metrics": {
                        "bleu1": test_stats.get("bleu1"),
                        "bleu2": test_stats.get("bleu2"),
                        "bleu3": test_stats.get("bleu3"),
                        "bleu4": test_stats.get("bleu4"),
                        "rouge_l": test_stats.get("rouge_l"),
                        "dev_loss": test_stats.get("loss"),
                        "train_loss": train_stats.get("loss"),
                        "inference_time_per_video_s": test_stats.get("inference_time_per_video_s"),
                        "inference_rtf": test_stats.get("inference_rtf"),
                    },
                }
                if lr_scheduler is not None:
                    state["lr_scheduler"] = lr_scheduler.state_dict()
                torch.save(state, output_dir / "best_checkpoint.pth")

        print(f"Max BELU-4: {max_accuracy:.2f}%")
        wandb.log(
            {
                "epoch": epoch + 1,
                "training/train_loss": train_stats["loss"],
                "dev/dev_loss": test_stats["loss"],
                "dev/Bleu_1": test_stats.get("bleu1", 0.0),
                "dev/Bleu_2": test_stats.get("bleu2", 0.0),
                "dev/Bleu_3": test_stats.get("bleu3", 0.0),
                "dev/Bleu_4": test_stats["belu4"],
                "dev/Best_Bleu_4": max_accuracy,
                "dev/ROUGE_L": test_stats.get("rouge_l", 0.0),
                "dev/inference_time_per_video_s": test_stats.get("inference_time_per_video_s", 0.0),
                "dev/inference_rtf": test_stats.get("inference_rtf", 0.0),
            },
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        if not args.skip_val:
            log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    train_peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0
    )
    print(
        f"[memory/training] peak RAM: {train_peak_ram_gb:.2f} GB  "
        f"peak VRAM: {train_peak_vram_gb:.2f} GB"
    )

    # Last epoch: load best checkpoint, evaluate, and run single-video memory benchmark
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir and not args.skip_val:
        test_model_path = output_dir / "best_checkpoint.pth"
        if not test_model_path.exists():
            test_model_path = output_dir / "checkpoint.pth"
            print(f"Best checkpoint {test_model_path} does not exist, using {test_model_path}.")
        checkpoint = torch.load(test_model_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        test_stats = evaluate(
            args,
            dev_dataloader,
            model,
            model_without_ddp,
            tokenizer,
            ce_criterion,
            config,
            UNK_IDX,
            SPECIAL_SYMBOLS,
            PAD_IDX,
            device,
            fps=args.fps,
        )
        print(
            f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}",
        )

        test_stats = evaluate(
            args,
            test_dataloader,
            model,
            model_without_ddp,
            tokenizer,
            ce_criterion,
            config,
            UNK_IDX,
            SPECIAL_SYMBOLS,
            PAD_IDX,
            device,
            fps=args.fps,
        )
        print(
            f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}",
        )

        # Single-video memory benchmark (same video every run: test_data[0])
        forced_bos_sv = (
            None
            if args.language_decoder == "gemma4"
            else tokenizer.lang_code_to_id["de_DE"]
        )
        sv_src, _ = test_data.collate_fn([test_data[0]])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        sv_ram_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3) if psutil else 0.0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _ = model_without_ddp.generate(
                sv_src,
                max_new_tokens=150,
                num_beams=8,
                forced_bos_token_id=forced_bos_sv,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        sv_peak_vram_gb = (
            torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0
        )
        print(
            f"[memory/single-video] RAM at inference: {sv_ram_gb:.2f} GB  "
            f"peak VRAM: {sv_peak_vram_gb:.2f} GB"
        )

        # Persist memory stats back into best checkpoint
        if (output_dir / "best_checkpoint.pth").exists():
            ckpt = torch.load(output_dir / "best_checkpoint.pth", map_location="cpu", weights_only=False)
            ckpt.setdefault("metrics", {}).update(
                {
                    "total_train_time_s": total_time,
                    "train_peak_ram_gb": train_peak_ram_gb,
                    "train_peak_vram_gb": train_peak_vram_gb,
                    "single_video_ram_gb": sv_ram_gb,
                    "single_video_peak_vram_gb": sv_peak_vram_gb,
                }
            )
            torch.save(ckpt, output_dir / "best_checkpoint.pth")

        wandb.log(
            {
                "memory/train_peak_ram_gb": train_peak_ram_gb,
                "memory/train_peak_vram_gb": train_peak_vram_gb,
                "memory/single_video_ram_gb": sv_ram_gb,
                "memory/single_video_peak_vram_gb": sv_peak_vram_gb,
                "training/total_time_s": total_time,
            }
        )


def train_one_epoch(
    args,
    model: torch.nn.Module,
    ce_criterion: nn.CrossEntropyLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config,
    max_norm: float = 0,
    set_training_mode=True,
):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}/{args.epochs}]"
    print_freq = 10

    peak_ram_gb = 0.0

    for step, (src_input, tgt_input) in enumerate(
        tqdm(
            metric_logger.log_every(data_loader, print_freq, header),
            total=len(data_loader),
            desc=header,
            leave=False,
            disable=not utils.is_main_process(),
        ),
    ):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out_logits = model(src_input, tgt_input)
            label = tgt_input["input_ids"].reshape(-1)
            logits = out_logits.reshape(-1, out_logits.shape[-1])
            ce_loss = ce_criterion(logits, label.to(device, non_blocking=True))

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

        loss_value = ce_loss.item()

        if psutil is not None:
            peak_ram_gb = max(peak_ram_gb, psutil.Process(os.getpid()).memory_info().rss / (1024**3))

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_llm=round(float(optimizer.param_groups[1]["lr"]), 8))

        if (step + 1) % 10 == 0 and args.visualize:
            utils.visualization(model.visualize())

        if args.debug and step >= 1:
            print("*** DEBUG MODE: stopping after 2 batches ***")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {
        **{k: meter.global_avg for k, meter in metric_logger.meters.items()},
        "peak_ram_gb": peak_ram_gb,
    }


def evaluate(
    args,
    dev_dataloader,
    model,
    model_without_ddp,
    tokenizer,
    criterion,
    config,
    UNK_IDX,
    SPECIAL_SYMBOLS,
    PAD_IDX,
    device,
    fps: float = 25.0,
):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    tgt_pres = []
    tgt_refs = []

    total_gen_time = 0.0
    total_video_frames = 0
    num_videos = 0
    forced_bos = (
        None
        if args.language_decoder == "gemma4"
        else tokenizer.lang_code_to_id["de_DE"]
    )

    with torch.no_grad():
        for step, (src_input, tgt_input) in enumerate(
            tqdm(
                metric_logger.log_every(dev_dataloader, 10, header),
                total=len(dev_dataloader),
                desc=header,
                leave=False,
                disable=not utils.is_main_process(),
            ),
        ):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out_logits = model(src_input, tgt_input)
                label = tgt_input["input_ids"].reshape(-1)
                logits = out_logits.reshape(-1, out_logits.shape[-1])
                tgt_loss = criterion(logits, label.to(device))
            metric_logger.update(loss=tgt_loss.item())

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t_gen_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model_without_ddp.generate(
                    src_input,
                    max_new_tokens=args.eval_max_new_tokens,
                    num_beams=args.eval_num_beams,
                    forced_bos_token_id=forced_bos,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            total_gen_time += time.perf_counter() - t_gen_start

            total_video_frames += src_input["src_length_batch"].sum().item()
            num_videos += len(src_input["src_length_batch"])

            pred_texts = tokenizer.batch_decode(output.detach().cpu(), skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(tgt_input["input_ids"], skip_special_tokens=True)
            tgt_pres.extend(pred_texts)
            tgt_refs.extend(ref_texts)

            if args.log_memory and utils.is_main_process() and ((step + 1) % 20 == 0):
                rss_gb = "N/A"
                if psutil is not None:
                    rss_gb = f"{psutil.Process(os.getpid()).memory_info().rss / (1024**3):.2f} GB"
                if torch.cuda.is_available():
                    alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                    print(
                        f"[memory] step={step + 1} rss={rss_gb} cuda_alloc={alloc_gb:.2f} GB cuda_reserved={reserved_gb:.2f} GB",
                    )
                else:
                    print(f"[memory] step={step + 1} rss={rss_gb}")

            if (step + 1) % 10 == 0 and args.visualize and utils.is_main_process():
                utils.visualization(model_without_ddp.visualize())

            if args.debug and step >= 1:
                print("*** DEBUG MODE: stopping after 2 batches ***")
                break

    # BLEU 1-4
    bleu_scores = {}
    for n in range(1, 5):
        b = BLEU(max_ngram_order=n)
        bleu_scores[f"bleu{n}"] = b.corpus_score(tgt_pres, [tgt_refs]).score
    metric_logger.meters["belu4"].update(bleu_scores["bleu4"])

    # ROUGE-L (corpus-level average of sentence F1, scaled to 0-100)
    _rouge_scorer = _rouge_module.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = (
        sum(
            _rouge_scorer.score(ref, pred)["rougeL"].fmeasure
            for ref, pred in zip(tgt_refs, tgt_pres)
        )
        / len(tgt_pres)
        * 100
        if tgt_pres
        else 0.0
    )

    # Timing
    avg_time_per_video = total_gen_time / num_videos if num_videos > 0 else 0.0
    total_video_secs = total_video_frames / fps
    inference_rtf = total_gen_time / total_video_secs if total_video_secs > 0 else 0.0

    if args.eval_metrics and compute_metrics is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            hyp_path = os.path.join(tmp_dir, "tmp_pres.txt")
            ref_path = os.path.join(tmp_dir, "tmp_refs.txt")
            with open(hyp_path, "w") as f:
                f.writelines(tgt_pres[i] + "\n" for i in range(len(tgt_pres)))
            with open(ref_path, "w") as f:
                f.writelines(tgt_refs[i] + "\n" for i in range(len(tgt_refs)))
            print("\n" + "*" * 80)
            metrics = compute_metrics(
                hypothesis=hyp_path,
                references=[ref_path],
                no_skipthoughts=True,
                no_glove=True,
            )
            print("*" * 80)

        def _maybe_pct(value):
            if value is None:
                return None
            return value * 100.0 if value <= 1.0 else value

        bleu1 = _maybe_pct(metrics.get("Bleu_1"))
        bleu2 = _maybe_pct(metrics.get("Bleu_2"))
        bleu3 = _maybe_pct(metrics.get("Bleu_3"))
        rouge_l = _maybe_pct(metrics.get("ROUGE_L"))

        if bleu1 is not None:
            metric_logger.update(bleu1=bleu1)
        if bleu2 is not None:
            metric_logger.update(bleu2=bleu2)
        if bleu3 is not None:
            metric_logger.update(bleu3=bleu3)
        if rouge_l is not None:
            metric_logger.update(rouge_l=rouge_l)

    metric_logger.synchronize_between_processes()
    print(
        f"* BLEU-1/2/3/4: {bleu_scores['bleu1']:.2f}/{bleu_scores['bleu2']:.2f}/"
        f"{bleu_scores['bleu3']:.2f}/{bleu_scores['bleu4']:.2f}  "
        f"ROUGE-L: {rouge_l:.2f}  "
        f"loss: {metric_logger.loss.global_avg:.3f}  "
        f"gen: {avg_time_per_video:.3f}s/video  rtf: {inference_rtf:.3f}s/s"
    )

    if args.eval:
        with open(args.output_dir + "/tmp_pres.txt", "w") as f:
            f.writelines(tgt_pres[i] + "\n" for i in range(len(tgt_pres)))
        with open(args.output_dir + "/tmp_refs.txt", "w") as f:
            f.writelines(tgt_refs[i] + "\n" for i in range(len(tgt_refs)))
        print("\n" + "*" * 80)
        compute_metrics(
            hypothesis=args.output_dir + "/tmp_pres.txt",
            references=[args.output_dir + "/tmp_refs.txt"],
            no_skipthoughts=True,
            no_glove=True,
        )
        print("*" * 80)

    return {
        **{k: meter.global_avg for k, meter in metric_logger.meters.items()},
        **bleu_scores,
        "rouge_l": rouge_l,
        "inference_time_per_video_s": avg_time_per_video,
        "inference_rtf": inference_rtf,
    }


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Avoid "Too many open files" when DataLoader workers share thousands of
    # pre-loaded tensors via the default file-descriptor strategy.
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser("MMSLT script", parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ["WANDB_MODE"] = config["training"]["wandb"] if not args.eval else "disabled"
    wandb.init(project="", config=config)
    wandb.run.name = args.output_dir.split("/")[-1]
    wandb.define_metric("epoch")
    wandb.define_metric("training/*", step_metric="epoch")
    wandb.define_metric("dev/*", step_metric="epoch")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)
