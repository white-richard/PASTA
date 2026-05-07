# *torch
import argparse
import datetime
import json
import math

# *basic
import os
import random
import sys
import time
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

import hpargparse
import numpy as np
import test as test

# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import utils as utils
import wandb
import yaml
from hpman.m import _
from loguru import logger

# *metric
# *user-defined
from models import MMSLT
from sacrebleu.metrics import BLEU
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader

# *transformers
from transformers import (
    MBart50TokenizerFast,
)

from datasets import S2T_Dataset

try:
    from nlgeval import compute_metrics
except:
    print("Please install nlgeval package.")

# *timm
# global definition
from definition import *
from timm.optim import create_optimizer
from timm.utils import NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser(
        "LLaVA-guided Sign Language Translation script", add_help=False
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=80, type=int)

    # * distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--local_rank", default=0, type=int)

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    # * Optimizer parameters
    parser.add_argument(
        "--opt", default="adamw", type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw"'
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
        "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)"
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
        "--lr", type=float, default=1.0e-3, metavar="LR", help="learning rate (default: 5e-4)"
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
        "--decay-epochs", type=float, default=30, metavar="N", help="epoch interval to decay LR"
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
        "--dist-eval", action="store_true", default=False, help="Enabling distributed evaluation"
    )
    parser.add_argument("--num_workers", default=8, type=int)
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
    parser.add_argument("--config", type=str, default="./configs/config_mmslt_phoenix.yaml")
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log process/CUDA memory periodically.",
    )

    # *Drop out params
    parser.add_argument(
        "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
    )
    parser.add_argument(
        "--drop-path", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)"
    )

    # * data process params
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="timm model name to use as vision backbone (e.g. resnet18, vit_base_patch14_dinov2.lvd142m).",
    )

    # * visualization
    parser.add_argument("--visualize", action="store_true")

    # * debug
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Run in debug mode: only 1 epoch and 2 batches per split.",
    )

    return parser


def main(args, config):
    # torch.multiprocessing.set_start_method('spawn')
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print("Creating dataset:")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
        src_lang="de_DE",
        tgt_lang="de_DE",
        model_max_length=1024,
    )

    train_data = S2T_Dataset(
        path=config["data"]["label_path"],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase="train",
    )
    print(train_data)

    dev_data = S2T_Dataset(
        path=config["data"]["label_path"],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase="dev",
    )
    print(dev_data)

    test_data = S2T_Dataset(
        path=config["data"]["label_path"],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase="test",
    )
    print(test_data)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)

    pin_mem = bool(args.pin_mem and args.device.startswith("cuda"))

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler if args.distributed else None,
        shuffle=(args.distributed is False),
        pin_memory=pin_mem,
    )
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        train_loader_kwargs["persistent_workers"] = True

    eval_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.eval_num_workers,
        pin_memory=pin_mem,
    )
    if args.eval_num_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = 1
        eval_loader_kwargs["persistent_workers"] = False

    train_dataloader = DataLoader(train_data, **train_loader_kwargs)

    dev_dataloader = DataLoader(
        dev_data,
        collate_fn=dev_data.collate_fn,
        sampler=dev_sampler if args.distributed else None,
        shuffle=(args.distributed is False),
        **eval_loader_kwargs,
    )

    test_dataloader = DataLoader(
        test_data,
        collate_fn=test_data.collate_fn,
        sampler=test_sampler if args.distributed else None,
        shuffle=(args.distributed is False),
        **eval_loader_kwargs,
    )

    print("Creating model:")

    model = MMSLT(config, args, backbone=args.backbone)
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

    # print(model)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"number of params: {n_parameters}M")

    optimizer = create_optimizer(args, model_without_ddp)
    print(optimizer)

    lr_scheduler = scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        eta_min=1e-8,
        T_max=args.epochs,
    )
    loss_scaler = NativeScaler()

    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}")
    if args.resume:
        print("Resuming Model Parameters... ")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        if not args.resume:
            logger.warning(
                "Please specify the trained model: --resume /path/to/best_checkpoint.pth"
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
        )
        print(
            f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f} "
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
        )
        print(
            f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}"
        )
        return

    if args.debug_mode:
        print("*** DEBUG MODE: overriding epochs to 1 ***")
        args.epochs = args.start_epoch + 1

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            args,
            model,
            ce_criterion,
            train_dataloader,
            optimizer,
            device,
            epoch,
            config,
            loss_scaler,
        )
        lr_scheduler.step(epoch)

        if args.output_dir and utils.is_main_process():
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
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
        )
        print(
            f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}"
        )

        if max_accuracy < test_stats["belu4"]:
            max_accuracy = test_stats["belu4"]
            if args.output_dir and utils.is_main_process():
                checkpoint_paths = [output_dir / "best_checkpoint.pth"]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

        print(f"Max BELU-4: {max_accuracy:.2f}%")
        if utils.is_main_process():
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "training/train_loss": train_stats["loss"],
                    "dev/dev_loss": test_stats["loss"],
                    "dev/Bleu_4": test_stats["belu4"],
                    "dev/Best_Bleu_4": max_accuracy,
                }
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        test_model_path = output_dir / "best_checkpoint.pth"
        if not test_model_path.exists():
            test_model_path = output_dir / "checkpoint.pth"
            print(f"Best checkpoint {test_model_path} does not exist, using {test_model_path}.")
        checkpoint = torch.load(test_model_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)

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
        )
        print(
            f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}"
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
        )
        print(
            f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}"
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def train_one_epoch(
    args,
    model: torch.nn.Module,
    ce_criterion: nn.CrossEntropyLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config,
    loss_scaler,
    max_norm: float = 0,
    set_training_mode=True,
):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}/{args.epochs}]"
    print_freq = 10

    for step, (src_input, tgt_input) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        out_logits = model(src_input, tgt_input)
        label = tgt_input["input_ids"].reshape(-1)
        logits = out_logits.reshape(-1, out_logits.shape[-1])
        ce_loss = ce_criterion(logits, label.to(device, non_blocking=True))

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

        loss_value = ce_loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_llm=round(float(optimizer.param_groups[1]["lr"]), 8))

        if (step + 1) % 10 == 0 and args.visualize and utils.is_main_process():
            utils.visualization(model.module.visualize())

        if args.debug_mode and step >= 1:
            print("*** DEBUG MODE: stopping after 2 batches ***")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    tgt_pres = []
    tgt_refs = []

    with torch.no_grad():
        for step, (src_input, tgt_input) in enumerate(
            metric_logger.log_every(dev_dataloader, 10, header)
        ):
            out_logits = model(src_input, tgt_input)
            label = tgt_input["input_ids"].reshape(-1)
            logits = out_logits.reshape(-1, out_logits.shape[-1])
            tgt_loss = criterion(logits, label.to(device))
            metric_logger.update(loss=tgt_loss.item())

            output = model_without_ddp.generate(
                src_input,
                max_new_tokens=150,
                num_beams=8,
                forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
            )
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
                        f"[memory] step={step + 1} rss={rss_gb} cuda_alloc={alloc_gb:.2f} GB cuda_reserved={reserved_gb:.2f} GB"
                    )
                else:
                    print(f"[memory] step={step + 1} rss={rss_gb}")

            if (step + 1) % 10 == 0 and args.visualize and utils.is_main_process():
                utils.visualization(model_without_ddp.visualize())

            if args.debug_mode and step >= 1:
                print("*** DEBUG MODE: stopping after 2 batches ***")
                break

    bleu = BLEU()
    bleu_s = bleu.corpus_score(tgt_pres, [tgt_refs]).score
    metric_logger.meters["belu4"].update(bleu_s)

    metric_logger.synchronize_between_processes()
    print(f"* BELU-4 {metric_logger.belu4.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}")

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir + "/tmp_pres.txt", "w") as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i] + "\n")
        with open(args.output_dir + "/tmp_refs.txt", "w") as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i] + "\n")
        print("\n" + "*" * 80)
        metrics_dict = compute_metrics(
            hypothesis=args.output_dir + "/tmp_pres.txt",
            references=[args.output_dir + "/tmp_refs.txt"],
            no_skipthoughts=True,
            no_glove=True,
        )
        print("*" * 80)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser("MMSLT script", parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ["WANDB_MODE"] = config["training"]["wandb"] if not args.eval else "disabled"
    if utils.is_main_process():
        wandb.init(project="", config=config)
        wandb.run.name = args.output_dir.split("/")[-1]
        wandb.define_metric("epoch")
        wandb.define_metric("training/*", step_metric="epoch")
        wandb.define_metric("dev/*", step_metric="epoch")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)
