"""Model loading and per-video inference for the TUI.

Designed to be imported from the project root (or with src/ on sys.path).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

# Ensure src/ is importable regardless of CWD
_ROOT = Path(__file__).parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


@dataclass
class VideoInfo:
    idx: int
    key: str  # e.g. "test/24March_2011_..."
    name: str  # annotation name field
    video_name: str  # directory name inside fullFrame-210x260px/test/
    reference: str  # ground-truth German translation
    num_frames: int  # number of image files


@dataclass
class InferenceResult:
    video_info: VideoInfo
    prediction: str
    reference: str
    inference_time: float  # seconds


@dataclass
class CumulativeStats:
    """Running BLEU-1/2/3/4 and ROUGE-L across all evaluated videos."""

    hypotheses: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    inference_times: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.hypotheses)

    def add(self, hyp: str, ref: str, t: float) -> None:
        self.hypotheses.append(hyp)
        self.references.append(ref)
        self.inference_times.append(t)

    def bleu(self, order: int) -> float:
        if not self.hypotheses:
            return 0.0
        from sacrebleu.metrics import BLEU

        metric = BLEU(max_ngram_order=order, effective_order=True)
        return metric.corpus_score(self.hypotheses, [self.references]).score

    def rouge_l(self) -> float:
        if not self.hypotheses:
            return 0.0
        scores = [
            _rouge_l_sentence(h, r) for h, r in zip(self.hypotheses, self.references, strict=False)
        ]
        return sum(scores) / len(scores) * 100.0

    def avg_inference_time(self) -> float:
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)


def _rouge_l_sentence(hyp: str, ref: str) -> float:
    """ROUGE-L F1 at word level."""
    h = hyp.lower().split()
    r = ref.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(r), len(h)
    # DP for LCS length
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    p = lcs / n
    r_score = lcs / m
    if p + r_score == 0:
        return 0.0
    return 2 * p * r_score / (p + r_score)


def load_test_videos(config_path: str, root: Path = _ROOT) -> list[VideoInfo]:
    """Scan the test video directory and return VideoInfo list immediately.

    Ground-truth translations are loaded from the Phoenix CSV if available;
    otherwise each video gets an empty reference string.  This function is
    intentionally fast — it does not load any model weights.
    """
    cfg_full = root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    with open(cfg_full, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    img_root = config["data"]["img_path"]
    if not Path(img_root).is_absolute():
        img_root = root / img_root
    test_dir = Path(img_root) / "test"

    # Attempt to load ground-truth translations from CSV
    ground_truth: dict[str, str] = {}
    try:
        label_path = config["data"]["label_path"]["test"]
        if not Path(label_path).is_absolute():
            label_path = str(root / label_path)
        from datasets import load_dataset_file

        raw = load_dataset_file(label_path)
        for key, val in raw.items():
            vid = key.split("/")[1] if "/" in key else key
            ground_truth[vid] = val.get("text", "")
    except Exception:
        pass  # Ground truth optional — inference still works

    # Scan filesystem for video directories
    videos: list[VideoInfo] = []
    if test_dir.exists():
        for idx, d in enumerate(sorted(test_dir.iterdir())):
            if not d.is_dir():
                continue
            frames = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
            videos.append(
                VideoInfo(
                    idx=idx,
                    key=f"test/{d.name}",
                    name=d.name,
                    video_name=d.name,
                    reference=ground_truth.get(d.name, ""),
                    num_frames=len(frames),
                ),
            )
    return videos


class InferenceEngine:
    """Loads an MMSLT checkpoint and exposes per-video inference."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "src/configs/config_mmslt_phoenix.yaml",
        language_decoder: str = "gemma4",
        gemma4_model_id: str = "google/gemma-4-E2B-it",
        vision_backbone: str = "resnet18",
        gmmlp_checkpoint: str = "",
        gmmlp_model_id: str = "google/gemma-4-E2B-it",
        gmmlp_model_family: str = "gemma4",
        gmmlp_lora_r: int = 16,
        gmmlp_lora_alpha: int = 32,
        gmmlp_lora_dropout: float = 0.05,
        gmmlp_num_latents: int = 64,
        gmmlp_num_media_embeds: int = 512,
        gmmlp_vision_chunk_size: int = 8,
        gmmlp_feat_cache: str = "",
        eval_max_new_tokens: int = 80,
        eval_num_beams: int = 4,
        device: str = "cuda",
        on_status: callable | None = None,
    ) -> None:
        self._on_status = on_status or (lambda msg: None)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.eval_max_new_tokens = eval_max_new_tokens
        self.eval_num_beams = eval_num_beams
        self.language_decoder = language_decoder
        self._config_path_str = config_path

        self._on_status("Loading config…")
        config_full = (
            Path(_ROOT) / config_path if not Path(config_path).is_absolute() else Path(config_path)
        )
        with open(config_full, encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # ── Tokenizer ────────────────────────────────────────────────────────
        self._on_status("Loading tokenizer…")
        if language_decoder == "gemma4":
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(gemma4_model_id)
        else:
            from transformers import MBart50TokenizerFast

            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt",
                src_lang="de_DE",
                tgt_lang="de_DE",
                model_max_length=1024,
            )

        # ── GMMLP encoder (optional) ──────────────────────────────────────
        self._on_status("Loading GMMLP encoder…" if gmmlp_checkpoint else "Skipping GMMLP encoder…")
        gmmlp_encoder = None
        if gmmlp_checkpoint:
            from train_gmmlp import GMMLPImageEncoder

            # Read D_vit from feature cache metadata to skip loading the ViT.
            preextracted_vit_dim = None
            if gmmlp_feat_cache:
                meta_path = Path(gmmlp_feat_cache) / "features_test" / "_meta.pt"
                if meta_path.exists():
                    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                    preextracted_vit_dim = meta.get("d_vit")

            gmmlp_encoder = GMMLPImageEncoder(
                model_id=gmmlp_model_id,
                model_family=gmmlp_model_family,
                lora_r=gmmlp_lora_r,
                lora_alpha=gmmlp_lora_alpha,
                lora_dropout=gmmlp_lora_dropout,
                num_latents=gmmlp_num_latents,
                num_media_embeds=gmmlp_num_media_embeds,
                vision_chunk_size=gmmlp_vision_chunk_size,
                temperature=0.07,
                preextracted_vit_dim=preextracted_vit_dim,
            )
            ckpt = torch.load(gmmlp_checkpoint, map_location="cpu", weights_only=False)
            prefix = "model_image."
            encoder_state = {
                k[len(prefix) :]: v for k, v in ckpt["model"].items() if k.startswith(prefix)
            }
            gmmlp_encoder.load_state_dict(encoder_state, strict=False)

        # ── MMSLT model ───────────────────────────────────────────────────
        self._on_status("Building model architecture…")

        # Build a minimal args namespace for MMSLT
        import argparse

        args = argparse.Namespace(
            vision_backbone=vision_backbone,
            language_decoder=language_decoder,
            gemma4_model_id=gemma4_model_id,
            input_size=224,
            resize=256,
            drop=0.0,
            drop_path=0.1,
        )

        from models import MMSLT

        self.model = MMSLT(
            self.config,
            args,
            vision_backbone=vision_backbone,
            language_decoder=language_decoder,
            gemma4_model_id=gemma4_model_id,
            gmmlp_encoder=gmmlp_encoder,
        )

        self._on_status("Moving model to device…")
        if language_decoder == "gemma4":
            for name, module in self.model.named_children():
                if name != "gemma4":
                    module.to(self.device)
        else:
            self.model.to(self.device)

        # ── Load checkpoint ───────────────────────────────────────────────
        self._on_status(f"Loading checkpoint: {checkpoint_path}…")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()

        # ── Dataset (test split, for ground truth + frame paths) ──────────
        self._on_status("Loading test annotations…")
        self._gmmlp_feat_cache = (
            Path(gmmlp_feat_cache) / "features_test" if gmmlp_feat_cache else None
        )
        self._use_gmmlp = bool(gmmlp_encoder)
        self._load_test_data()

    # ── Dataset helpers ────────────────────────────────────────────────────

    def _load_test_data(self) -> None:
        """Build VideoInfo list from the filesystem (fast) + CSV ground truth."""
        self._videos = load_test_videos(self._config_path_str)

        img_root = self.config["data"]["img_path"]
        if not Path(img_root).is_absolute():
            img_root = str(_ROOT / img_root)
        self._img_root = img_root
        self._on_status(f"Loaded {len(self._videos)} test videos.")

    @property
    def videos(self) -> list[VideoInfo]:
        return self._videos

    # ── Inference ──────────────────────────────────────────────────────────

    def run_single(self, video: VideoInfo) -> InferenceResult:
        """Run generate() on one video and return the result."""
        src_input = self._build_src_input(video)
        t0 = time.perf_counter()

        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            forced_bos = (
                None
                if self.language_decoder == "gemma4"
                else self.tokenizer.lang_code_to_id["de_DE"]
            )
            output = self.model.generate(
                src_input,
                max_new_tokens=self.eval_max_new_tokens,
                num_beams=self.eval_num_beams,
                forced_bos_token_id=forced_bos,
            )

        elapsed = time.perf_counter() - t0
        pred = self.tokenizer.batch_decode(output.detach().cpu(), skip_special_tokens=True)
        prediction = pred[0] if pred else ""

        return InferenceResult(
            video_info=video,
            prediction=prediction,
            reference=video.reference,
            inference_time=elapsed,
        )

    def _build_src_input(self, video: VideoInfo) -> dict:
        """Build the src_input dict that MMSLT.generate() expects."""
        if self._use_gmmlp and self._gmmlp_feat_cache is not None:
            # Fast path: load pre-extracted ViT features from cache
            pt_path = self._gmmlp_feat_cache / f"{video.video_name}.pt"
            if not pt_path.exists():
                msg = (
                    f"GMMLP feature cache missing for {video.video_name}.\n"
                    f"Expected: {pt_path}\n"
                    "Run with --preextract_gmmlp first."
                )
                raise FileNotFoundError(
                    msg,
                )
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            vis_feats = data["vis"] if isinstance(data, dict) else data  # (T, P, D) or (T, D)
            return {
                "vis_feats": [vis_feats],
                "name_batch": [video.video_name],
                "src_length_batch": torch.tensor([len(vis_feats)]),
            }

        # Scan video directory for frames (works for both GMMLP PIL and standard backbone)
        frame_dir = Path(self._img_root) / "test" / video.video_name
        img_paths = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))
        if not img_paths:
            msg = f"No frames found in {frame_dir}"
            raise FileNotFoundError(msg)

        import cv2
        import numpy as np
        from PIL import Image

        if self._use_gmmlp:
            frames = []
            for p in img_paths:
                img = cv2.imread(str(p))
                if img is None:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(img))
            return {"pil_frames": [frames], "name_batch": [video.video_name]}

        # Standard backbone path
        from torchvision import transforms

        max_length = self.config["data"].get("max_length", 150)
        if len(img_paths) > max_length:
            import random

            selected = sorted(random.sample(range(len(img_paths)), k=max_length))
            img_paths = [img_paths[i] for i in selected]

        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
        )
        imgs = []
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((224, 224))
            imgs.append(data_transform(img))

        input_img = torch.stack(imgs)  # (T, 3, 224, 224)
        return {
            "input_img": input_img,
            "src_length_batch": [len(imgs)],
            "attention_mask": torch.ones(1, len(imgs), dtype=torch.long),
            "name_batch": [video.video_name],
        }
