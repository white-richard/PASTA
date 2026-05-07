# from utils import create_mask
import timm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# import pytorchvideo.models.x3d as x3d

""" PyTorch MBART model."""

import numpy as np
from hpman.m import _
from transformers import (
    BitsAndBytesConfig,
    Gemma4ForConditionalGeneration,
    MBartForConditionalGeneration,
)

# global definition
from definition import *


def make_resnet(name="resnet18"):
    if name == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif name == "resnet34":
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    elif name == "resnet101":
        model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
    else:
        msg = "There are no supported resnet model {}.".format(_("resnet"))
        raise Exception(msg)

    model.fc = nn.Identity()
    # model.fc = nn.Linear(inchannel, 768)
    return model


def to_btc(x, lengths):

    x_batch = []
    start = 0
    for length in lengths:
        end = start + length
        x_batch.append(x[start:end])
        start = end
    return pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)


class resnet(nn.Module):
    def __init__(self, frozen=False) -> None:
        super().__init__()
        self.resnet = make_resnet(name="resnet18")

        if frozen:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        return pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)
        # return x_batch


class TimmBackbone(nn.Module):
    def __init__(self, name, frozen=False) -> None:
        super().__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=0)
        # Probe output dim once at init
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.output_dim = self.model(dummy).shape[-1]
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, lengths):
        x = self.model(x)  # (total_frames, output_dim)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        return pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)


class DummyBackbone(nn.Module):
    """Tiny single-conv backbone with random weights for testing."""

    output_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, self.output_dim, kernel_size=7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, lengths):
        x = self.pool(self.conv(x)).flatten(1)  # (total_frames, output_dim)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        return pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)


def build_backbone(name="resnet18"):
    """Returns (backbone_module, output_dim)."""
    if name == "dummy":
        backbone = DummyBackbone()
        return backbone, backbone.output_dim
    if name == "resnet18":
        return resnet(), 512
    backbone = TimmBackbone(name)
    return backbone, backbone.output_dim


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ["K3"]
        elif self.conv_type == 1:
            self.kernel_size = ["K5", "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ["K5", "P2", "K5", "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == "P":
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == "K":
                modules.append(
                    nn.Conv1d(
                        input_sz,
                        self.hidden_size,
                        kernel_size=int(ks[1]),
                        stride=1,
                        padding=0,
                    ),
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.temporal_conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class MMSLT(nn.Module):
    def __init__(
        self,
        config,
        args,
        inplanes=768,
        planes=1024,
        pretrain=None,
        vision_backbone="resnet18",
        language_decoder="mbart",
        gemma4_model_id="google/gemma-4-E2B-it",
        ppasta_encoder=None,
        local_rank=0,
    ) -> None:
        super().__init__()
        self.config = config
        self.args = args
        self.language_decoder = language_decoder

        if language_decoder == "gemma4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.gemma4 = Gemma4ForConditionalGeneration.from_pretrained(
                gemma4_model_id,
                quantization_config=quant_config,
                device_map={"": local_rank},
                low_cpu_mem_usage=True,
            )
            self.gemma4.generation_config.max_length = None
            planes_out = self.gemma4.config.text_config.hidden_size
            # Regex to target only the text language model layers, not vision/audio towers.
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.gemma4 = get_peft_model(self.gemma4, lora_config)
        else:
            model_id = "facebook/mbart-large-50-many-to-many-mmt"
            self.mbart = MBartForConditionalGeneration.from_pretrained(model_id, device_map=None)
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            self.mbart = get_peft_model(self.mbart, lora_config)
            self.mbart.generation_config.max_length = None
            planes_out = planes  # MBart hidden size is 1024

        if ppasta_encoder is not None:
            # Use pretrained PPASTA vision encoder (SigLIP2 ViT + Perceiver) instead of
            # the standard backbone+descriptproj+conv pipeline. The encoder produces
            # (B, K, D_vit) Perceiver latents that are projected directly to the LM space.
            self.ppasta_encoder = ppasta_encoder
            ppasta_dim = ppasta_encoder.vit_hidden
            self.projector = Projector(
                input_dim=ppasta_dim, hidden_dim=planes, output_dim=planes_out
            )
        else:
            self.backbone, backbone_dim = build_backbone(vision_backbone)
            # Description mapper
            self.descriptproj = Projector(
                input_dim=backbone_dim,
                hidden_dim=planes,
                output_dim=inplanes,
            )
            # Modality adapter
            self.conv = TemporalConv(
                input_size=backbone_dim + inplanes,
                hidden_size=planes,
                conv_type=2,
            )
            self.projector = Projector(input_dim=planes, hidden_dim=planes, output_dim=planes_out)
            # Freeze DM
            for param in self.descriptproj.parameters():
                param.requires_grad = False

    def set_decoder_prompt(self, prompt_ids: torch.Tensor) -> None:
        """Register tokenized prompt token IDs (1, P) to prepend before target tokens."""
        self.register_buffer("_prompt_ids", prompt_ids, persistent=False)

    def _prepend_prompt_embeds(self, inputs_embeds, attention_mask):
        """If a decoder prompt is set, embed it and prepend between vision and text tokens.

        Returns (combined_embeds, combined_mask, prompt_len).
        """
        if not hasattr(self, "_prompt_ids") or self._prompt_ids is None:
            return inputs_embeds, attention_mask, 0

        B = inputs_embeds.shape[0]
        device = inputs_embeds.device
        if self.language_decoder == "gemma4":
            prompt_ids = self._prompt_ids.to(device).expand(B, -1)
            prompt_embeds = self._g4_text.embed_tokens(prompt_ids)  # (B, P, D)
        else:
            prompt_ids = self._prompt_ids.to(device).expand(B, -1)
            prompt_embeds = self.mbart.model.shared(prompt_ids)  # (B, P, D)

        P = prompt_embeds.shape[1]
        prompt_mask = torch.ones(B, P, dtype=attention_mask.dtype, device=device)
        combined_embeds = torch.cat([inputs_embeds, prompt_embeds], dim=1)
        combined_mask = torch.cat([attention_mask, prompt_mask], dim=1)
        return combined_embeds, combined_mask, P

    # Gemma4 helpers: bypass Gemma4Model.forward() to avoid the OOM from
    # per-layer embedding (PLE) recomputation over the full multimodal sequence.

    @property
    def _g4_text(self):
        """Gemma4TextModel with LoRA applied."""
        return self.gemma4.base_model.model.model.language_model

    @property
    def _g4_head(self):
        return self.gemma4.base_model.model.lm_head

    def _g4_logits(self, hidden_states):
        """Apply lm_head + final_logit_softcapping."""
        logits = self._g4_head(hidden_states)
        softcap = self.gemma4.config.text_config.final_logit_softcapping
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _g4_ple(self, B, T_vis, device, dtype, tgt_ids=None):
        """Build the per-layer embedding tensor.

        Returns shape (B, T_vis [+ L], num_layers, h_ple) or None if the
        model does not use PLE (hidden_size_per_layer_input == 0).
        """
        h_ple = self.gemma4.config.text_config.hidden_size_per_layer_input
        if not h_ple:
            return None
        n = self.gemma4.config.text_config.num_hidden_layers
        vis_ple = torch.zeros(B, T_vis, n, h_ple, dtype=dtype, device=device)
        if tgt_ids is None:
            return vis_ple
        text_ple = self._g4_text.get_per_layer_inputs(tgt_ids, None)  # (B, L, n, h_ple)
        return torch.cat([vis_ple, text_ple], dim=1)

    @torch.no_grad()
    def _gemma4_generate(
        self,
        vis_embeds,
        attention_mask,
        max_new_tokens,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 4,
    ):
        """Greedy decoding with prefix KV-cache seeded by vision embeddings."""
        from transformers import DynamicCache

        B, T, _ = vis_embeds.shape
        device = vis_embeds.device
        vocab_size = self.gemma4.config.text_config.vocab_size

        def _apply_penalties(logits, generated):
            if repetition_penalty != 1.0 and generated.shape[1] > 0:
                hit = torch.zeros(B, vocab_size, dtype=torch.bool, device=logits.device)
                hit.scatter_(1, generated.clamp(0, vocab_size - 1), True)
                penalized = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
                logits = torch.where(hit, penalized, logits)
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                for b in range(B):
                    seq = generated[b].tolist()
                    prefix = tuple(seq[-(n - 1):])
                    banned = {seq[i + n - 1] for i in range(len(seq) - (n - 1)) if tuple(seq[i: i + n - 1]) == prefix}
                    for tid in banned:
                        logits[b, tid] = -float("inf")
            return logits

        past_kv = DynamicCache()
        vis_ple = self._g4_ple(B, T, device, vis_embeds.dtype)
        outputs = self._g4_text(
            inputs_embeds=vis_embeds,
            attention_mask=attention_mask,
            per_layer_inputs=vis_ple,
            past_key_values=past_kv,
            use_cache=True,
        )
        logits = self._g4_logits(outputs.last_hidden_state[:, -1, :])
        generated = torch.empty(B, 0, dtype=torch.long, device=device)
        logits = _apply_penalties(logits, generated)
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = next_token

        prefix_len = attention_mask.shape[1]
        full_mask_buf = torch.ones(B, prefix_len + max_new_tokens, dtype=attention_mask.dtype, device=device)
        full_mask_buf[:, :prefix_len] = attention_mask

        eos_id = self.gemma4.config.text_config.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_new_tokens - 1):
            cur_mask_len = prefix_len + step + 1
            tok_embeds = self._g4_text.embed_tokens(next_token)
            step_ple = self._g4_ple(B, 0, device, tok_embeds.dtype, next_token)
            outputs = self._g4_text(
                inputs_embeds=tok_embeds,
                attention_mask=full_mask_buf[:, :cur_mask_len],
                per_layer_inputs=step_ple,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = self._g4_logits(outputs.last_hidden_state[:, -1, :])
            logits = _apply_penalties(logits, generated)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return generated

    def share_forward(self, src_input):
        if hasattr(self, "ppasta_encoder"):
            # PPASTA path: frames → Perceiver latents (B, K, D_vit) → projector.
            # Prefers pre-extracted vis_feats (skips ViT); falls back to PIL frames.
            if "vis_feats" in src_input:
                enc_input = {"vis_feats": src_input["vis_feats"]}
            else:
                enc_input = {"images": src_input["pil_frames"]}
            latents = self.ppasta_encoder.forward_latents(enc_input)  # (B, K, D_vit)
            inputs_embeds = self.projector(latents)  # (B, K, planes_out)
            B, K, _ = inputs_embeds.shape
            attention_mask = torch.ones(B, K, dtype=torch.long)
            return inputs_embeds, attention_mask

        img_feature = self.backbone(src_input["input_img"].cuda(), src_input["src_length_batch"])
        descript_feature = self.descriptproj(img_feature)
        inputs_embeds = torch.cat([img_feature, descript_feature], dim=-1)
        inputs_embeds = self.conv(inputs_embeds)
        inputs_embeds = self.projector(inputs_embeds)

        attention_mask = src_input["attention_mask"]

        return inputs_embeds, attention_mask

    def forward(self, src_input, tgt_input):

        inputs_embeds, attention_mask = self.share_forward(src_input)

        if self.language_decoder == "gemma4":
            B, T, _ = inputs_embeds.shape
            vis_embeds_with_prompt, vis_mask_with_prompt, P = self._prepend_prompt_embeds(
                inputs_embeds,
                attention_mask.cuda(),
            )
            T_total = T + P
            tgt_ids = tgt_input["input_ids"].cuda()
            tok_embeds = self._g4_text.embed_tokens(tgt_ids)  # (B, L, D)
            combined_embeds = torch.cat(
                [vis_embeds_with_prompt, tok_embeds], dim=1
            )  # (B, T_total+L, D)
            combined_mask = torch.cat(
                [vis_mask_with_prompt, tgt_input["attention_mask"].cuda()],
                dim=1,
            )
            per_layer_inputs = self._g4_ple(
                B, T_total, inputs_embeds.device, inputs_embeds.dtype, tgt_ids
            )
            outputs = self._g4_text(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                per_layer_inputs=per_layer_inputs,
            )
            # Logits aligned with tgt labels: last vis+prompt position predicts tok_0, etc.
            L = tgt_ids.shape[1]
            return self._g4_logits(outputs.last_hidden_state)[:, T_total - 1 : T_total + L - 1, :]
        out = self.mbart(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.cuda(),
            labels=tgt_input["input_ids"].cuda(),
            decoder_attention_mask=tgt_input["attention_mask"].cuda(),
            return_dict=True,
        )
        return out["logits"]

    def encode_vision(self, src_input):
        """Run just the vision encoder; return (inputs_embeds, attention_mask).

        Call this once per eval batch and pass the results to forward_from_embeds
        and generate_from_embeds to avoid running the vision stack twice.
        """
        return self.share_forward(src_input)

    def forward_from_embeds(self, inputs_embeds, attention_mask, tgt_input):
        """Like forward(), but takes pre-computed vision embeddings."""
        if self.language_decoder == "gemma4":
            B, T, _ = inputs_embeds.shape
            vis_embeds_with_prompt, vis_mask_with_prompt, P = self._prepend_prompt_embeds(
                inputs_embeds,
                attention_mask.cuda(),
            )
            T_total = T + P
            tgt_ids = tgt_input["input_ids"].cuda()
            tok_embeds = self._g4_text.embed_tokens(tgt_ids)
            combined_embeds = torch.cat([vis_embeds_with_prompt, tok_embeds], dim=1)
            combined_mask = torch.cat(
                [vis_mask_with_prompt, tgt_input["attention_mask"].cuda()],
                dim=1,
            )
            per_layer_inputs = self._g4_ple(
                B, T_total, inputs_embeds.device, inputs_embeds.dtype, tgt_ids
            )
            outputs = self._g4_text(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                per_layer_inputs=per_layer_inputs,
            )
            L = tgt_ids.shape[1]
            return self._g4_logits(outputs.last_hidden_state)[:, T_total - 1 : T_total + L - 1, :]
        out = self.mbart(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.cuda(),
            labels=tgt_input["input_ids"].cuda(),
            decoder_attention_mask=tgt_input["attention_mask"].cuda(),
            return_dict=True,
        )
        return out["logits"]

    def generate_from_embeds(
        self,
        inputs_embeds,
        attention_mask,
        max_new_tokens,
        num_beams,
        forced_bos_token_id=None,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 4,
    ):
        """Like generate(), but takes pre-computed vision embeddings."""
        if self.language_decoder == "gemma4":
            vis_embeds_with_prompt, vis_mask_with_prompt, _ = self._prepend_prompt_embeds(
                inputs_embeds,
                attention_mask.cuda(),
            )
            return self._gemma4_generate(
                vis_embeds_with_prompt,
                vis_mask_with_prompt,
                max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return self.mbart.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.cuda(),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            forced_bos_token_id=forced_bos_token_id,
        )

    def generate(
        self,
        src_input,
        max_new_tokens,
        num_beams,
        forced_bos_token_id=None,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 4,
    ):
        inputs_embeds, attention_mask = self.share_forward(src_input)
        return self.generate_from_embeds(
            inputs_embeds,
            attention_mask,
            max_new_tokens,
            num_beams,
            forced_bos_token_id=forced_bos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.grad_cache = True
        self.model_txt = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt",
        ).get_encoder()
        for param in self.model_txt.parameters():
            param.requires_grad = False

    def forward(self, tgt_input):
        # If using gradcahe, then is not needed
        if self.grad_cache:
            txt_logits = self.model_txt(
                input_ids=tgt_input["input_ids"].cuda(),
                attention_mask=tgt_input["attention_mask"].cuda(),
            )[0]
        else:
            with torch.no_grad():
                txt_logits = self.model_txt(
                    input_ids=tgt_input["input_ids"].cuda(),
                    attention_mask=tgt_input["attention_mask"].cuda(),
                )[0]

        return txt_logits.mean(dim=1)  # [b, 1024]


class ImageEncoder(nn.Module):
    def __init__(self, inplanes=768, planes=1024, head_type="linear", backbone="resnet18") -> None:
        super().__init__()

        self.backbone, backbone_dim = build_backbone(backbone)
        # Description mapper
        self.descriptproj = Projector(
            input_dim=backbone_dim,
            hidden_dim=planes,
            output_dim=inplanes,
        )
        # Modality Adapter
        self.conv = TemporalConv(
            input_size=backbone_dim + inplanes,
            hidden_size=planes,
            conv_type=2,
        )
        self.projector = Projector(input_dim=planes, hidden_dim=planes, output_dim=planes)
        # Multimodal encoder
        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt",
        ).get_encoder()
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
            ],  # Apply LoRA to query and value layers
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.trans_encoder = get_peft_model(self.trans_encoder, lora_config)

    def forward(self, src_input):

        tgt_descript = to_btc(src_input["input_descript"].cuda(), src_input["src_length_batch"])
        img_feature = self.backbone(src_input["input_img"].cuda(), src_input["src_length_batch"])
        descript_feature = self.descriptproj(img_feature)
        mse_loss = F.mse_loss(descript_feature, tgt_descript)

        inputs_embeds = torch.cat([img_feature, descript_feature], dim=-1)
        inputs_embeds = self.conv(inputs_embeds)
        inputs_embeds = self.projector(inputs_embeds)

        attention_mask = src_input["attention_mask"]

        outs = self.trans_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.cuda(),
            return_dict=True,
        )
        last_hidden_state = outs["last_hidden_state"]
        # output = last_hidden_state[:, 0, :] #[b, 1024]
        output = last_hidden_state.mean(dim=1)

        return output, mse_loss


class MMLP(nn.Module):
    def __init__(self, config, embed_dim=1024, vision_backbone="resnet18") -> None:
        super().__init__()
        self.model_text = TextEncoder()
        self.model_image = ImageEncoder(inplanes=768, planes=embed_dim, backbone=vision_backbone)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_sign(self, src_input):
        """Returns (normalized_sign_embedding, mse_loss)."""
        output, mse_loss = self.model_image(src_input)
        return F.normalize(output, p=2, dim=-1), mse_loss

    def encode_text(self, tgt_input):
        """Returns normalized text embedding."""
        output = self.model_text(tgt_input)
        return F.normalize(output, p=2, dim=-1)

    def forward(self, src_input, tgt_input):
        text_features = self.model_text(tgt_input)
        image_features, descript_loss = self.model_image(src_input)

        # normalized features
        norm_text = F.normalize(text_features, p=2, dim=-1)
        norm_images = F.normalize(image_features, p=2, dim=-1)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        sim_text = torch.matmul(norm_text, norm_images.t()) * logit_scale
        sim_image = torch.matmul(norm_images, norm_text.t()) * logit_scale

        return sim_text, sim_image, descript_loss
