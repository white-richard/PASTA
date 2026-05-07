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
        gmmlp_encoder=None,
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

        if gmmlp_encoder is not None:
            # Use pretrained GMMLP vision encoder (SigLIP2 ViT + Perceiver) instead of
            # the standard backbone+descriptproj+conv pipeline. The encoder produces
            # (B, K, D_vit) Perceiver latents that are projected directly to the LM space.
            self.gmmlp_encoder = gmmlp_encoder
            gmmlp_dim = gmmlp_encoder.vit_hidden
            self.projector = Projector(
                input_dim=gmmlp_dim, hidden_dim=planes, output_dim=planes_out
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

    def share_forward(self, src_input):
        if hasattr(self, "gmmlp_encoder"):
            # GMMLP path: frames → Perceiver latents (B, K, D_vit) → projector.
            # Prefers pre-extracted vis_feats (skips ViT); falls back to PIL frames.
            if "vis_feats" in src_input:
                enc_input = {"vis_feats": src_input["vis_feats"]}
            else:
                enc_input = {"images": src_input["pil_frames"]}
            latents = self.gmmlp_encoder.forward_latents(enc_input)  # (B, K, D_vit)
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
