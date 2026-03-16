import torch
import torch.nn.functional as F
import torch.utils.checkpoint

# from utils import create_mask
import timm
import torchvision
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# import pytorchvideo.models.x3d as x3d

""" PyTorch MBART model."""

import numpy as np
from hpman.m import _
from transformers import (
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
    def __init__(self, name, frozen=False):
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

    def __init__(self):
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
        backbone="resnet18",
    ) -> None:
        super().__init__()
        self.config = config
        self.args = args

        model_id = "facebook/mbart-large-50-many-to-many-mmt"
        self.mbart = MBartForConditionalGeneration.from_pretrained(model_id, device_map=None)
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
            task_type="SEQ_2_SEQ_LM",
        )
        self.mbart = get_peft_model(self.mbart, lora_config)
        self.mbart.generation_config.max_length = None

        self.backbone, backbone_dim = build_backbone(backbone)
        # Description mapper
        self.descriptproj = Projector(input_dim=backbone_dim, hidden_dim=planes, output_dim=inplanes)
        # Modality adapter
        self.conv = TemporalConv(input_size=backbone_dim + inplanes, hidden_size=planes, conv_type=2)
        self.projector = Projector(input_dim=planes, hidden_dim=planes, output_dim=planes)
        # Freeze DM
        for param in self.descriptproj.parameters():
            param.requires_grad = False

    def share_forward(self, src_input):

        img_feature = self.backbone(src_input["input_img"].cuda(), src_input["src_length_batch"])
        descript_feature = self.descriptproj(img_feature)
        inputs_embeds = torch.cat([img_feature, descript_feature], dim=-1)
        inputs_embeds = self.conv(inputs_embeds)
        inputs_embeds = self.projector(inputs_embeds)

        attention_mask = src_input["attention_mask"]

        return inputs_embeds, attention_mask

    def forward(self, src_input, tgt_input):

        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # decoder_input_ids = tgt_input['input_ids'].cuda(),
            labels=tgt_input["input_ids"].cuda(),
            decoder_attention_mask=tgt_input["attention_mask"].cuda(),
            return_dict=True,
        )
        return out["logits"]

    def generate(self, src_input, max_new_tokens, num_beams, forced_bos_token_id):

        inputs_embeds, attention_mask = self.share_forward(src_input)

        return self.mbart.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.cuda(),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            forced_bos_token_id=forced_bos_token_id,
        )


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt",
        ).get_encoder()
        for param in self.model_txt.parameters():
            param.requires_grad = False

    def forward(self, tgt_input):
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
        self.descriptproj = Projector(input_dim=backbone_dim, hidden_dim=planes, output_dim=inplanes)
        # Modality Adapter
        self.conv = TemporalConv(input_size=backbone_dim + inplanes, hidden_size=planes, conv_type=2)
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
    def __init__(self, config, embed_dim=1024, backbone="resnet18") -> None:
        super().__init__()
        self.model_text = TextEncoder()
        self.model_image = ImageEncoder(inplanes=768, planes=embed_dim, backbone=backbone)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

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
