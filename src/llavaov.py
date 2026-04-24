from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from collections.abc import Iterable


class LLaVA(nn.Module):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",  # llava-hf/llava-v1.6-mistral-7b-hf | llava-hf/llava-onevision-qwen2-7b-ov-hf
        max_model_len: int = 8192,
        max_tokens: int = 80,
        use_turboquant: bool = True,
        extract_hidden_states: bool = False,
        hidden_state_layer: int = 20,  # which Qwen2 layer, 0-indexed (0–27 for 7B)
        do_image_splitting=False,
    ) -> None:
        super().__init__()

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.extract_hidden_states = extract_hidden_states
        self.hidden_state_layer = hidden_state_layer
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.processor.image_processor.do_image_splitting = do_image_splitting

        if extract_hidden_states:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
        else:
            self.model = LLM(
                model=model_id,
                dtype="bfloat16",
                max_model_len=max_model_len,
                attention_config={"backend": "CUSTOM"} if use_turboquant else None,
                enable_prefix_caching=True,
                limit_mm_per_prompt={"image": 1},
            )

        self.params = SamplingParams(max_tokens=max_tokens)

    def _build_prompt(self) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Describe only the motion and gestures of the person "
                            "in the image. Focus on hands and face."
                        ),
                    },
                ],
            },
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def _forward_hidden_states(self, images: list) -> list[tuple[torch.Tensor, torch.Tensor]]:
        inputs = self.processor(
            text=[self._build_prompt()] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        mid_container: list[torch.Tensor] = []
        last_container: list[torch.Tensor] = []

        def _make_hook(container):
            def _hook(module, input, output) -> None:
                hs = output[0] if isinstance(output, tuple) else output
                container.append(hs)

            return _hook

        # LlavaOnevisionForConditionalGeneration
        #   .model          → LlavaOnevisionModel
        #   .language_model → Qwen2Model
        #   .layers[N]      → Qwen2DecoderLayer
        layers = self.model.model.language_model.layers
        hook_mid = layers[self.hidden_state_layer].register_forward_hook(_make_hook(mid_container))
        hook_last = layers[-1].register_forward_hook(_make_hook(last_container))

        try:
            self.model(**inputs, use_cache=False)
        finally:
            hook_mid.remove()
            hook_last.remove()

        image_token_id = self.model.config.image_token_index
        mid_hs = mid_container[0]  # [B, seq_len, D]
        last_hs = last_container[0]  # [B, seq_len, D]

        results = []
        for b in range(len(images)):
            img_start = (
                (inputs["input_ids"][b] == image_token_id).nonzero(as_tuple=True)[0][0].item()
            )
            mid_visual = mid_hs[b, img_start : img_start + 729, :].mean(dim=0).cpu()  # [D]
            last_visual = last_hs[b, img_start : img_start + 729, :].mean(dim=0).cpu()  # [D]
            results.append((mid_visual, last_visual))

        return results

    def forward(self, images: Iterable) -> list[str]:
        images = list(images)
        if not images:
            return []

        if self.extract_hidden_states:
            return self._forward_hidden_states(images)

        prompt = self._build_prompt()
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in images]
        outputs = self.model.generate(inputs, sampling_params=self.params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]
