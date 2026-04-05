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
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).cuda()
            self.model = torch.compile(self.model)
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
    def _forward_hidden_states(self, images: list) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device

        inputs = self.processor(
            text=[self._build_prompt()] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        pooled_container: list[torch.Tensor] = []

        def _hidden_state_hook(module, input, output) -> None:
            pooled_container.append(output)

        target_layer = self.model.language_model.layers[self.hidden_state_layer]
        hook = target_layer.register_forward_hook(_hidden_state_hook)

        self.model(**inputs, use_cache=False)
        hook.remove()

        image_token_id = self.model.config.image_token_index
        hs = pooled_container[0]  # [B, seq_len, D]

        results = []
        for b in range(len(images)):
            img_start = (
                (inputs["input_ids"][b] == image_token_id).nonzero(as_tuple=True)[0][0].item()
            )
            visual_hs = hs[b, img_start : img_start + 729, :]  # [729, D]
            results.append(visual_hs.mean(dim=0).cpu())  # [D]

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
