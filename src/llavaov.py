from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from collections.abc import Iterable


class LLaVA(nn.Module):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        max_model_len: int = 8192,
        max_tokens: int = 80,
    ) -> None:
        super().__init__()

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model = LLM(
            model=model_id,
            dtype="bfloat16",
            max_model_len=max_model_len,
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

    def forward(self, images: Iterable) -> list[str]:
        images = list(images)
        if not images:
            return []

        prompt = self._build_prompt()
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in images]
        outputs = self.model.generate(inputs, sampling_params=self.params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]
