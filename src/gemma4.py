from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from collections.abc import Iterable


class Gemma4(nn.Module):
    """Gemma 4 multimodal wrapper — same interface as LLaVA in llavaov.py.

    forward(images) → list[str]            text-generation mode
    forward(images) → list[torch.Tensor]   hidden-state extraction mode

    Architecture note (hidden-state mode):
      LLM decoder layers live at model.model.language_model.model.layers.
      Visual token count is detected dynamically from input_ids so the code
      works regardless of whether pan-and-scan is enabled.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-4b-it",
        max_model_len: int = 8192,
        max_tokens: int = 80,
        use_turboquant: bool = False,
        extract_hidden_states: bool = False,
        hidden_state_layer: int = 20,
        do_image_splitting: bool = False,
    ) -> None:
        super().__init__()

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.extract_hidden_states = extract_hidden_states
        self.hidden_state_layer = hidden_state_layer

        self.processor = AutoProcessor.from_pretrained(model_id)
        # Disable pan-and-scan / multi-crop so pixel_values stays 4-D (B, C, H, W).
        if hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
            for attr in ("do_image_splitting", "do_pan_and_scan"):
                if hasattr(ip, attr):
                    setattr(ip, attr, do_image_splitting)

        if extract_hidden_states:
            from transformers import Gemma4ForConditionalGeneration

            self.model = Gemma4ForConditionalGeneration.from_pretrained(
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

        # Gemma4 LLM decoder layers: model.model.language_model.model.layers
        target_layer = self.model.model.language_model.model.layers[self.hidden_state_layer]
        hook = target_layer.register_forward_hook(_hidden_state_hook)

        self.model(**inputs, use_cache=False)
        hook.remove()

        # image_token_index is the standard name; fall back to image_token_id if needed.
        image_token_id = getattr(
            self.model.config,
            "image_token_index",
            getattr(self.model.config, "image_token_id", None),
        )
        hs = pooled_container[0]  # (B, seq_len, D)

        results = []
        for b in range(len(images)):
            # Locate the visual token span dynamically — Gemma4's token count
            # depends on resolution and whether pan-and-scan is enabled.
            positions = (inputs["input_ids"][b] == image_token_id).nonzero(as_tuple=True)[0]
            n_visual = len(positions)
            img_start = positions[0].item()
            visual_hs = hs[b, img_start : img_start + n_visual, :]  # (n_visual, D)
            results.append(visual_hs.mean(dim=0).cpu())              # (D,)

        return results

    def forward(self, images: Iterable) -> list:
        images = list(images)
        if not images:
            return []

        if self.extract_hidden_states:
            return self._forward_hidden_states(images)

        prompt = self._build_prompt()
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in images]
        outputs = self.model.generate(inputs, sampling_params=self.params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]
