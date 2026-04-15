from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoProcessor
from vllm import SamplingParams

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
        model_id: str = "google/gemma-4-E2B-it",
        hf_model_id: str = "google/gemma-4-E2B-it",
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

        # Processor must come from a standard HF repo (GGUF repos lack
        # preprocessor_config.json).  For hidden-state mode we use hf_model_id;
        # for vLLM mode we fall back to canonical_id since model_id may be GGUF.
        canonical_id = "google/gemma-4-26B-A4B-it"
        processor_id = hf_model_id if extract_hidden_states else canonical_id
        self.processor = AutoProcessor.from_pretrained(processor_id)
        # Disable pan-and-scan / multi-crop so pixel_values stays 4-D (B, C, H, W).
        if hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
            for attr in ("do_image_splitting", "do_pan_and_scan"):
                if hasattr(ip, attr):
                    setattr(ip, attr, do_image_splitting)
        if extract_hidden_states:
            from accelerate import init_empty_weights, infer_auto_device_map
            from transformers import AutoConfig, BitsAndBytesConfig, Gemma4ForConditionalGeneration

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )

            # device_map="auto" causes bitsandbytes 4-bit to reject CPU-dispatched
            # modules even with llm_int8_enable_fp32_cpu_offload=True; a custom dict
            # is required.  infer_auto_device_map estimates sizes in fp16, but 4-bit
            # weights are ~4x smaller, so we scale the GPU budget by 4 to keep all
            # quantized linear layers on CUDA.  Any remaining modules (norms, embeds)
            # that don't fit will be offloaded to CPU in fp32 via the flag above.
            config = AutoConfig.from_pretrained(hf_model_id)
            with init_empty_weights():
                empty = Gemma4ForConditionalGeneration(config)
            total_vram = torch.cuda.get_device_properties(0).total_memory
            device_map = infer_auto_device_map(
                empty,
                max_memory={0: total_vram * 4, "cpu": 200 * 1024**3},
                no_split_module_classes=["Gemma4DecoderLayer"],
                dtype=torch.float16,
            )
            del empty
            torch.cuda.empty_cache()

            self.model = Gemma4ForConditionalGeneration.from_pretrained(
                hf_model_id,
                quantization_config=quant_config,
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder="offload",
            )
            self.model.eval()
        else:
            self.model = LLM(
                model=model_id,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
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

    @torch.no_grad()
    def _forward_hidden_states(self, images: list) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device

        # Gemma4's processor requires images as a list-of-lists: one inner list
        # per text prompt. A flat list is interpreted as multiple frames for a
        # single prompt, causing a batch-size mismatch when len(images) > 1.
        inputs = self.processor(
            text=[self._build_prompt()] * len(images),
            images=[[img] for img in images],
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Hook-based extraction: only one layer's tensor is ever live in VRAM,
        # unlike output_hidden_states=True which materialises all N+1 layers.
        # Since torch.compile is not used here, hooks have no graph-break cost.
        pooled_container: list[torch.Tensor] = []

        def _hidden_state_hook(module, input, output) -> None:
            # output may be a tuple (hidden, ...) depending on layer type.
            hs = output[0] if isinstance(output, tuple) else output
            pooled_container.append(hs)

        # Gemma4ForConditionalGeneration
        #   .model          → Gemma4Model
        #   .language_model → Gemma4TextModel
        #   .layers[N]      → Gemma4TextDecoderLayer
        target_layer = self.model.model.language_model.layers[self.hidden_state_layer]
        hook = target_layer.register_forward_hook(_hidden_state_hook)
        try:
            self.model(**inputs, use_cache=False)
        finally:
            hook.remove()

        hs = pooled_container[0]  # (B, seq_len, D)

        # image_token_index is the standard name; fall back to image_token_id if needed.
        image_token_id = getattr(
            self.model.config,
            "image_token_index",
            getattr(self.model.config, "image_token_id", None),
        )

        results = []
        for b in range(len(images)):
            # Locate the visual token span dynamically — Gemma4's token count
            # depends on resolution and whether pan-and-scan is enabled.
            positions = (inputs["input_ids"][b] == image_token_id).nonzero(as_tuple=True)[0]
            n_visual = len(positions)
            img_start = positions[0].item()
            visual_hs = hs[b, img_start : img_start + n_visual, :]  # (n_visual, D)
            results.append(visual_hs.mean(dim=0).cpu())  # (D,)

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
