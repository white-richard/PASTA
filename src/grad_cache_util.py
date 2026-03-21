from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grad_cache.grad_cache import GradCache


def split_src_input(model_input, chunk_size):
    """Split src_input along sample boundaries using src_length_batch."""
    lengths = model_input["src_length_batch"]
    batch_size = len(lengths)
    chunks = []

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk_lengths = lengths[start:end]

        # input_img and input_descript are flat [total_frames, ...] —
        # need to slice the correct frame ranges for this sample chunk
        frame_start = sum(lengths[:start])
        frame_end = sum(lengths[:end])

        chunk = {}
        for k, v in model_input.items():
            if k == "src_length_batch":
                chunk[k] = chunk_lengths
            elif k in ("input_img", "input_descript"):
                if isinstance(v, torch.Tensor):
                    chunk[k] = v[frame_start:frame_end]
            elif isinstance(v, (torch.Tensor, list)):
                chunk[k] = v[start:end]
            else:
                chunk[k] = v

        # Recompute attention_mask for this chunk's post-conv sequence lengths.
        # The full-batch mask was padded to the global max, but this chunk may
        # have a shorter max, causing a size mismatch in the attention layer.
        if "new_src_length_batch" in chunk:
            new_lens = chunk["new_src_length_batch"]
            max_len = int(new_lens.max())
            mask = torch.zeros(len(new_lens), max_len, dtype=torch.long)
            for i, l in enumerate(new_lens):
                mask[i, :int(l)] = 1
            chunk["attention_mask"] = mask

        chunks.append(chunk)
    return chunks


def split_tgt_input(model_input, chunk_size):
    """Split tgt_input — all values are [B, ...] tensors."""
    batch_size = next(len(v) for v in model_input.values() if isinstance(v, (torch.Tensor, list)))
    chunks = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk = {}
        for k, v in model_input.items():
            if isinstance(v, (torch.Tensor, list)):
                chunk[k] = v[start:end]
            else:
                chunk[k] = v
        chunks.append(chunk)
    return chunks


def split_input_fn(model_input, chunk_size):
    if "src_length_batch" in model_input:
        return split_src_input(model_input, chunk_size)
    return split_tgt_input(model_input, chunk_size)


class DictInputWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        # Frozen encoders produce outputs with no grad_fn, which breaks
        # GradCache's surrogate backward. A dummy param anchors the output
        # to the autograd graph without affecting values or gradients.
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
        # Accumulates auxiliary losses (e.g. descript mse_loss) from
        # GradCache's forward_backward pass so they can be backpropagated
        # separately without rerunning the full batch.
        self.aux_losses = []

    def forward(self, **kwargs):
        out = self.model(kwargs)
        if isinstance(out, tuple):
            rep = out[0] + 0.0 * self.dummy.sum()
            # During GradCache's forward_backward (grad enabled), stash
            # auxiliary losses so they can be backpropagated after gc().
            if torch.is_grad_enabled() and len(out) > 1:
                self.aux_losses.append(out[1])
            return rep, *out[1:]
        return out + 0.0 * self.dummy.sum()


class GradCacheWithAux(GradCache):
    """GradCache subclass that backprops auxiliary losses (e.g. descript mse_loss)
    captured by DictInputWrapper during the forward_backward pass."""

    def __init__(self, *args, aux_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_weight = aux_weight
        self.last_aux_loss = 0.0

    def forward_backward(
        self,
        model: nn.Module,
        model_inputs,
        cached_gradients: list[Tensor],
        random_states: list,
        no_sync_except_last: bool = False,
    ) -> None:
        has_aux = isinstance(model, DictInputWrapper) and hasattr(model, "aux_losses")
        if has_aux:
            model.aux_losses.clear()

        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(
            model_inputs, random_states, cached_gradients, sync_contexts, strict=False
        ):
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                if has_aux and model.aux_losses:
                    # retain graph so aux loss can still backward through
                    # shared intermediates (e.g. backbone features)
                    surrogate.backward(retain_graph=True)
                    aux = model.aux_losses[-1]
                    (self.aux_weight * aux / len(model_inputs)).backward()
                else:
                    surrogate.backward()

        if has_aux and model.aux_losses:
            self.last_aux_loss = sum(l.detach() for l in model.aux_losses) / len(model.aux_losses)


def split_dict_input(model_input, chunk_size):
    """Split a dict of tensors/lists into chunks along the batch dimension."""
    # Find the batch size from the first tensor-like value
    keys = list(model_input.keys())
    batch_size = len(model_input[keys[0]])

    chunks = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk = {}
        for k, v in model_input.items():
            if isinstance(v, (torch.Tensor, list)):
                chunk[k] = v[start:end]
            else:
                chunk[k] = v  # scalars/configs — pass through unchanged
        chunks.append(chunk)
    return chunks


def contrastive_loss_fn(sign_reps, text_reps, tau=0.07):
    """sign_reps, text_reps: [N, C] normalized embeddings for the FULL batch.
    GradCache assembles these from chunks before calling this.
    """
    sim = sign_reps @ text_reps.T / tau  # [N, N]
    targets = torch.arange(sim.size(0), device=sim.device)
    loss_s2t = F.cross_entropy(sim, targets)
    loss_t2s = F.cross_entropy(sim.T, targets)
    return (loss_s2t + loss_t2s) / 2.0
