import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, **kwargs):
        out = self.model(kwargs)
        if isinstance(out, tuple):
            return out[0] + 0.0 * self.dummy.sum(), *out[1:]
        return out + 0.0 * self.dummy.sum()


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
