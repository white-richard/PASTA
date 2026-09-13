import torch
import torch.nn.functional as F
from torch import Tensor, nn


def split_src_input(model_input, chunk_size):
    """Split src_input along sample boundaries using src_length_batch."""
    lengths = model_input["src_length_batch"]
    batch_size = len(lengths)
    cumsum = lengths.cumsum(0)
    chunks = []

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk_lengths = lengths[start:end]

        frame_start = int(cumsum[start - 1]) if start > 0 else 0
        frame_end = int(cumsum[end - 1])

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
                mask[i, : int(l)] = 1
            chunk["attention_mask"] = mask

        chunks.append(chunk)
    return chunks


def split_tgt_input(model_input, chunk_size):
    """Split tgt_input -- all values are [B, ...] tensors."""
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


def contrastive_loss_fn(sign_reps, text_reps, temperature=0.07):
    """sign_reps, text_reps: [N, C] normalized embeddings for the FULL batch.
    GradCache assembles these from chunks before calling this.
    """
    sim = sign_reps @ text_reps.T / temperature  # [N, N]
    targets = torch.arange(sim.size(0), device=sim.device)
    loss_s2t = F.cross_entropy(sim, targets)
    loss_t2s = F.cross_entropy(sim.T, targets)
    return (loss_s2t + loss_t2s) / 2.0


def filip_loss_fn(v_tokens: Tensor, t_tokens: Tensor, temperature: float = 0.07) -> Tensor:
    """FILIP symmetric contrastive loss (ICLR 2022).

    v_tokens: (N, K, D) — L2-normalised video token reps (Perceiver latents).
    t_tokens: (N, L, D) — L2-normalised text token reps, zero-padded to batch
              max L.  Padded positions are detected by near-zero L2-norm.

    Score(v_i, t_j):
      v→t: mean_l  max_k  cos(v_i_k, t_j_l)   [text tokens are queries]
      t→v: mean_k  max_l  cos(v_i_k, t_j_l)   [video tokens are queries]
    Symmetric loss = (CE(S_vt, diag) + CE(S_tv.T, diag)) / 2.
    """
    # (N, N, K, L) — all cross-sample token similarities
    sim = torch.einsum("ikd,jld->ijkl", v_tokens, t_tokens)

    # Padding mask for text: real tokens have unit norm; padded zeros map to zero.
    t_mask = t_tokens.norm(dim=-1) > 1e-6  # (N, L) bool

    # v->t: for each text token l find best video token k, then mean over l
    # sim.max(dim=2).values: (N_v, N_t, L)
    max_over_k = sim.max(dim=2).values  # (Nv, Nt, L)
    t_mask_j = t_mask.unsqueeze(0).float()  # (1, Nt, L)  — broadcast over Nv
    sim_vt = (max_over_k * t_mask_j).sum(dim=-1) / t_mask_j.sum(dim=-1).clamp(min=1)

    # t->v: for each video token k find best real text token l, then mean over k
    # Mask padded text positions to -inf before taking max.
    sim_tv_masked = sim.masked_fill(
        ~t_mask.unsqueeze(0).unsqueeze(2),
        float("-inf"),
    )  # (Nv, Nt, K, L)
    max_over_l = sim_tv_masked.max(dim=3).values  # (Nv, Nt, K)
    sim_tv = max_over_l.mean(dim=-1)  # (Nv, Nt)

    sim_vt = sim_vt / temperature
    sim_tv = sim_tv / temperature

    targets = torch.arange(v_tokens.shape[0], device=v_tokens.device)
    loss_vt = F.cross_entropy(sim_vt, targets)
    loss_tv = F.cross_entropy(sim_tv.T, targets)
    return (loss_vt + loss_tv) / 2


class _AnchoredEncoder(nn.Module):
    """Base class that anchors frozen encoder outputs to the autograd graph.

    A dummy param anchors the output to the autograd graph without affecting
    values or gradients, so GradCache's surrogate backward works even when the
    encoder has no non-frozen params with requires_grad.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
