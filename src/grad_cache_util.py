from contextlib import nullcontext

import torch
import torch.nn.functional as F
from grad_cache.grad_cache import GradCache
from torch import Tensor, nn


def split_src_input(model_input, chunk_size):
    """Split src_input along sample boundaries using src_length_batch."""
    lengths = model_input["src_length_batch"]
    batch_size = len(lengths)
    # Bug 1 fix: compute cumsum once (O(B)) instead of re-summing each iteration (O(B²))
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


def contrastive_loss_fn(sign_reps, text_reps, temperature=0.07):
    """sign_reps, text_reps: [N, C] normalized embeddings for the FULL batch.
    GradCache assembles these from chunks before calling this.
    """
    sim = sign_reps @ text_reps.T / temperature  # [N, N]
    targets = torch.arange(sim.size(0), device=sim.device)
    loss_s2t = F.cross_entropy(sim, targets)
    loss_t2s = F.cross_entropy(sim.T, targets)
    return (loss_s2t + loss_t2s) / 2.0


# Bug 8 fix: shared base for the dummy-param anchor pattern
class _AnchoredEncoder(nn.Module):
    """Base class that anchors frozen encoder outputs to the autograd graph.

    A dummy param anchors the output to the autograd graph without affecting
    values or gradients, so GradCache's surrogate backward works even when the
    encoder has no non-frozen params with requires_grad.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)


class DictInputWrapper(_AnchoredEncoder):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
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
    captured by DictInputWrapper during the forward_backward pass.
    """

    def __init__(self, *args, aux_weight=0.1, **kwargs) -> None:
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

        # Bug 4 fix: weight aux loss by fraction of actual samples, not by 1/num_chunks,
        # so the last (possibly smaller) chunk is weighted correctly.
        def _chunk_n(x):
            if "src_length_batch" in x:
                return len(x["src_length_batch"])
            return len(next(v for v in x.values() if isinstance(v, (torch.Tensor, list))))

        total_n = sum(_chunk_n(x) for x in model_inputs) if has_aux else 1

        # Bug 5 fix: ensure aux_losses is cleared even if the loop raises.
        try:
            for x, state, gradient, sync_context in zip(
                model_inputs,
                random_states,
                cached_gradients,
                sync_contexts,
                strict=False,
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
                        chunk_n = _chunk_n(x)
                        (self.aux_weight * aux * (chunk_n / total_n)).backward()
                    else:
                        surrogate.backward()
        finally:
            if has_aux:
                # Store averaged aux loss before clearing, so __call__ can read it.
                if model.aux_losses:
                    self.last_aux_loss = sum(
                        l.detach() for l in model.aux_losses
                    ) / len(model.aux_losses)
                model.aux_losses.clear()


class _AlignOnlyWrapper(_AnchoredEncoder):
    """Thin wrapper around GMMLPImageEncoder for alignment-only GradCache.

    Returns sentence_emb only (drops student_vid). Used for the InfoNCE
    grounding path where the grounding loss is computed in a separate pass
    after the alignment GradCache step.

    Not a DictInputWrapper subclass so GradCacheWithAux treats it as a plain
    encoder (no aux-loss side-channel).
    """

    def __init__(self, img_encoder: nn.Module) -> None:
        super().__init__()
        self.model = img_encoder

    def forward(self, **kwargs) -> Tensor:
        out = self.model(kwargs)  # (sentence_emb, student_vid)
        sentence_emb = out[0] if isinstance(out, tuple) else out
        return sentence_emb + 0.0 * self.dummy.sum()


class _GroundingAuxWrapper(DictInputWrapper):
    """Wraps GMMLPImageEncoder to compute MSE grounding loss as a per-chunk aux loss.

    Used with GradCacheWithAux for the 'mse' grounding path.

    forward(**kwargs) → sentence_emb  (for GradCache's contrastive alignment)

    Side-effect during the with-grad second pass:
        Computes MSE(normalize(student_vid), normalize(kwargs["grounding_feats"]))
        and appends to self.aux_losses. GradCacheWithAux then backpropagates
        this loss scaled by aux_weight (= lambda_ground).

    MSE is per-sample separable, so computing it per chunk and averaging is
    mathematically equivalent to computing it on the full batch.
    """

    # Bug 3 fix: removed `temperature` parameter — it was unused for MSE and
    # created a footgun if the class were reused for an InfoNCE path.
    def __init__(self, img_encoder: nn.Module) -> None:
        super().__init__(img_encoder)

    def forward(self, **kwargs) -> Tensor:
        out = self.model(kwargs)  # (sentence_emb, student_vid)
        sentence_emb, student_vid = out
        sentence_emb = sentence_emb + 0.0 * self.dummy.sum()

        if torch.is_grad_enabled():
            device, dtype = student_vid.device, student_vid.dtype
            teacher = F.normalize(
                kwargs["grounding_feats"].to(device, dtype=dtype), dim=-1
            )
            student_norm = F.normalize(student_vid, dim=-1)
            self.aux_losses.append(F.mse_loss(student_norm, teacher))

        return sentence_emb  # sentence_emb only — not a tuple


class GradCacheWithGrounding:
    """Gradient-cached training step for GMMLP combining L_align and L_ground.

    L_align (contrastive InfoNCE, sentence_emb vs siglip_feat):
        Always handled by GradCache via GradCacheWithAux.  The image encoder
        is called in chunks; representations are assembled into a full-batch
        contrastive loss whose gradient is cached and replayed.

    L_ground (student_vid vs pre-extracted grounding_feats):
        'mse'     — MSE per chunk, backpropagated as an aux loss inside the
                    GradCache second pass (_GroundingAuxWrapper).  MSE is
                    separable across samples so per-chunk averaging is exact.
        'infonce' — Full-batch InfoNCE.  After the alignment GradCache pass,
                    a two-pass GradCache-style loop collects cached reps
                    no-grad, computes the gradient w.r.t. those reps, then
                    replays each chunk with grad and backprops immediately.
                    This ensures all negative pairs are in scope without
                    holding all activations in memory simultaneously.

    __call__(src_input, tgt_input) → (align_val: float, ground_val: float)
        Both values are unscaled (lambda_ground not applied); the caller is
        responsible for scaling ground_val when computing a display loss.
        Note: for InfoNCE grounding the backpropagated gradient IS scaled by
        lambda_ground internally (via the surrogate), even though the returned
        float is unscaled.
    """

    def __init__(
        self,
        img_encoder: nn.Module,
        txt_encoder: nn.Module,
        img_chunk_size: int,
        txt_chunk_size: int,
        align_loss_fn,
        split_src_fn,
        split_tgt_fn,
        lambda_ground: float,
        temperature: float,
        ground_loss_type: str,
        device: torch.device,
    ) -> None:
        self.img_encoder = img_encoder
        self.img_chunk_size = img_chunk_size
        self.split_src_fn = split_src_fn
        self.lambda_ground = lambda_ground
        self.temperature = temperature
        self.ground_loss_type = ground_loss_type
        self.device = device

        txt_wrapper = DictInputWrapper(txt_encoder)

        def _split_fn(inp, cs):
            if "images" in inp or "vis_feats" in inp:
                return split_src_fn(inp, cs)
            return split_tgt_fn(inp, cs)

        def _align_fn(a, b):
            return align_loss_fn(a, b, temperature)

        if ground_loss_type == "mse" and lambda_ground > 0.0:
            # MSE is separable — compute per-chunk as an aux loss.
            # Bug 3 fix: no longer passing temperature (unused for MSE).
            self._img_wrapper: nn.Module = _GroundingAuxWrapper(img_encoder)
            aux_weight = lambda_ground
        else:
            # InfoNCE grounding needs full-batch negatives, or grounding is disabled.
            self._img_wrapper = _AlignOnlyWrapper(img_encoder)
            aux_weight = 0.0

        self._gc = GradCacheWithAux(
            models=[self._img_wrapper, txt_wrapper],
            chunk_sizes=[img_chunk_size, txt_chunk_size],
            loss_fn=_align_fn,
            split_input_fn=_split_fn,
            get_rep_fn=lambda out: out[0] if isinstance(out, tuple) else out,
            aux_weight=aux_weight,
            device=device,
        )

    def __call__(self, src_input: dict, tgt_input: dict) -> tuple[float, float]:
        """Run one training step.

        Returns:
            align_val:  Unscaled alignment InfoNCE loss (float).
            ground_val: Unscaled grounding loss (float).
        """
        align_loss = self._gc(src_input, tgt_input)
        align_val = align_loss.item()

        if self.ground_loss_type == "mse":
            # MSE already backpropagated inside GradCache; just read the value.
            # Bug 6 fix: guard against 0.0 sentinel on first step.
            ground_val = float(self._gc.last_aux_loss) if self._gc.last_aux_loss else 0.0
        else:
            # InfoNCE grounding: separate two-pass GradCache-style step.
            ground_val = self._run_infonce_grounding(src_input)

        return align_val, ground_val

    def _run_infonce_grounding(self, src_input: dict) -> float:
        """Two-pass GradCache for InfoNCE grounding loss.

        Bug 2 fix: replaces the single-pass approach that held all activations
        simultaneously. Now mirrors GradCache's pattern:
          Pass 1 (no_grad): collect cached normalised student_vid reps.
          Compute InfoNCE gradient w.r.t. those cached reps.
          Pass 2 (with_grad): replay each chunk; backprop via surrogate
            dot(reps_chunk, grad_chunk) scaled by lambda_ground, then free
            each chunk's graph immediately.

        Returns the unscaled grounding loss float (lambda_ground is applied
        to the surrogate during backprop, not to the returned value).
        """
        chunks = self.split_src_fn(src_input, self.img_chunk_size)
        device = self.device

        # Pass 1: no-grad — collect cached normalised reps
        with torch.no_grad():
            cached_reps: list[Tensor] = []
            for chunk in chunks:
                _, student_vid = self.img_encoder(chunk)
                cached_reps.append(F.normalize(student_vid.float(), dim=-1))

        reps_full = torch.cat(cached_reps, dim=0)  # (B, D)
        dtype = reps_full.dtype
        teacher = F.normalize(
            src_input["grounding_feats"].to(device, dtype=dtype), dim=-1  # type: ignore[index]
        )

        # Compute InfoNCE loss and gradient w.r.t. cached reps only
        reps_for_grad = reps_full.detach().requires_grad_(True)
        ground_loss = contrastive_loss_fn(reps_for_grad, teacher, self.temperature)
        ground_loss.backward()
        grad_reps = reps_for_grad.grad  # (B, D)

        # Pass 2: with-grad — surrogate per chunk, backward immediately
        offset = 0
        for chunk in chunks:
            _, student_vid = self.img_encoder(chunk)
            chunk_n = student_vid.shape[0]
            reps_chunk = F.normalize(student_vid, dim=-1)
            grad_chunk = grad_reps[offset : offset + chunk_n].detach()
            # Scale by lambda_ground so the optimiser sees ∂(λ·L_ground)/∂params
            surrogate = self.lambda_ground * torch.dot(
                reps_chunk.flatten(), grad_chunk.flatten()
            )
            surrogate.backward()
            offset += chunk_n

        return ground_loss.detach().item()
