"""In-process compatibility shim for mlx-vlm's Qwen3-Omni audio tower.

mlx-vlm's ``qwen3_omni_moe.AudioModel.__call__`` ships with two upstream
defects that we patch in place when the running mlx-vlm version predates the
fixes:

1. ``_get_feat_extract_output_lengths`` returns 14 (not 13) for full
   100-frame mel chunks because the ``input_lengths_leave == 0`` branch adds a
   stray "+1". When used per-chunk this gives the wrong token count.

2. ``AudioModel.__call__`` builds the post-CNN attention mask and the
   ``cu_seqlens`` list from the *per-sample* ``feature_lens_after_cnn`` instead
   of the per-chunk lengths it actually padded. Only chunk 0 of the mask is
   filled, so chunks 1..N emit zero audio embeddings — the model "hears"
   ~2 s of audio and the rest is zero-padding.

The fix is harmless when the upstream mlx-vlm corrects the bug — we just
check for a marker attribute and skip re-patching. We also bail out early on
mlx-vlm >= 0.6.10, where both fixes have landed upstream.

This module is intentionally Apple-Silicon-only at runtime. The heavy imports
are deferred to the function body so the rest of ``bimomni.inference`` still
imports cleanly on Linux CI.
"""

from __future__ import annotations

import importlib.metadata
import logging

log = logging.getLogger(__name__)

AUDIO_PATCH_MARKER = "_bimomni_audio_patch_v1"


def ensure_audio_patch() -> None:
    """Patch mlx_vlm's Qwen3-Omni audio tower in-place if needed.

    Idempotent — skips when the marker attribute is already set, which keeps
    the patch safe across re-imports and lets upstream fixes land without our
    code blowing up.
    """
    try:
        from mlx_vlm.models import qwen3_omni_moe  # noqa: F401  (validates import)
        from mlx_vlm.models.qwen3_omni_moe import audio as _audio
    except ImportError as exc:
        raise RuntimeError(
            "mlx_vlm is required for the Qwen3-Omni audio patch (Apple Silicon only)"
        ) from exc

    version = tuple(
        int(part) for part in importlib.metadata.version("mlx-vlm").split(".")[:3]
    )
    if version >= (0, 6, 10):
        return

    if getattr(_audio, AUDIO_PATCH_MARKER, False):
        return

    import mlx.core as mx
    import numpy as np

    def _patched_get_feat_extract_output_lengths(input_lengths):
        input_lengths_leave = input_lengths % 100
        if input_lengths_leave == 0:
            return (input_lengths // 100) * 13
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
            + (input_lengths // 100) * 13
        )
        return output_lengths

    def _patched_call(
        self,
        input_features: mx.array,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        if input_features.ndim == 3:
            input_features = input_features[0]
        if feature_lens is None:
            feature_lens = mx.array([input_features.shape[-1]], dtype=mx.int32)

        _aftercnn_lens = _patched_get_feat_extract_output_lengths(feature_lens)
        feature_lens_np = np.array(feature_lens).astype(np.int32)
        n_window_step = self.n_window * 2
        chunk_num = np.ceil(feature_lens_np / n_window_step).astype(np.int32)

        chunk_lengths_list: list[int] = []
        tail_chunk_info: list[tuple[int, int]] = []
        cumsum = 0
        for sample_idx, num_chunks in enumerate(chunk_num.tolist()):
            num_int = int(num_chunks)
            chunk_lengths_list.extend([n_window_step] * num_int)
            if num_int > 0:
                tail_chunk_info.append((cumsum + num_int - 1, sample_idx))
            cumsum += num_int

        for tail_idx, sample_idx in tail_chunk_info:
            remainder = feature_lens_np[sample_idx] % n_window_step
            if remainder == 0:
                remainder = n_window_step
            chunk_lengths_list[tail_idx] = int(remainder)

        chunk_lengths = mx.array(chunk_lengths_list, dtype=mx.int32)
        total_chunks = len(chunk_lengths_list)
        max_chunk_len = int(chunk_lengths.max())
        # Build padded_feature by concatenating per-chunk slices (avoids an
        # MLX in-place assignment bug at chunk boundaries when the tail chunk
        # is shorter than ``max_chunk_len``). On older MLX the in-place form
        # ``padded_feature[i, :, :chunk_len] = sl`` succeeds; on the version
        # pinned here it raises ``broadcast_shapes (1, 128, max) vs (128, n)``.
        chunk_arrays = []
        start_idx = 0
        for chunk_len in chunk_lengths_list:
            end_idx = start_idx + chunk_len
            chunk = input_features[:, start_idx:end_idx]
            if chunk.shape[-1] < max_chunk_len:
                pad = mx.zeros(
                    (self.num_mel_bins, max_chunk_len - chunk.shape[-1]),
                    dtype=input_features.dtype,
                )
                chunk = mx.concatenate([chunk, pad], axis=-1)
            chunk_arrays.append(chunk)
            start_idx = end_idx
        padded_feature = mx.stack(chunk_arrays, axis=0)
        padded_feature = padded_feature[:, None, :, :]

        chunk_aftercnn_lens_np = np.array(
            [
                int(_patched_get_feat_extract_output_lengths(mx.array([cl]))[0])
                for cl in chunk_lengths_list
            ],
            dtype=np.int32,
        )
        max_len_after_cnn = int(chunk_aftercnn_lens_np.max())
        padded_mask_after_cnn = mx.zeros(
            (total_chunks, max_len_after_cnn), dtype=mx.bool_
        )
        for i, length in enumerate(chunk_aftercnn_lens_np.tolist()):
            padded_mask_after_cnn[i, :length] = True

        import mlx.nn as nn

        padded_embeds = []
        for i in range(0, total_chunks, self.conv_chunksize):
            end_idx = min(i + self.conv_chunksize, total_chunks)
            chunk = padded_feature[i:end_idx].transpose(0, 2, 3, 1)
            padded_embed = nn.gelu(self.conv2d1(chunk))
            padded_embed = nn.gelu(self.conv2d2(padded_embed))
            padded_embed = nn.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)

        padded_embed = mx.concatenate(padded_embeds, axis=0)
        b, h, w, c = padded_embed.shape
        padded_embed = padded_embed.transpose(0, 2, 3, 1).reshape(b, w, c * h)
        padded_embed = self.conv_out(padded_embed)

        seq_len = padded_embed.shape[1]
        positional_embedding = self.positional_embedding(seq_len)[None]
        padded_embed = padded_embed + positional_embedding

        linear_indices: list[int] = []
        for i in range(total_chunks):
            mask_array = np.array(padded_mask_after_cnn[i])
            chunk_indices = np.where(mask_array)[0]
            linear_indices.extend([i * seq_len + idx for idx in chunk_indices])

        padded_embed_flat = padded_embed.reshape(-1, padded_embed.shape[-1])
        hidden_states = mx.take(
            padded_embed_flat,
            mx.array(np.array(linear_indices, dtype=np.int32)),
            axis=0,
        )

        chunks_per_window = max(self.n_window_infer // (self.n_window * 2), 1)
        cu_chunk_lens: list[int] = []
        window_tokens = 0
        for chunk_tokens in chunk_aftercnn_lens_np.tolist():
            if window_tokens >= chunks_per_window * 13:
                cu_chunk_lens.append(window_tokens)
                window_tokens = 0
            window_tokens += int(chunk_tokens)
        if window_tokens > 0:
            cu_chunk_lens.append(window_tokens)

        cu_seqlens = mx.cumsum(mx.array(cu_chunk_lens, dtype=mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), constant_values=0)

        for i, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, cu_seqlens)
            if i % 2 == 0:
                mx.eval(hidden_states)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    _audio._get_feat_extract_output_lengths = _patched_get_feat_extract_output_lengths
    _audio.AudioModel.__call__ = _patched_call
    setattr(_audio, AUDIO_PATCH_MARKER, True)
    log.info("patched mlx_vlm qwen3_omni_moe.audio in-process")
