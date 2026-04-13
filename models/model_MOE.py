import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.utils.generation import InferenceParams
from models.model_blocks import (
    SwinEncoder,
    Pos2D,
    TransformerDecoderBlock,
    MambaDecoderBlock,
    MambaMOEDecoderBlock,
)


class MambaMOEDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model=d_model, layer_idx=0),
                MambaDecoderBlock(d_model=d_model, layer_idx=1),
                MambaMOEDecoderBlock(d_model=d_model, layer_idx=2),
                TransformerDecoderBlock(d_model=d_model, layer_idx=3),
                MambaDecoderBlock(d_model=d_model, layer_idx=4),
                MambaMOEDecoderBlock(d_model=d_model, layer_idx=5),
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self, memory, tgt, step=None, inference_params=None, cross_attn_caches=None
    ):
        x = self.embed(tgt)
        total_aux_loss = 0
        if step is None:
            seq_len = tgt.size(1)
            x = x + self.pos[:, :seq_len]
        else:
            x = x + self.pos[:, step]
        x = x.contiguous()
        if cross_attn_caches is None:
            cross_attn_caches = [None] * len(self.blocks)
        self_attn_caches = [None] * len(self.blocks)
        if inference_params is not None:
            if hasattr(inference_params, "key_value_memory_dict"):
                for i in range(len(self.blocks)):
                    if i not in inference_params.key_value_memory_dict:
                        inference_params.key_value_memory_dict[i] = {}
                    self_attn_caches[i] = inference_params.key_value_memory_dict[i]
            elif isinstance(inference_params, list):
                self_attn_caches = inference_params
            elif isinstance(inference_params, dict):
                for i in range(len(self.blocks)):
                    if i not in inference_params:
                        inference_params[i] = {}
                    self_attn_caches[i] = inference_params[i]

        for blk, c_cache, s_cache in zip(
            self.blocks, cross_attn_caches, self_attn_caches
        ):
            if isinstance(blk, MambaMOEDecoderBlock):
                x, block_aux_loss = blk(
                    x,
                    memory,
                    inference_params=inference_params,
                    cross_attn_cache=c_cache,
                    self_attn_cache=s_cache,
                )
                total_aux_loss += block_aux_loss
            else:
                x = blk(
                    x,
                    memory,
                    inference_params=inference_params,
                    cross_attn_cache=c_cache,
                    self_attn_cache=s_cache,
                )

        x = self.norm(x)
        return self.fc(x), total_aux_loss


class SwinMoETex(nn.Module):
    def __init__(self, vocab_size, d_model=512, max_len=512):
        super().__init__()
        self.encoder = SwinEncoder(d_model)
        self.pos_encoder = Pos2D(d_model=d_model)
        self.decoder = MambaMOEDecoder(vocab_size, d_model, max_len=max_len)

    def forward(self, images, tgt):
        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)
        return self.decoder(memory, tgt)

    @torch.no_grad()
    def generate(
        self,
        images,
        start_token_id,
        max_new_tokens=100,
        eos_token_id=None,
        temperature=1.0,
    ):
        self.eval()
        B = images.size(0)
        device = images.device
        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)
        if isinstance(start_token_id, int):
            current_tokens = torch.full(
                (B, 1), start_token_id, dtype=torch.long, device=device
            )
        else:
            current_tokens = start_token_id
            if current_tokens.dim() == 1:
                current_tokens = current_tokens.unsqueeze(1)

        prompt_len = current_tokens.size(1)
        max_seqlen = prompt_len + max_new_tokens

        inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=B)
        cache_dtype = memory.dtype
        if hasattr(torch, "amp") and hasattr(torch.amp, "is_autocast_enabled"):
            if torch.amp.is_autocast_enabled("cuda"):
                cache_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(torch, "is_autocast_enabled") and torch.is_autocast_enabled():
            cache_dtype = torch.get_autocast_gpu_dtype()
        else:
            try:
                cache_dtype = next(self.decoder.parameters()).dtype
            except StopIteration:
                pass
        for i, blk in enumerate(self.decoder.blocks):
            if hasattr(blk, "ssm1"):
                inference_params.key_value_memory_dict[i] = (
                    blk.ssm1.allocate_inference_cache(
                        batch_size=B, max_seqlen=max_seqlen, dtype=cache_dtype
                    )
                )
            else:
                inference_params.key_value_memory_dict[i] = {}

        cross_attn_caches = [{} for _ in range(len(self.decoder.blocks))]

        generated_tokens = []
        is_finished = torch.zeros(B, dtype=torch.bool, device=device)
        logits, _ = self.decoder(
            memory=memory,
            tgt=current_tokens,
            step=None,
            inference_params=inference_params,
            cross_attn_caches=cross_attn_caches,
        )
        next_token_logits = logits[:, -1, :] / temperature
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        generated_tokens.append(next_token)
        current_tokens = next_token

        if eos_token_id is not None:
            is_finished |= next_token.squeeze(-1) == eos_token_id

        inference_params.seqlen_offset += prompt_len
        if not is_finished.all():
            for _ in range(max_new_tokens - 1):
                logits, _ = self.decoder(
                    memory=memory,
                    tgt=current_tokens,
                    step=inference_params.seqlen_offset,
                    inference_params=inference_params,
                    cross_attn_caches=cross_attn_caches,
                )

                next_token_logits = logits[:, -1, :] / temperature
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                generated_tokens.append(next_token)
                current_tokens = next_token

                if eos_token_id is not None:
                    is_finished |= next_token.squeeze(-1) == eos_token_id
                    if is_finished.all():
                        break
                inference_params.seqlen_offset += 1

        return torch.cat(generated_tokens, dim=1)

    @torch.no_grad()
    def generate_beam_search(
        self, images, start_token_id, beam_size=3, max_new_tokens=100, eos_token_id=None
    ):
        # To unefficent???
        self.eval()
        B = images.size(0)
        device = images.device

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        B_eff = B * beam_size
        memory = memory.repeat_interleave(beam_size, dim=0)

        # Format start_token_id
        if isinstance(start_token_id, int):
            current_tokens = torch.full(
                (B, 1), start_token_id, dtype=torch.long, device=device
            )
        else:
            current_tokens = start_token_id
            if current_tokens.dim() == 1:
                current_tokens = current_tokens.unsqueeze(1)

        current_tokens = current_tokens.repeat_interleave(beam_size, dim=0)

        prompt_len = current_tokens.size(1)
        max_seqlen = prompt_len + max_new_tokens

        inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=B_eff)

        cache_dtype = memory.dtype
        if hasattr(torch, "amp") and hasattr(torch.amp, "is_autocast_enabled"):
            if torch.amp.is_autocast_enabled("cuda"):
                cache_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(torch, "is_autocast_enabled") and torch.is_autocast_enabled():
            cache_dtype = torch.get_autocast_gpu_dtype()
        else:
            try:
                cache_dtype = next(self.decoder.parameters()).dtype
            except StopIteration:
                pass

        for i, blk in enumerate(self.decoder.blocks):
            if hasattr(blk, "ssm1"):
                inference_params.key_value_memory_dict[i] = (
                    blk.ssm1.allocate_inference_cache(
                        batch_size=B_eff, max_seqlen=max_seqlen, dtype=cache_dtype
                    )
                )
            else:
                inference_params.key_value_memory_dict[i] = {}

        cross_attn_caches = [{} for _ in range(len(self.decoder.blocks))]

        def reorder_cache(cache, beam_idx):
            if isinstance(cache, torch.Tensor):
                return cache.index_select(0, beam_idx)
            elif isinstance(cache, tuple):
                return tuple(reorder_cache(c, beam_idx) for c in cache)
            elif isinstance(cache, list):
                return [reorder_cache(c, beam_idx) for c in cache]
            elif isinstance(cache, dict):
                return {k: reorder_cache(v, beam_idx) for k, v in cache.items()}
            else:
                return cache

        logits, _ = self.decoder(
            memory=memory,
            tgt=current_tokens,
            step=None,
            inference_params=inference_params,
            cross_attn_caches=cross_attn_caches,
        )

        vocab_size = logits.size(-1)
        next_token_logits = logits[:, -1, :]
        log_probs = F.log_softmax(next_token_logits, dim=-1)

        first_beam_log_probs = log_probs[::beam_size, :]

        topk_scores, topk_tokens = torch.topk(first_beam_log_probs, beam_size, dim=-1)

        beam_scores = topk_scores.view(B_eff)
        current_tokens = topk_tokens.view(B_eff, 1)

        all_generated_tokens = current_tokens.clone()

        is_finished = torch.zeros(B_eff, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            is_finished |= current_tokens.squeeze(-1) == eos_token_id

        inference_params.seqlen_offset += prompt_len

        for _ in range(max_new_tokens - 1):
            if is_finished.all():
                break

            logits, _ = self.decoder(
                memory=memory,
                tgt=current_tokens,
                step=inference_params.seqlen_offset,
                inference_params=inference_params,
                cross_attn_caches=cross_attn_caches,
            )

            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            if eos_token_id is not None:
                log_probs[is_finished, :] = -float("inf")
                log_probs[is_finished, eos_token_id] = 0.0

            next_scores = beam_scores.unsqueeze(-1) + log_probs

            next_scores = next_scores.view(B, beam_size * vocab_size)

            topk_scores, topk_indices = torch.topk(next_scores, beam_size, dim=-1)

            beam_scores = topk_scores.view(B_eff)

            prev_beam_idx = topk_indices // vocab_size
            next_tokens = topk_indices % vocab_size

            batch_offset = torch.arange(B, device=device).unsqueeze(1) * beam_size
            global_beam_idx = (prev_beam_idx + batch_offset).view(B_eff)

            current_tokens = next_tokens.view(B_eff, 1)
            all_generated_tokens = all_generated_tokens[global_beam_idx]
            all_generated_tokens = torch.cat(
                [all_generated_tokens, current_tokens], dim=1
            )

            inference_params.key_value_memory_dict = reorder_cache(
                inference_params.key_value_memory_dict, global_beam_idx
            )
            cross_attn_caches = reorder_cache(cross_attn_caches, global_beam_idx)

            is_finished = is_finished[global_beam_idx]
            if eos_token_id is not None:
                is_finished |= current_tokens.squeeze(-1) == eos_token_id

            inference_params.seqlen_offset += 1
        best_sequences = all_generated_tokens.view(B, beam_size, -1)

        return best_sequences
