import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_blocks import SwinEncoder, Pos2D, TransformerDecoderBlock


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, depth=8, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))

        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(d_model, layer_idx=i) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self, memory, tgt, step=None, inference_params=None, cross_attn_caches=None
    ):
        x = self.embed(tgt)

        if step is None:
            seq_len = tgt.size(1)
            x = x + self.pos[:, :seq_len]
        else:
            x = x + self.pos[:, step : step + 1]

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
            x = blk(x, memory, cross_attn_cache=c_cache, self_attn_cache=s_cache)

        x = self.norm(x)
        return self.fc(x)


class SwinTransformerTex(nn.Module):
    def __init__(self, vocab_size, d_model=512, decoder_depth=7, max_len=512):
        super().__init__()
        self.encoder = SwinEncoder(d_model)
        self.pos_encoder = Pos2D(d_model=d_model)
        self.decoder = TransformerDecoder(
            vocab_size, d_model, depth=decoder_depth, max_len=max_len
        )

    def forward(self, images, tgt):
        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)
        return self.decoder(memory, tgt), (H, W)

    @torch.no_grad()
    @torch.compiler.disable
    def generate(
        self, images, start_token_id, eos_token_id, max_new_tokens=256, device="cuda"
    ):
        self.eval()

        if isinstance(start_token_id, int):
            start_token_id = torch.full(
                (images.size(0), 1),
                start_token_id,
                dtype=torch.long,
                device=images.device,
            )

        elif isinstance(start_token_id, torch.Tensor) and start_token_id.dim() == 1:
            start_token_id = start_token_id.unsqueeze(1)

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        cross_attn_caches = [{} for _ in self.decoder.blocks]
        self_attn_caches = [{} for _ in self.decoder.blocks]

        logits = self.decoder(
            memory,
            start_token_id,
            step=None,
            inference_params=self_attn_caches,
            cross_attn_caches=cross_attn_caches,
        )

        tokens = start_token_id
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

        tokens = torch.cat([tokens, next_token], dim=1)

        max_pos_len = self.decoder.pos.size(1)
        target_steps = start_token_id.size(1) + max_new_tokens - 1
        limit = min(target_steps, max_pos_len)

        for step in range(start_token_id.size(1), limit):
            logits = self.decoder(
                memory,
                tokens[:, -1:],
                step=step,
                inference_params=self_attn_caches,
                cross_attn_caches=cross_attn_caches,
            )

            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return tokens

    @torch.no_grad()
    @torch.compiler.disable
    def generate_beam_search(
        self, images, start_token_id, beam_size=3, max_new_tokens=100, eos_token_id=None
    ):
        self.eval()

        B = images.size(0)
        device = images.device

        if isinstance(start_token_id, int):
            start_token_id = torch.full(
                (B, 1), start_token_id, dtype=torch.long, device=device
            )
        elif isinstance(start_token_id, torch.Tensor) and start_token_id.dim() == 1:
            start_token_id = start_token_id.unsqueeze(1)

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        memory = (
            memory.unsqueeze(1)
            .expand(-1, beam_size, -1, -1)
            .reshape(B * beam_size, memory.size(1), memory.size(2))
        )

        tokens = (
            start_token_id.unsqueeze(1)
            .expand(-1, beam_size, -1)
            .reshape(B * beam_size, -1)
        )
        scores = torch.zeros((B, beam_size), dtype=torch.float, device=device)
        scores[:, 1:] = -1e9
        scores = scores.view(-1)

        cross_attn_caches = [{} for _ in self.decoder.blocks]
        self_attn_caches = [{} for _ in self.decoder.blocks]

        def reorder_cache(caches, indices):
            for cache in caches:
                if "k" in cache:
                    cache["k"] = cache["k"][indices]
                if "v" in cache:
                    cache["v"] = cache["v"][indices]

        logits = self.decoder(
            memory,
            tokens,
            step=None,
            inference_params=self_attn_caches,
            cross_attn_caches=cross_attn_caches,
        )
        vocab_size = logits.size(-1)

        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        next_scores = scores.unsqueeze(1) + log_probs
        next_scores = next_scores.view(B, beam_size * vocab_size)

        topk_scores, topk_indices = next_scores.topk(beam_size, dim=1)
        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size

        batch_indices = torch.arange(B, device=device).unsqueeze(1)
        batch_beam_indices = (batch_indices * beam_size + beam_indices).view(-1)

        tokens = torch.cat(
            [tokens[batch_beam_indices], token_indices.view(-1, 1)], dim=1
        )
        scores = topk_scores.view(-1)

        reorder_cache(self_attn_caches, batch_beam_indices)
        reorder_cache(cross_attn_caches, batch_beam_indices)

        max_pos_len = self.decoder.pos.size(1)
        target_steps = start_token_id.size(1) + max_new_tokens - 1
        limit = min(target_steps, max_pos_len)

        for step in range(start_token_id.size(1), limit):
            logits = self.decoder(
                memory,
                tokens[:, -1:],
                step=step,
                inference_params=self_attn_caches,
                cross_attn_caches=cross_attn_caches,
            )

            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            if eos_token_id is not None:
                is_finished = tokens[:, -1] == eos_token_id
                if is_finished.any():
                    log_probs[is_finished, :] = -float("inf")
                    log_probs[is_finished, eos_token_id] = 0.0

            next_scores = scores.unsqueeze(1) + log_probs
            next_scores = next_scores.view(B, beam_size * vocab_size)

            topk_scores, topk_indices = next_scores.topk(beam_size, dim=1)

            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            batch_beam_indices = (batch_indices * beam_size + beam_indices).view(-1)

            tokens = torch.cat(
                [tokens[batch_beam_indices], token_indices.view(-1, 1)], dim=1
            )
            scores = topk_scores.view(-1)

            reorder_cache(self_attn_caches, batch_beam_indices)
            reorder_cache(cross_attn_caches, batch_beam_indices)

            if eos_token_id is not None and (tokens[:, -1] == eos_token_id).all():
                break

        generated_tokens = tokens
        best_sequences = generated_tokens.view(B, beam_size, -1)
        return best_sequences
