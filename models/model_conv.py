import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_blocks import SwinEncoder, Pos2D, ConvDecoderBlock


class HybridDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, depth=6, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        self.blocks = nn.ModuleList([ConvDecoderBlock(d_model) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, memory, tgt, step=None, cross_attn_caches=None):
        x = self.embed(tgt)

        if step is None:
            seq_len = tgt.size(1)
            x = x + self.pos[:, :seq_len]
        else:
            x = x + self.pos[:, step : step + 1]
        if cross_attn_caches is None:
            cross_attn_caches = [None] * len(self.blocks)

        for blk, cache in zip(self.blocks, cross_attn_caches):
            x = blk(x, memory, cross_attn_cache=cache)

        x = self.norm(x)
        return self.fc(x)


class SwinGConvTex(nn.Module):
    def __init__(self, vocab_size, d_model=512, decoder_depth=9, max_len=512):
        super().__init__()
        self.encoder = SwinEncoder(d_model)
        self.pos_encoder = Pos2D(d_model=d_model)
        self.decoder = HybridDecoder(
            vocab_size, d_model, depth=decoder_depth, max_len=max_len
        )
        self.d_model = d_model

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

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        B = images.size(0)
        tokens = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        cross_attn_caches = [{} for _ in self.decoder.blocks]
        outputs = []

        is_finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            tgt = tokens

            logits = self.decoder(
                memory, tgt, step=None, cross_attn_caches=cross_attn_caches
            )

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            next_token[is_finished] = eos_token_id

            tokens = torch.cat([tokens, next_token], dim=1)
            outputs.append(next_token)

            is_finished |= next_token.squeeze(-1) == eos_token_id

            if is_finished.all():
                break

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def generate_beam_search(
        self, images, start_token_id, eos_token_id, beam_size=3, max_new_tokens=256
    ):
        self.eval()
        B = images.size(0)
        device = images.device

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        memory = memory.repeat_interleave(beam_size, dim=0)

        cross_attn_caches = [{} for _ in range(len(self.decoder.blocks))]

        beam_scores = torch.full((B, beam_size), -1e9, dtype=torch.float, device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)

        generated_tokens = torch.full(
            (B * beam_size, 1), start_token_id, dtype=torch.long, device=device
        )

        is_finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            logits = self.decoder(
                memory=memory,
                tgt=generated_tokens,
                step=None,
                cross_attn_caches=cross_attn_caches,
            )

            next_token_logits = logits[:, -1, :]
            next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)

            if eos_token_id is not None:
                next_token_logprobs[is_finished, :] = -1e9
                next_token_logprobs[is_finished, eos_token_id] = 0.0

            vocab_size = next_token_logprobs.shape[-1]

            next_token_logprobs = next_token_logprobs.view(B, beam_size, vocab_size)
            cumulative_scores = beam_scores.view(B, beam_size, 1) + next_token_logprobs
            cumulative_scores = cumulative_scores.view(B, beam_size * vocab_size)

            top_scores, top_indices = torch.topk(cumulative_scores, beam_size, dim=1)

            beam_scores = top_scores.view(-1)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            batch_offset = torch.arange(B, device=device).unsqueeze(1) * beam_size
            global_beam_indices = (batch_offset + beam_indices).view(-1)

            generated_tokens = generated_tokens[global_beam_indices]
            current_tokens = token_indices.view(-1, 1)
            generated_tokens = torch.cat([generated_tokens, current_tokens], dim=1)

            is_finished = is_finished[global_beam_indices]
            if eos_token_id is not None:
                is_finished = is_finished | (current_tokens.squeeze(-1) == eos_token_id)
                if is_finished.all():
                    break

            for cache in cross_attn_caches:
                if "k" in cache:
                    cache["k"] = cache["k"][global_beam_indices]
                    cache["v"] = cache["v"][global_beam_indices]

        best_sequences = generated_tokens.view(B, beam_size, -1)

        return best_sequences
