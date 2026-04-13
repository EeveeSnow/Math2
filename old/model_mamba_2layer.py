"""
SwinMambaTex — Image-to-LaTeX

Fixed bugs vs original:
  [F1] generate() uses _encode_memory() → identical pooling+pos as forward()
  [F2] RoPE applied to BOTH Q and K in cross-attention
  [F3] Every Mamba2 block has a unique layer_idx (critical for InferenceParams)
  [F4] RefinementDecoder is truly bidirectional (forward + backward Mamba2)
  [F5] forward() returns coverage tensor for optional loss usage
  [F6] generate() uses InferenceParams → O(n) recurrent inference, not O(n²)
  [F7] F.adaptive_avg_pool2d replaces fixed kernel_size=2 (safe for odd sizes)
  [F8] Dropout added throughout for regularisation
  [F9] _encode_memory() shared helper — one source of truth for memory prep

New:
  [N1] generate(temperature=) — greedy (1.0) or multinomial sampling (other)
  [N2] generate_beam_search() — full-sequence forward with proper beam reorder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import timm
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams


# ─────────────────────────────────────────────────────────────────────────────
#  RoPE
# ─────────────────────────────────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Pre-computed RoPE buffers.
    forward(x, offset) applies positions [offset, offset+T) to x.
    offset > 0 is used during single-step recurrent inference.
    """

    def __init__(self, dim: int, max_pos: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10_000 ** (torch.arange(0, dim, 2).float() / dim))
        t        = torch.arange(max_pos).float()
        freqs    = torch.einsum("i,j->ij", t, inv_freq)
        emb      = torch.cat([freqs, freqs], dim=-1)             # [max_pos, dim]
        self.register_buffer("cos", emb.cos()[None, None])       # [1, 1, max_pos, dim]
        self.register_buffer("sin", emb.sin()[None, None])

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x : [B, heads, T, head_dim]"""
        T   = x.shape[2]
        cos = self.cos[:, :, offset : offset + T]
        sin = self.sin[:, :, offset : offset + T]
        return (x * cos) + (rotate_half(x) * sin)


# ─────────────────────────────────────────────────────────────────────────────
#  Swin Encoder + FPN
# ─────────────────────────────────────────────────────────────────────────────

class SwinEncoder(nn.Module):

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384",
            pretrained=True,
            features_only=True,
            in_chans=1,
            img_size=(384, 384),
            dynamic_img_size=True,
        )
        ch = self.backbone.feature_info.channels()

        self.proj2 = nn.Conv2d(ch[-3], d_model, 1)
        self.proj3 = nn.Conv2d(ch[-2], d_model, 1)
        self.proj4 = nn.Conv2d(ch[-1], d_model, 1)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        feats       = self.backbone(x)
        f2, f3, f4  = feats[-3], feats[-2], feats[-1]

        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)

        p4 = self.proj4(f4)
        p3 = self.proj3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.proj2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)

        B, C, H, W = p2.shape
        memory     = p2.flatten(2).transpose(1, 2)
        return self.norm(memory), (H, W)


class Pos2D(nn.Module):

    def __init__(self, d_model: int = 512, max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(max_h, d_model // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(max_w, d_model // 2) * 0.02)

    def forward(self, H: int, W: int) -> torch.Tensor:
        y   = self.row_embed[:H].unsqueeze(1).expand(-1, W, -1)
        x   = self.col_embed[:W].unsqueeze(0).expand(H, -1, -1)
        pos = torch.cat([y, x], dim=-1)
        return pos.reshape(1, H * W, -1)


class DraftDecoderBlock(nn.Module):

    def __init__(
        self,
        d_model:   int   = 512,
        nhead:     int   = 8,
        layer_idx: int   = 0,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead    = nhead
        self.head_dim = d_model // nhead

        self.ssm = Mamba2(
            d_model=d_model, d_state=64, d_conv=4, expand=2, layer_idx=layer_idx
        )

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj  = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rotary = RotaryEmbedding(self.head_dim)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.ln3  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _cross_attn(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        coverage: torch.Tensor,
        q_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        S       = memory.shape[1]

        xf = x
        mf = memory

        q  = self.q_proj(xf).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(mf)
        k, v = kv.chunk(2, dim=-1)
        k  = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v  = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        q = self.rotary(q, offset=q_offset)
        k = self.rotary(k, offset=0)

        scale  = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale 
        attn_w = torch.softmax(scores, dim=-1)                  
        attn   = torch.matmul(attn_w, v)                        

        coverage = coverage + attn_w.mean(1).mean(1)

        out = attn.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out), coverage


    def forward(
        self,
        x:                torch.Tensor,
        memory:           torch.Tensor,
        coverage:         torch.Tensor,
        inference_params  = None,
        q_offset:         int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = x + self.drop(self.ssm(self.ln1(x), inference_params=inference_params))

        attn, coverage = self._cross_attn(self.ln2(x), memory, coverage, q_offset)
        x = x + self.drop(attn)

        x = x + self.drop(self.ffn(self.ln3(x)))

        return x, coverage


class DraftDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 512,
        depth:      int   = 8,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.embed      = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=d_model ** -0.5)
        self.embed_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DraftDecoderBlock(d_model, nhead=8, layer_idx=i, dropout=dropout)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc   = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embed.weight          # Weight tying

    def forward(
        self,
        memory:           torch.Tensor,
        tgt:              torch.Tensor,
        inference_params  = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_offset = inference_params.seqlen_offset if inference_params is not None else 0

        x   = self.embed_drop(self.embed(tgt) * (self.d_model ** 0.5))
        B   = x.shape[0]
        cov = torch.zeros(B, memory.shape[1], device=x.device, dtype=x.dtype)

        for blk in self.blocks:
            x, cov = blk(x, memory, cov, inference_params, q_offset)

        return self.fc(self.norm(x)), cov
    

class RefinementDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 512,
        depth:      int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.embed      = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=d_model ** -0.5)
        self.embed_drop = nn.Dropout(dropout)

        self.fwd_blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, layer_idx=i)
            for i in range(depth)
        ])
        self.bwd_blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, layer_idx=depth + i)
            for i in range(depth)
        ])

        self.fwd_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.bwd_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])

        self.merge  = nn.Linear(d_model * 2, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)
        self.norm   = nn.LayerNorm(d_model)
        self.fc     = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x   = self.embed_drop(self.embed(tokens) * (self.d_model ** 0.5))
        fwd = x
        bwd = x.flip(1)

        for f_blk, b_blk, f_ln, b_ln in zip(
            self.fwd_blocks, self.bwd_blocks, self.fwd_norms, self.bwd_norms
        ):
            fwd = fwd + self.drop(f_blk(f_ln(fwd)))
            bwd = bwd + self.drop(b_blk(b_ln(bwd)))

        bwd    = bwd.flip(1)
        merged = self.drop(self.merge(torch.cat([fwd, bwd], dim=-1)))
        return self.fc(self.norm(merged))


class SwinMambaTex(nn.Module):

    def __init__(
        self,
        vocab_size:   int,
        d_model:      int   = 512,
        draft_depth:  int   = 8,
        refine_depth: int   = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.encoder       = SwinEncoder(d_model)
        self.pos           = Pos2D(d_model)
        self.draft_decoder = DraftDecoder(vocab_size, d_model, draft_depth, dropout)
        self.refiner       = RefinementDecoder(vocab_size, d_model, refine_depth, dropout)


    def _encode_memory(self, images: torch.Tensor) -> torch.Tensor:
        memory, (H, W) = self.encoder(images)
        B, _, C        = memory.shape
        H2, W2         = max(H // 2, 1), max(W // 2, 1)

        memory = memory.view(B, H, W, C).permute(0, 3, 1, 2)
        memory = F.adaptive_avg_pool2d(memory, (H2, W2))
        memory = memory.permute(0, 2, 3, 1).reshape(B, -1, C)
        memory = memory + self.pos(H2, W2).to(memory.device)
        return memory

    def forward(
        self,
        images: torch.Tensor,
        tgt:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory                 = self._encode_memory(images)
        draft_logits, coverage = self.draft_decoder(memory, tgt)
        draft_tokens           = draft_logits.argmax(-1)
        refined_logits         = self.refiner(draft_tokens)
        return draft_logits, refined_logits, coverage


    @torch.no_grad()
    def generate(
        self,
        images:         torch.Tensor,
        start_token_id: int,
        max_new_tokens: int            = 100,
        eos_token_id:   Optional[int]  = None,
        temperature:    float          = 1.0,
    ) -> torch.Tensor:
        self.eval()
        device = images.device
        B      = images.size(0)

        memory = self._encode_memory(images)

        inference_params = InferenceParams(
            max_seqlen     = max_new_tokens + 1,
            max_batch_size = B,
        )

        cur_tok   = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        generated = [cur_tok]

        for step in range(max_new_tokens):
            inference_params.seqlen_offset = step

            logits, _  = self.draft_decoder(memory, cur_tok, inference_params)
            next_logit  = logits[:, -1]

            if temperature <= 0.0 or temperature == 1.0:
                next_tok = next_logit.argmax(dim=-1, keepdim=True)
            else:
                probs    = F.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            generated.append(next_tok)
            cur_tok = next_tok

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        all_tokens     = torch.cat(generated, dim=1)
        refined_logits = self.refiner(all_tokens)
        return refined_logits.argmax(-1) 


    @torch.no_grad()
    def generate_beam_search(
        self,
        images: torch.Tensor,
        start_token_id: int,
        beam_size: int = 5,
        max_new_tokens: int = 256,
        eos_token_id: Optional[int] = None,
    ):
        self.eval()
        device = images.device
        B = images.size(0)

        memory = self._encode_memory(images)
        _, S, C = memory.shape

        memory = (
            memory.unsqueeze(1)
            .expand(B, beam_size, S, C)
            .reshape(B * beam_size, S, C)
            .contiguous()
        )

        inference_params = InferenceParams(
            max_seqlen=max_new_tokens + 1,
            max_batch_size=B * beam_size,
        )

        beam_tokens = torch.full(
            (B * beam_size, 1), start_token_id, dtype=torch.long, device=device
        )

        beam_lp = torch.full((B, beam_size), float("-inf"), device=device)
        beam_lp[:, 0] = 0

        beam_done = torch.zeros(B, beam_size, dtype=torch.bool, device=device)

        cur_tok = beam_tokens[:, -1:]

        for step in range(max_new_tokens):

            if beam_done.all():
                break

            inference_params.seqlen_offset = step

            logits, _ = self.draft_decoder(memory, cur_tok, inference_params)
            logits = logits[:, -1]

            V = logits.size(-1)

            next_lp = F.log_softmax(logits, dim=-1)
            next_lp = next_lp.view(B, beam_size, V)

            scores = beam_lp.unsqueeze(-1) + next_lp

            if eos_token_id is not None:
                done_mask = beam_done.unsqueeze(-1)
                sink = torch.full_like(scores, float("-inf"))
                sink[:, :, eos_token_id] = beam_lp
                scores = torch.where(done_mask, sink, scores)

            top_scores, top_idx = scores.view(B, -1).topk(beam_size, dim=-1)

            prev_beam = top_idx // V
            next_tok = top_idx % V

            new_tokens = []
            new_done = torch.zeros_like(beam_done)

            reorder_indices = []

            for b in range(B):
                for k in range(beam_size):

                    pb = prev_beam[b, k].item()

                    reorder_indices.append(b * beam_size + pb)

                    old_seq = beam_tokens[b * beam_size + pb]
                    new_seq = torch.cat([old_seq, next_tok[b, k:k+1]])

                    new_tokens.append(new_seq)

                    is_eos = (
                        eos_token_id is not None
                        and next_tok[b, k].item() == eos_token_id
                    )

                    new_done[b, k] = beam_done[b, pb] or is_eos

            reorder_indices = torch.tensor(reorder_indices, device=device)
            beam_tokens = torch.stack(new_tokens)

            for layer in inference_params.key_value_memory_dict:
                cache = inference_params.key_value_memory_dict[layer]
                if isinstance(cache, tuple):
                    reordered = tuple(
                        t.index_select(0, reorder_indices) if torch.is_tensor(t) else t
                        for t in cache
                    )
                else:
                    reordered = cache.index_select(0, reorder_indices)
                inference_params.key_value_memory_dict[layer] = reordered

            beam_done = new_done
            beam_lp = top_scores

            cur_tok = next_tok.reshape(B * beam_size, 1)

        topk = torch.topk(beam_lp, k=3, dim=-1).indices
        results = []

        for b in range(B):
            seqs = []
            for i in range(3):
                idx = b * beam_size + topk[b, i].item()
                seqs.append(beam_tokens[idx])
            results.append(torch.stack(seqs))

        results = torch.stack(results)

        refined = self.refiner(results.view(B * 3, -1))
        refined = refined.argmax(-1)

        return refined.view(B, 3, -1)