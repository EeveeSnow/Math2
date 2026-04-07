import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple


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
        f3, f4  = feats[-2], feats[-1]

        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)

        p4 = self.proj4(f4)
        p3 = self.proj3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)

        B, C, H, W = p3.shape
        memory     = p3.flatten(2).transpose(1, 2)
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
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, layer_idx=None):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.self_qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.self_out_proj = nn.Linear(d_model, d_model)

        self.cross_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_kv_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.cross_out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def causal_self_attention(self, x, self_attn_cache=None):
        B, T, C = x.shape

        qkv = self.self_qkv_proj(x).view(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = (T > 1) 

        if self_attn_cache is not None:
            if "k" in self_attn_cache:
                k = torch.cat([self_attn_cache["k"], k], dim=2)
                v = torch.cat([self_attn_cache["v"], v], dim=2)

            self_attn_cache["k"] = k
            self_attn_cache["v"] = v

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn = attn.transpose(1, 2).contiguous().reshape(B, T, C)

        return self.self_out_proj(attn)

    def cross_attention(self, x, memory, cross_attn_cache=None):
        B, T, C = x.shape

        q = self.cross_q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        if cross_attn_cache is not None and "k" in cross_attn_cache:
            k = cross_attn_cache["k"]
            v = cross_attn_cache["v"]
        else:
            S = memory.shape[1]
            kv = self.cross_kv_proj(memory).view(B, S, 2, self.nhead, self.head_dim)
            k, v = kv.unbind(dim=2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if cross_attn_cache is not None:
                cross_attn_cache["k"] = k
                cross_attn_cache["v"] = v

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().reshape(B, T, C)

        return self.cross_out_proj(attn)

    def forward(self, x, memory, cross_attn_cache=None, self_attn_cache=None):
        x = x + self.causal_self_attention(self.ln1(x), self_attn_cache)
        x = x + self.cross_attention(self.ln2(x), memory, cross_attn_cache)
        x = x + self.ffn(self.ln3(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, depth=8, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        
        self.blocks = nn.ModuleList([TransformerDecoderBlock(d_model, layer_idx=i) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, memory, tgt, step=None, inference_params=None, cross_attn_caches=None):
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

        for blk, c_cache, s_cache in zip(self.blocks, cross_attn_caches, self_attn_caches):
            x = blk(
                x, 
                memory, 
                cross_attn_cache=c_cache,
                self_attn_cache=s_cache
            )
            
        x = self.norm(x)
        return self.fc(x)

class SwinTransformerTex(nn.Module):
    def __init__(self, vocab_size, d_model=512, decoder_depth=7, max_len=512):
        super().__init__()
        self.encoder = SwinEncoder(d_model)
        self.pos_encoder = Pos2D(d_model=d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model, depth=decoder_depth, max_len=max_len)

    def forward(self, images, tgt):
        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)
        return self.decoder(memory, tgt), (H, W)


    @torch.no_grad()
    @torch.compiler.disable
    def generate(
        self,
        images,
        start_token_id,
        eos_token_id,
        max_new_tokens=256,
        device="cuda"
    ):
        self.eval()

        if isinstance(start_token_id, int):
            start_token_id = torch.full(
                (images.size(0), 1), 
                start_token_id, 
                dtype=torch.long, 
                device=images.device
            )

        elif isinstance(start_token_id, torch.Tensor) and start_token_id.dim() == 1:
            start_token_id = start_token_id.unsqueeze(1)

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        cross_attn_caches = [{} for _ in self.decoder.blocks]
        self_attn_caches  = [{} for _ in self.decoder.blocks]

        logits = self.decoder(
            memory,
            start_token_id,
            step=None,
            inference_params=self_attn_caches,
            cross_attn_caches=cross_attn_caches
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
                cross_attn_caches=cross_attn_caches
            )

            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return tokens
    
    @torch.no_grad()
    @torch.compiler.disable
    def generate_beam_search(self, images, start_token_id, beam_size=3, max_new_tokens=100, eos_token_id=None):
        self.eval()
        
        B = images.size(0)
        device = images.device

        if isinstance(start_token_id, int):
            start_token_id = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        elif isinstance(start_token_id, torch.Tensor) and start_token_id.dim() == 1:
            start_token_id = start_token_id.unsqueeze(1)

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        memory = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(B * beam_size, memory.size(1), memory.size(2))
        
        tokens = start_token_id.unsqueeze(1).expand(-1, beam_size, -1).reshape(B * beam_size, -1)
        scores = torch.zeros((B, beam_size), dtype=torch.float, device=device)
        scores[:, 1:] = -1e9
        scores = scores.view(-1)

        cross_attn_caches = [{} for _ in self.decoder.blocks]
        self_attn_caches  = [{} for _ in self.decoder.blocks]

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
            cross_attn_caches=cross_attn_caches
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

        tokens = torch.cat([tokens[batch_beam_indices], token_indices.view(-1, 1)], dim=1)
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
                cross_attn_caches=cross_attn_caches
            )

            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            if eos_token_id is not None:
                is_finished = (tokens[:, -1] == eos_token_id)
                if is_finished.any():
                    log_probs[is_finished, :] = -float('inf')
                    log_probs[is_finished, eos_token_id] = 0.0

            next_scores = scores.unsqueeze(1) + log_probs
            next_scores = next_scores.view(B, beam_size * vocab_size)

            topk_scores, topk_indices = next_scores.topk(beam_size, dim=1)

            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            batch_beam_indices = (batch_indices * beam_size + beam_indices).view(-1)

            tokens = torch.cat([tokens[batch_beam_indices], token_indices.view(-1, 1)], dim=1)
            scores = topk_scores.view(-1)

            reorder_cache(self_attn_caches, batch_beam_indices)
            reorder_cache(cross_attn_caches, batch_beam_indices)

            if eos_token_id is not None and (tokens[:, -1] == eos_token_id).all():
                break

        generated_tokens = tokens
        best_sequences = generated_tokens.view(B, beam_size, -1)
        return best_sequences