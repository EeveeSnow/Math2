import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from mamba_ssm import Mamba2
# Import InferenceParams for Mamba's cache
from mamba_ssm.utils.generation import InferenceParams 
from typing import Optional, Tuple
from mamba_ssm import Mamba2


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
    

class MambaDecoderBlock(nn.Module):
    def __init__(self, d_model=512, layer_idx=None):
        super().__init__()

        self.ssm1 = Mamba2(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            layer_idx=layer_idx
        )

        self.nhead = 8
        self.head_dim = d_model // self.nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def cross_attention(self, x, memory, cross_attn_cache=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        if cross_attn_cache is not None and "k" in cross_attn_cache:
            k = cross_attn_cache["k"]
            v = cross_attn_cache["v"]
        else:
            S = memory.shape[1]
            kv = self.kv_proj(memory).view(B, S, 2, self.nhead, self.head_dim)
            k, v = kv.unbind(dim=2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if cross_attn_cache is not None:
                cross_attn_cache["k"] = k
                cross_attn_cache["v"] = v

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(attn)

    def forward(self, x, memory, inference_params=None, cross_attn_cache=None):
        x = x + self.ssm1(self.ln1(x), inference_params=inference_params)
        
        x = x + self.cross_attention(self.ln2(x), memory, cross_attn_cache)
        
        x = x + self.ffn(self.ln3(x))
        return x

class MambaDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, depth=8, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        
        self.blocks = nn.ModuleList(
            [MambaDecoderBlock(d_model, layer_idx=i) for i in range(depth)]
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
            # x = x + self.pos[:, step] cuda graph optimizations
        x = x.contiguous() 
        if cross_attn_caches is None:
            cross_attn_caches = [None] * len(self.blocks)

        for blk, cache in zip(self.blocks, cross_attn_caches):
            x = blk(
                x, 
                memory, 
                inference_params=inference_params, 
                cross_attn_cache=cache
            )
            
        x = self.norm(x)
        return self.fc(x)

class SwinMambaTex(nn.Module):
    def __init__(self, vocab_size, d_model=512, decoder_depth=6, max_len=512):
        super().__init__()
        self.encoder = SwinEncoder(d_model)
        self.pos_encoder = Pos2D(d_model=d_model)
        self.decoder = MambaDecoder(vocab_size, d_model, depth=decoder_depth, max_len=max_len)

    def forward(self, images, tgt):
        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)
        return self.decoder(memory, tgt), (H, W)

    @torch.no_grad()
    def generate(self, images, start_token_id, max_new_tokens=100, eos_token_id=None, temperature=1.0):
        self.eval()
        B = images.size(0)
        device = images.device

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        inference_params = InferenceParams(max_seqlen=max_new_tokens, max_batch_size=B)
        
        dummy_input = torch.empty(1, memory.size(-1), device=device, dtype=memory.dtype)
        cache_dtype = self.decoder.blocks[0].q_proj(dummy_input).dtype
        
        for blk in self.decoder.blocks:
            layer_idx = blk.ssm1.layer_idx
            inference_params.key_value_memory_dict[layer_idx] = blk.ssm1.allocate_inference_cache(
                batch_size=B,
                max_seqlen=max_new_tokens,
                dtype=cache_dtype
            )

        cross_attn_caches = [{} for _ in range(len(self.decoder.blocks))]

        current_tokens = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        generated_tokens = []

        for step in range(max_new_tokens):
            logits = self.decoder(
                memory=memory, 
                tgt=current_tokens, 
                step=step,
                inference_params=inference_params,
                cross_attn_caches=cross_attn_caches
            )

            next_token_logits = logits[:, -1, :] / temperature
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            current_tokens = next_token

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            inference_params.seqlen_offset += 1

        return torch.cat(generated_tokens, dim=1)
    

    @torch.no_grad()
    def generate_beam_search(self, images, start_token_id, beam_size=3, max_new_tokens=100, eos_token_id=None):
        self.eval()
        B = images.size(0)
        device = images.device

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)
        memory = memory.repeat_interleave(beam_size, dim=0)

        inference_params = InferenceParams(max_seqlen=max_new_tokens, max_batch_size=B * beam_size)
        
        dummy_input = torch.empty(1, memory.size(-1), device=device, dtype=memory.dtype)
        cache_dtype = self.decoder.blocks[0].q_proj(dummy_input).dtype
        
        for blk in self.decoder.blocks:
            layer_idx = blk.ssm1.layer_idx
            inference_params.key_value_memory_dict[layer_idx] = blk.ssm1.allocate_inference_cache(
                batch_size=B * beam_size,
                max_seqlen=max_new_tokens,
                dtype=cache_dtype
            )

        cross_attn_caches = [{} for _ in range(len(self.decoder.blocks))]

        beam_scores = torch.full((B, beam_size), -1e9, dtype=torch.float, device=device)
        beam_scores[:, 0] = 0.0 
        
        generated_tokens = torch.full((B * beam_size, 1), start_token_id, dtype=torch.long, device=device)
        current_tokens = generated_tokens.clone()
        is_finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            logits = self.decoder(
                memory=memory, 
                tgt=current_tokens, 
                step=step, 
                inference_params=inference_params,
                cross_attn_caches=cross_attn_caches
            )

            next_token_logits = logits[:, -1, :]
            next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)

            if eos_token_id is not None:
                next_token_logprobs[is_finished, :] = -1e9
                next_token_logprobs[is_finished, eos_token_id] = 0.0

            vocab_size = next_token_logprobs.shape[-1]
            next_token_logprobs = next_token_logprobs.view(B, beam_size, vocab_size)
            cumulative_scores = beam_scores.unsqueeze(-1) + next_token_logprobs
            cumulative_scores = cumulative_scores.view(B, beam_size * vocab_size)

            top_scores, top_indices = torch.topk(cumulative_scores, beam_size, dim=1)
            beam_scores = top_scores

            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            batch_offset = torch.arange(B, device=device).unsqueeze(1) * beam_size
            global_beam_indices = (batch_offset + beam_indices).view(-1)

            for layer_idx, state in inference_params.key_value_memory_dict.items():
                if isinstance(state, tuple):
                    inference_params.key_value_memory_dict[layer_idx] = tuple(
                        s[global_beam_indices] for s in state
                    )
                else:
                    inference_params.key_value_memory_dict[layer_idx] = state[global_beam_indices]

            generated_tokens = generated_tokens[global_beam_indices]
            current_tokens = token_indices.view(-1, 1)
            generated_tokens = torch.cat([generated_tokens, current_tokens], dim=1)

            is_finished = is_finished[global_beam_indices]
            if eos_token_id is not None:
                is_finished = is_finished | (current_tokens.squeeze(-1) == eos_token_id)
                if is_finished.all():
                    break
        
            inference_params.seqlen_offset += 1
            
        best_sequences = generated_tokens.view(B, beam_size, -1)
        return best_sequences

    @torch.no_grad()
    def generate_cuda_graph(self, images, start_token_id, max_new_tokens=100, eos_token_id=None, temperature=1.0):
        self.eval()
        B = images.size(0)
        device = images.device

        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)

        inference_params = InferenceParams(max_seqlen=max_new_tokens, max_batch_size=B)
        cross_attn_caches =[{} for _ in range(len(self.decoder.blocks))]

        static_current_tokens = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        static_step = torch.zeros((1,), dtype=torch.long, device=device)
        static_next_token = torch.zeros_like(static_current_tokens)

        def capture_step():
            logits = self.decoder(
                memory=memory, 
                tgt=static_current_tokens, 
                step=static_step,
                inference_params=inference_params,
                cross_attn_caches=cross_attn_caches
            )
            next_token_logits = logits[:, -1, :] / temperature
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            static_next_token.copy_(next_token)

        capture_step()
        generated_tokens =[static_next_token.clone()]
        static_current_tokens.copy_(static_next_token)
        static_step.add_(1)
        inference_params.seqlen_offset += 1

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                capture_step()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            capture_step()

        for _ in range(1, max_new_tokens):
            g.replay()

            next_token_clone = static_next_token.clone()
            generated_tokens.append(next_token_clone)
            
            if eos_token_id is not None and (next_token_clone == eos_token_id).all():
                break

            static_current_tokens.copy_(static_next_token)
            static_step.add_(1)

            inference_params.seqlen_offset += 1 

        return torch.cat(generated_tokens, dim=1)