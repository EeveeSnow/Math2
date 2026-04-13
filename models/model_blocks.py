import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from mamba_ssm import Mamba2
from typing import Tuple


@torch.no_grad()
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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        feats = self.backbone(x)
        f3, f4 = feats[-2], feats[-1]

        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)

        p4 = self.proj4(f4)
        p3 = self.proj3(f3) + F.interpolate(
            p4, size=f3.shape[-2:], mode="bilinear", align_corners=False
        )

        B, C, H, W = p3.shape
        memory = p3.flatten(2).transpose(1, 2)
        return self.norm(memory), (H, W)


@torch.no_grad()
class Pos2D(nn.Module):
    def __init__(self, d_model: int = 512, max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(max_h, d_model // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(max_w, d_model // 2) * 0.02)

    def forward(self, H: int, W: int) -> torch.Tensor:
        y = self.row_embed[:H].unsqueeze(1).expand(-1, W, -1)
        x = self.col_embed[:W].unsqueeze(0).expand(H, -1, -1)
        pos = torch.cat([y, x], dim=-1)
        return pos.reshape(1, H * W, -1)


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, layer_idx: int = None):
        super().__init__()
        self.nhead = 8
        self.head_dim = d_model // self.nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model),
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


class MambaDecoderBlock(DecoderBlock):
    def __init__(self, d_model=512, layer_idx=None):
        super().__init__()

        self.ssm1 = Mamba2(
            d_model=d_model, d_state=64, d_conv=4, expand=2, layer_idx=layer_idx
        )

    def forward(
        self,
        x,
        memory,
        inference_params=None,
        cross_attn_cache=None,
        self_attn_cache=None,
    ):
        x = x + self.ssm1(self.ln1(x), inference_params=inference_params)
        x = x + self.cross_attention(self.ln2(x), memory, cross_attn_cache)
        x = x + self.ffn(self.ln3(x))
        return x


class GatedConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=15):
        super().__init__()

        self.conv = nn.Conv1d(
            d_model, d_model * 2, kernel_size, padding=kernel_size - 1, groups=d_model
        )

        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.conv(x.transpose(1, 2))[:, :, : -self.conv.padding[0]]
        y = y.transpose(1, 2)

        u, v = y.chunk(2, dim=-1)
        x = x + self.proj(u * torch.sigmoid(v))
        return self.norm(x)


class ConvDecoderBlock(DecoderBlock):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.conv = GatedConvBlock(d_model)

    def forward(self, x, memory, cross_attn_cache=None):
        x = self.ln1(self.conv(x))
        x = x + self.cross_attention(self.ln2(x), memory, cross_attn_cache)
        x = self.ln3(x + self.ffn(x))
        return x


class TransformerDecoderBlock(DecoderBlock):
    def __init__(self, d_model=512, nhead=8, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.self_out_proj = nn.Linear(d_model, d_model)

    def causal_self_attention(self, x, self_attn_cache=None):
        B, T, C = x.shape

        qkv = self.self_qkv_proj(x).view(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = T > 1

        if self_attn_cache is not None:
            if "k" in self_attn_cache:
                k = torch.cat([self_attn_cache["k"], k], dim=2)
                v = torch.cat([self_attn_cache["v"], v], dim=2)

            self_attn_cache["k"] = k
            self_attn_cache["v"] = v

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn = attn.transpose(1, 2).contiguous().reshape(B, T, C)

        return self.self_out_proj(attn)

    def forward(
        self,
        x,
        memory,
        inference_params=None,
        cross_attn_cache=None,
        self_attn_cache=None,
    ):
        x = x + self.causal_self_attention(self.ln1(x), self_attn_cache)
        x = x + self.cross_attention(self.ln2(x), memory, cross_attn_cache)
        x = x + self.ffn(self.ln3(x))
        return x


class MoEMambaTransformerBlock(nn.Module):
    def __init__(self, d_model=512, layer_idx=None):
        super().__init__()
        self.ssm1 = Mamba2(
            d_model=d_model, d_state=64, d_conv=4, expand=2, layer_idx=layer_idx
        )

        self.nhead = 8
        self.head_dim = d_model // self.nhead
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=self.nhead, batch_first=True
        )

        self.router = nn.Linear(d_model, 2)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

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

    def forward(
        self, x, memory, inference_params=None, cross_attn_cache=None, attn_mask=None
    ):
        norm_x = self.ln1(x)
        out_mamba = self.ssm1(norm_x, inference_params=inference_params)
        if attn_mask is None:
            seq_len = x.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=x.device
            )

        out_attn, _ = self.self_attn(norm_x, norm_x, norm_x, is_causal=True)
        router_logits = self.router(norm_x)
        routing_weights = F.softmax(router_logits, dim=-1)

        mixed_sequence_out = (routing_weights[..., 0:1] * out_mamba) + (
            routing_weights[..., 1:2] * out_attn
        )

        x = x + mixed_sequence_out
        x = x + self.cross_attention(self.ln2(x), memory, cross_attn_cache)
        x = x + self.ffn(self.ln3(x))

        return x


class SparseMoEFFN(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1, aux_loss_coef=0.02):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(4 * d_model, d_model),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.size(0)

        logits = self.router(x_flat)
        gate_probs = F.softmax(logits, dim=-1)

        weights, expert_ids = torch.topk(gate_probs, self.top_k, dim=-1)

        weights = weights / weights.sum(dim=-1, keepdim=True)

        expert_ids_flat = expert_ids.reshape(-1)
        weights_flat = weights.reshape(-1)
        token_indices = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        sorted_experts, perm = torch.sort(expert_ids_flat)
        token_indices = token_indices[perm]
        weights_flat = weights_flat[perm]

        x_expanded = x_flat[token_indices]

        outputs = torch.zeros_like(x_expanded)
        start = 0
        for i in range(self.num_experts):
            mask = sorted_experts == i
            count = mask.sum().item()

            if count > 0:
                end = start + count

                chunk = x_expanded[start:end]
                out = self.experts[i](chunk)

                outputs[start:end] = out * weights_flat[start:end].unsqueeze(-1)

                start = end
        final_output = torch.zeros_like(x_flat)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)

        outputs = outputs[inv_perm]
        outputs = outputs.view(N, self.top_k, D)
        final_output = outputs.sum(dim=1)
        density = F.one_hot(expert_ids[:, 0], self.num_experts).float().mean(0)
        avg_probs = gate_probs.mean(0)
        aux_loss = self.num_experts * torch.sum(density * avg_probs)

        return final_output.view(B, T, D), aux_loss


class MambaMOEDecoderBlock(DecoderBlock):
    def __init__(self, d_model=512, layer_idx=None):
        super().__init__()
        self.ssm1 = Mamba2(
            d_model=d_model, d_state=64, d_conv=4, expand=2, layer_idx=layer_idx
        )
        self.ffn = SparseMoEFFN(d_model=d_model, num_experts=4)

    def forward(
        self,
        x,
        memory,
        inference_params=None,
        cross_attn_cache=None,
        self_attn_cache=None,
    ):
        x = x + self.ssm1(self.ln1(x), inference_params=inference_params)
        x = x + self.cross_attention(self.ln2(x), memory, cross_attn_cache)
        expert_out, aux_loss = self.ffn(self.ln3(x))
        x = x + expert_out
        return x, aux_loss
