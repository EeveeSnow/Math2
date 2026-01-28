import torch
import torch.nn as nn
import timm
from mamba_ssm import Mamba

class SwinEncoder(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            features_only=True
        )
        self.proj = nn.Linear(1024, d_model)

    def forward(self, x):
        feats = self.backbone(x)[-1]        # (B, C, H, W)
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return self.proj(feats), (H, W)


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            expand=2
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        attn, _ = self.cross_attn(x, memory, memory)
        x = self.ln1(x + attn)
        x = self.ln2(x + self.mamba(x))
        x = self.ln3(x + self.ffn(x))
        return x


class HybridDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, depth=6, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model) for _ in range(depth)]
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, memory, tgt):
        x = self.embed(tgt) + self.pos[:, :tgt.size(1)]
        for blk in self.blocks:
            x = blk(x, memory)
        return self.fc(x)

class Pos2D(nn.Module):
    def __init__(self, max_h=50, max_w=50, d_model=512):
        super().__init__()
        self.x_embed = nn.Embedding(max_w, d_model // 2)
        self.y_embed = nn.Embedding(max_h, d_model // 2)

    def forward(self, H, W, device):
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)

        pos_y = self.y_embed(y)
        pos_x = self.x_embed(x)

        pos = torch.cat([
            pos_y[:, None, :].expand(H, W, -1),
            pos_x[None, :, :].expand(H, W, -1)
        ], dim=-1)

        return pos.reshape(H * W, -1)


class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = SwinEncoder()
        self.decoder = HybridDecoder(vocab_size)

    def forward(self, images, tgt):
        memory, (H, W) = self.encoder(images)
        pos2d = pos2d(H, W, images.device)
        memory = memory + pos2d.unsqueeze(0)
        return self.decoder(memory, tgt), (H, W)
