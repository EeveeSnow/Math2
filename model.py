import torch
import torch.nn as nn
import timm

class SwinEncoder(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            features_only=True,
            in_chans=1
        )
        # 1024 dim -> 512 dim
        self.proj = nn.Linear(1024, d_model)

    def forward(self, x):
        feats = self.backbone(x)[-1]
        
        if feats.shape[1] == 1024:
            # NCHW -> NHWC
            feats = feats.permute(0, 2, 3, 1)
        B, H, W, C = feats.shape
        
        # (B, H, W, 1024) -> (B, H, W, 512)
        feats = self.proj(feats)
        
        # (B, H*W, 512)
        feats = feats.flatten(1, 2)
        
        return feats, (H, W)

class GatedConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_model * 2,
            kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        y = self.conv(x.transpose(1, 2))[:, :, :-self.conv.padding[0]]
        y = y.transpose(1, 2)

        u, v = y.chunk(2, dim=-1)
        x = x + self.proj(u * torch.sigmoid(v))
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.conv = GatedConvBlock(d_model)
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
        x = self.ln2(self.conv(x))
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
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.encoder = SwinEncoder(d_model)
        self.decoder = HybridDecoder(vocab_size, d_model)
        self.pos_enc = Pos2D(d_model=d_model)

    def forward(self, images, tgt):
        memory, (H, W) = self.encoder(images)
        pos_embeddings = self.pos_enc(H, W, images.device)
        memory = memory + pos_embeddings.unsqueeze(0)
        
        return self.decoder(memory, tgt), (H, W)