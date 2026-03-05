import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SwinEncoder(nn.Module):
    def __init__(self, d_model=512, swin_model="swin_base_patch4_window12_384"):
        super().__init__()

        # Create Swin backbone
        self.backbone = timm.create_model(
            swin_model,
            pretrained=True,
            features_only=True,
            in_chans=1,
            img_size=(384, 384),
            dynamic_img_size=True
        )

        # Dynamically determine channels of stage3 and stage4
        ch3 = self.backbone.feature_info.channels()[-2]  # stage3
        ch4 = self.backbone.feature_info.channels()[-1]  # stage4
        self.in_ch = ch3 + ch4

        self.proj = nn.Linear(self.in_ch, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        feats = self.backbone(x)

        # stage3 and stage4 features
        f3 = feats[-2]  # (B, C3, H3, W3)
        f4 = feats[-1]  # (B, C4, H4, W4)
        f3 = f3.permute(0, 3, 2, 1)
        f4 = f4.permute(0, 3, 2, 1)
        B, C4, H4, W4 = f4.shape

        # Upsample f3 до разрешения f4 (NCHW!)
        f3 = F.interpolate(f3, size=(H4, W4), mode="bilinear", align_corners=False)  # NCHW
        # Теперь f3: (B, C3, H4, W4)
        # Переводим в NHWC для concat
        f3 = f3.permute(0, 2, 3, 1)  # (B, H4, W4, C3)
        f4 = f4.permute(0, 2, 3, 1)  # (B, H4, W4, C4)
        # Конкатенация по каналам
        f = torch.cat([f3, f4], dim=-1)  # (B, H4, W4, C3+C4)

        # Проекция
        f = self.proj(f)  # (B, H4, W4, d_model)
        # print("After proj:", f.shape)  # (16, 12, 12, 512)

        # Flatten
        f = f.view(B, -1, f.shape[-1])  # (B, H4*W4, d_model)
        f = self.norm(f)

        return f, (H4, W4)    

class GatedConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=15):
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
    def __init__(self, d_model=512):
        super().__init__()
        # We'll split d_model equally between height and width embeddings
        self.d_model = d_model

    def forward(self, H, W, device):
        # Dynamic embeddings based on H and W
        d_half = self.d_model // 2

        # Learnable embeddings for height and width
        y_embed = nn.Parameter(torch.randn(H, d_half, device=device))
        x_embed = nn.Parameter(torch.randn(W, d_half, device=device))

        # Expand and combine
        pos = torch.cat([
            y_embed[:, None, :].expand(H, W, -1),  # (H, W, d/2)
            x_embed[None, :, :].expand(H, W, -1)   # (H, W, d/2)
        ], dim=-1)  # (H, W, d_model)

        return pos.reshape(H * W, self.d_model)  # Flatten to (H*W, d_model)
    

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, decoder_depth=6, max_len=512):
        super().__init__()
        self.encoder = SwinEncoder(d_model)
        self.decoder = HybridDecoder(vocab_size, d_model, depth=decoder_depth, max_len=max_len)
        self.d_model = d_model

    def forward(self, images, tgt):
        memory, (H, W) = self.encoder(images)

        # Dynamic positional embeddings
        device = images.device
        pos_embeddings = Pos2D(d_model=self.d_model)(H, W, device)
        memory = memory + pos_embeddings.unsqueeze(0)  # (1, H*W, d_model) -> broadcast to batch

        # Pass to decoder
        return self.decoder(memory, tgt), (H, W)