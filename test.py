import torch
from model_mamba_2layer import SwinMambaTex

device = "cuda"
vocab_size = 500
B, T = 2, 32

model = SwinMambaTex(vocab_size).to(device)
model.train()

images  = torch.randn(B, 1, 384, 384, device=device)
tokens  = torch.randint(0, vocab_size, (B, T), device=device)


with torch.no_grad():
    d, r, cov = model(images, tokens)
    for name, val in [("draft", d), ("refine", r), ("coverage", cov)]:
        print(f"{name:10s}  finite={torch.isfinite(val).all().item()}"
              f"  max={val.abs().max().item():.3f}")


from torch.amp import autocast
with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
    d, r, cov = model(images, tokens)
    for name, val in [("draft bf16", d), ("refine bf16", r), ("cov bf16", cov)]:
        print(f"{name:12s}  finite={torch.isfinite(val).all().item()}"
              f"  max={val.abs().max().item():.3f}")