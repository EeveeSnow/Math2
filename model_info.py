from torchinfo import summary
import torch
from model_mamba_1layer import SwinMambaTex
from model_conv import SwinGConvTex
from model_transformer import SwinTransformerTex

device = "cuda"
vocab_size = 230
B, T = 1, 100

model = SwinMambaTex(vocab_size).to(device, dtype=torch.bfloat16)


print(summary(
    model, 
    input_size=[(B, 1, 384, 384), (B, T)], 
    dtypes=[torch.bfloat16, torch.long],
    device=device
))

model = SwinGConvTex(vocab_size).to(device, dtype=torch.bfloat16)
print(summary(
    model, 
    input_size=[(B, 1, 384, 384), (B, T)], 
    dtypes=[torch.bfloat16, torch.long],
    device=device
))

model = SwinTransformerTex(vocab_size).to(device, dtype=torch.bfloat16)
print(summary(
    model, 
    input_size=[(B, 1, 384, 384), (B, T)], 
    dtypes=[torch.bfloat16, torch.long],
    device=device
))