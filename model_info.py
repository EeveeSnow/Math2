from torchinfo import summary
import torch
from models.model_MOE import SwinMoETex
from models.model_conv import SwinGConvTex
from models.model_transformer import SwinTransformerTex

device = "cuda"
vocab_size = 256
B, T = 1, 100

model = SwinMoETex(vocab_size).to(device, dtype=torch.bfloat16)


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