import torch
import torch.nn as nn
from model import Im2LatexModel as model



        
def generate(model, image, max_len=512):
    memory, _ = model.encoder(image)
    out = torch.tensor([[SOS]], device=image.device)

    for _ in range(max_len):
        logits = model.decoder(memory, out)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
        if next_tok.item() == EOS:
            break
    return out
