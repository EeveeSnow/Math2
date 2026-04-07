import torch
from safetensors.torch import save_file
import json
import wraper

ckpt = torch.load("checkpoints/conv/epoch_020.pt", map_location="cpu",weights_only=False)

state_dict = ckpt["model_state_dict"]
print(len(state_dict), "original keys")
state_dict_bf16 = {}

state_dict = {
    k.replace("_orig_mod.", ""): v
    for k, v in state_dict.items()
}

for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        if torch.is_floating_point(v):
            state_dict_bf16[k] = v.to(torch.bfloat16)
        else:
            state_dict_bf16[k] = v

vocab = ckpt["vocab"].itos
print(len(state_dict_bf16), "tensor keys")
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
# save
save_file(state_dict_bf16, "SwinGConvTex_bf16_020.safetensors")