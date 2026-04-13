from transformers import PretrainedConfig, PreTrainedModel
from models.model_blocks import SwinEncoder, Pos2D
from models.model_MOE import MambaMOEDecoder
from safetensors.torch import load_file
import torch


class SwinMoETexConfig(PretrainedConfig):
    model_type = "swin_moe_tex"

    def __init__(self, vocab_size=256, d_model=512, max_len=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

class SwinMoETex(PreTrainedModel):
    config_class = SwinMoETexConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = SwinEncoder(config.d_model)
        self.pos_encoder = Pos2D(d_model=config.d_model)
        self.decoder = MambaMOEDecoder(config.vocab_size, config.d_model, max_len=config.max_len)

    def forward(self, images, tgt):
        memory, (H, W) = self.encoder(images)
        memory = memory + self.pos_encoder(H, W)
        return self.decoder(memory, tgt)


config = SwinMoETexConfig(vocab_size=256, d_model=512)
model = SwinMoETex(config)

state_dict = load_file("outputs/moe_bf16.safetensors")

model.load_state_dict(state_dict)
model = model.to(torch.bfloat16)
model.push_to_hub("EeveeSnow/SwinMoETex")