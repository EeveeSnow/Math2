import sys
import wraper 
sys.modules['data'] = wraper 


import torch
import torch.nn as nn
import timm
import gradio as gr
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import re
from model import Im2LatexModel
from interface import beam_search
from wraper import Vocabulary
from interface import predict_latex


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_WEIGHTS_PATH = "checkpoints\\best_model.pt"
MAX_LEN = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading checkpoint from {MODEL_WEIGHTS_PATH}...")
checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)

vocab = checkpoint['vocab'].itos

VOCAB_SIZE = len(vocab)

print(f"Loaded vocabulary with {VOCAB_SIZE} tokens.")
print(checkpoint['train_loss'])
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
print(list(vocab.keys())[:10])
print("Initializing model...")
model = Im2LatexModel(vocab_size=VOCAB_SIZE, d_model=512).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()



def process_image(img):
    return predict_latex(img, model, DEVICE, vocab)


with gr.Blocks() as demo:
    gr.Markdown("# LaTeX Converter")
    gr.Markdown("Загрузите изобрадение математичемского выражения и получить соответствующий код LaTeX.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Загрузить изображение")
            submit_btn = gr.Button("Преобразовать в LaTeX", variant="primary")
            
        with gr.Column():
            output_code = gr.Textbox(label="Полученнный LaTeX код", lines=3)
            output_render = gr.Markdown(label="Полученное выражение")
            
    submit_btn.click(
        fn=process_image,
        inputs=[input_img],
        outputs=[output_code, output_render]
    )

if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=False)