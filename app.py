import torch
import torch.nn as nn
import timm
import gradio as gr
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import re
from model import Im2LatexModel
from interface import beam_search
from data import Vocabulary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_WEIGHTS_PATH = "weights\\best_model.pt"
MAX_LEN = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading checkpoint from {MODEL_WEIGHTS_PATH}...")
checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)

stoi = checkpoint['vocab']
IDX2TOKEN = {v: k for k, v in stoi.items()}

VOCAB_SIZE = len(IDX2TOKEN)

print(f"Loaded vocabulary with {VOCAB_SIZE} tokens.")

image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
print(list(IDX2TOKEN.keys())[:10])
print("Initializing model...")
model = Im2LatexModel(vocab_size=VOCAB_SIZE, d_model=512).to(DEVICE)

model.load_state_dict(checkpoint['model'])
model.eval()

def decode_tokens(vocab_obj, token_ids):
    itos = vocab_obj
    latex = []
    for tid in token_ids:
        tid = tid.item() if hasattr(tid, 'item') else int(tid)
        
        if tid == 2:
            break
            
        if tid not in [0, 1, 3]: 
            token_str = itos.get(tid, "")
            latex.append(str(token_str))
            
    return " ".join(latex)

def predict_latex(image):
    if image is None:
        return "", ""
    
    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
    prediction = beam_search(model, img_tensor, beam_size=3)
    print(prediction)
    latex_str = decode_tokens(IDX2TOKEN, prediction[0])
    print(latex_str)
    rendered_math = f"$$ {latex_str} $$"
    
    return latex_str, rendered_math


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
        fn=predict_latex,
        inputs=[input_img],
        outputs=[output_code, output_render]
    )

if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=False)