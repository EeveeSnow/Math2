import json
import os
import sys
import re
import tempfile
from datetime import datetime

import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from safetensors.torch import load_file

import wraper 
sys.modules['data'] = wraper 

import gradio as gr
import torch
from datasets import load_dataset
from torchvision import transforms

from interface import beam_search, decode_tokens, predict_latex
from metrics import avg_edit_distance, exact_match, token_accuracy
from image_processing import RandomWidth, ResizePadHW
from wraper import encode_batch

from model import SwiGLiT 
from model_mamba import SwinMambaTex

sys.modules["data"] = wraper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Variables
vocab_obj = None
vocab = None
VOCAB_SIZE = 0
model = None


class CustomVocab:
    def __init__(self, vocab_data):
        if isinstance(vocab_data, dict):
            # Check if keys are stringified integers (e.g., "0", "1")
            sample_key = next(iter(vocab_data.keys()))
            
            if sample_key.isdigit():
                # Format: {"0": "<PAD>", "1": "<SOS>", ...}
                max_idx = max(int(k) for k in vocab_data.keys())
                self.itos = [""] * (max_idx + 1)
                self.stoi = {}
                for k, v in vocab_data.items():
                    idx = int(k)
                    self.itos[idx] = v
                    self.stoi[v] = idx
            else:
                # Format: {"<PAD>": 0, "<SOS>": 1, ...}
                self.stoi = vocab_data
                max_idx = max(int(v) for v in vocab_data.values())
                self.itos = [""] * (max_idx + 1)
                for token, idx in self.stoi.items():
                    self.itos[int(idx)] = token

        elif isinstance(vocab_data, list):
            # Format: ["<PAD>", "<SOS>", ...]
            self.itos = vocab_data
            self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        else:
            raise ValueError("Vocabulary must be a dictionary or a list.")

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get("<UNK>", 3))
    
    def tokenize(self, text):
        tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z]+|\d+|[{}()=+\-*/^_]', text)
        return tokens


image_transform = transforms.Compose([
    transforms.Grayscale(),
    RandomWidth(),
    ResizePadHW(*(384, 384)),
    transforms.ToTensor()
])

def load_custom_model(arch_name, vocab_file, weights_file):
    """Function to dynamically load vocab and safetensors model weights"""
    global model, vocab, vocab_obj, VOCAB_SIZE
    
    if not vocab_file:
        return "Error: Please upload a vocab.json file."
    
    try:
        # 1. Load Vocabulary
        with open(vocab_file.name, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            
        vocab_obj = CustomVocab(vocab_data)
        
        # Create a dictionary mapping int -> string so decode_tokens can use .get()
        vocab = {idx: token for idx, token in enumerate(vocab_obj.itos)}
        VOCAB_SIZE = len(vocab_obj.itos)
        
    except Exception as e:
        return f"Error loading vocabulary: {str(e)}"

    if not weights_file:
        return f"Vocabulary loaded (Size: {VOCAB_SIZE}). Please upload a .safetensors file to load the model."
        
    if not arch_name:
        return "Error: Please select a model architecture."

    try:
        # 2. Initialize the requested architecture
        if arch_name == "SwinMambaTex":
            new_model = SwinMambaTex(vocab_size=VOCAB_SIZE, d_model=512).to(DEVICE)
        elif arch_name == "SwiGLiT":
            new_model = SwiGLiT(vocab_size=VOCAB_SIZE, d_model=512).to(DEVICE)
        else:
            return f"Error: Unknown architecture '{arch_name}'."

        # 3. Load Safetensors weights
        state_dict = load_file(weights_file.name)
        new_model.load_state_dict(state_dict)
        new_model.half()
        new_model.eval()

        
        model = new_model
        return f"Successfully loaded {arch_name}!\nVocab size: {VOCAB_SIZE}\nDevice: {DEVICE}"
    except Exception as e:
        return f"Error loading model weights: {str(e)}"


def process_image(img):
    if model is None or vocab is None:
        return "Error: Please load a model and vocabulary first.", ""
    return predict_latex(img, model, DEVICE, vocab)

def normalize_latex(s):
    s = s.replace(" ", "")
    s = s.replace("_{o}", "_{0}")
    s = re.sub(r'_([a-zA-Z0-9])(?![a-zA-Z0-9])', r'_{\1}', s)
    s = re.sub(r'\^([a-zA-Z0-9])(?![a-zA-Z0-9])', r'^{\1}', s)
    return s

BUCKET_KEYS = [
    "<10", "10-19", "20-29", "30-39", "40-49", 
    "50-59", "60-69", "70-79", "80-89", "90-100", ">100"
]

def _bucket_name(length):
    if length < 10:
        return "<10"
    if length > 100:
        return ">100"
    if length == 100:
        return "90-100"
        
    lower = (length // 10) * 10
    upper = lower + 9
    return f"{lower}-{upper}"


def tokenize_latex(s):
    """Tokenize LaTeX string into commands and single characters, stripping spaces"""
    s = str(s).replace(" ", "")
    # This regex pulls out full latex commands (e.g. \frac) OR individual characters
    return re.findall(r'\\[a-zA-Z]+|.', s)


def calc_bleu(gt, pred):
    smoother = SmoothingFunction().method1
    
    # Properly tokenize mathematical expressions
    gt_tokens = tokenize_latex(gt)
    pred_tokens = tokenize_latex(pred)
    
    if not gt_tokens:
        return 0.0
    return sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)


def _prepare_gradio_data(log_data):
    metrics = log_data["metrics"]
    
    summary_df = pd.DataFrame([{
        "Samples": metrics.get("samples", 0),
        "Token Acc": f"{metrics.get('token_accuracy', 0.0):.4f}",
        "Exact Match": f"{metrics.get('exact_match', 0.0):.4f}",
        "BLEU Score": f"{metrics.get('bleu', 0.0):.4f}",
        "Avg Edit Dist": f"{metrics.get('avg_edit_distance', 0.0):.4f}",
        "Norm Edit Dist": f"{metrics.get('normalized_edit_distance', 0.0):.4f}",
        "Len Abs Error": f"{metrics.get('avg_length_abs_error', 0.0):.4f}"
    }])
    
    buckets = list(log_data["bucket_stats"].keys())
    bucket_df = pd.DataFrame({
        "Length": buckets,
        "Sample Count": [log_data["bucket_stats"][k].get("sample_count", 0) for k in buckets],
        "Exact Match": [log_data["bucket_stats"][k].get("exact_match", 0.0) for k in buckets],
        "BLEU": [log_data["bucket_stats"][k].get("avg_bleu", 0.0) for k in buckets],
        "Avg Edit Distance": [log_data["bucket_stats"][k].get("avg_edit_distance", 0.0) for k in buckets]
    })
    
    details_df = pd.DataFrame(log_data["items"])
    if not details_df.empty:
        details_df["candidates"] = details_df["candidates"].apply(lambda x: " | ".join(x) if isinstance(x, list) else x)
        cols = ["index", "gt_length", "pred_length", "exact_match", "bleu", "edit_distance", "ground_truth", "prediction", "candidates"]
        details_df = details_df[[c for c in cols if c in details_df.columns]]
        
    return summary_df, bucket_df, details_df, log_data

def run_test_dataset(max_samples, batch_size, progress=gr.Progress(track_tqdm=True)):
    if model is None or vocab_obj is None:
        raise ValueError("Cannot run tests without a loaded model and vocabulary.")
        
    max_samples = int(max_samples)
    batch_size = int(batch_size)

    split_name = "test" if max_samples <= 0 else f"test[:{max_samples}]"
    ds = load_dataset("deepcopy/MathWriting-Human", split=split_name)

    all_pred_texts = []
    all_gt_texts = []
    items = []
    
    bucket_raw = {k: [] for k in BUCKET_KEYS}

    for i in progress.tqdm(range(0, len(ds), batch_size), desc="Running test"):
        batch = ds[i : i + batch_size]
        images = [image_transform(img) for img in batch["image"]]
        image_tensor = torch.stack(images).to(DEVICE).half()

        with torch.no_grad():
            predictions = beam_search(model, image_tensor, beam_size=3)

        gt_texts = batch["latex"]
        pred_texts = []

        for local_idx in range(predictions.shape[0]):
            candidates = [decode_tokens(vocab, predictions[local_idx][k]) for k in range(3)]
            pred_text = candidates[0]
            pred_texts.append(pred_text)

            gt = gt_texts[local_idx]
            gt_len = len(''.join(gt.split()))
            pred_len = len(''.join(pred_text.split()))
            
            p_tensor = encode_batch([pred_text], vocab_obj)
            t_tensor = encode_batch([gt], vocab_obj)
            
            ed = avg_edit_distance(p_tensor, t_tensor)
            em = exact_match(p_tensor, t_tensor)
            bleu_val = calc_bleu(gt, pred_text)

            bucket = _bucket_name(gt_len)
            bucket_raw[bucket].append({"exact_match": em, "edit_distance": ed, "bleu": bleu_val})

            items.append({
                "index": i + local_idx,
                "ground_truth": gt,
                "prediction": pred_text,
                "candidates": candidates,
                "gt_length": gt_len,
                "pred_length": pred_len,
                "exact_match": float(em),
                "edit_distance": float(ed),
                "bleu": float(bleu_val)
            })

    all_pred_texts.extend(pred_texts)
    all_gt_texts.extend(gt_texts)

    pred_tensor = encode_batch(all_pred_texts, vocab_obj)
    tgt_tensor = encode_batch(all_gt_texts, vocab_obj)

    token_acc = token_accuracy(pred_tensor, tgt_tensor)
    em_score = 0
    avg_bleu = 0.0
    edit_score = avg_edit_distance(pred_tensor, tgt_tensor)

    norm_ed = 0.0
    len_abs_err = 0.0
    for item in items:
        norm_ed += item["edit_distance"] / max(item["gt_length"], 1)
        len_abs_err += abs(item["pred_length"] - item["gt_length"])
        avg_bleu += item["bleu"]

    total = max(len(items), 1)
    norm_ed /= total
    len_abs_err /= total
    avg_bleu /= total

    bucket_stats = {}
    for bucket in BUCKET_KEYS:
        values = bucket_raw[bucket]
        if not values:
            bucket_stats[bucket] = {"sample_count": 0, "exact_match": 0.0, "avg_edit_distance": 0.0, "avg_bleu": 0.0}
        else:
            bucket_stats[bucket] = {
                "sample_count": len(values),
                "exact_match": sum(v["exact_match"] for v in values) / len(values),
                "avg_edit_distance": sum(v["edit_distance"] for v in values) / len(values),
                "avg_bleu": sum(v["bleu"] for v in values) / len(values)
            }
            em_score += sum(v["exact_match"] for v in values)
            
    em_score /= total
    log_data = {
        "created_at": datetime.utcnow().isoformat(),
        "metrics": {
            "samples": len(items),
            "token_accuracy": float(token_acc),
            "exact_match": float(em_score),
            "bleu": float(avg_bleu),
            "avg_edit_distance": float(edit_score),
            "normalized_edit_distance": float(norm_ed),
            "avg_length_abs_error": float(len_abs_err),
        },
        "bucket_stats": bucket_stats,
        "items": items,
    }

    fd, out_path = tempfile.mkstemp(prefix="test_logs_", suffix=".json")
    os.close(fd)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    summary_df, bucket_df, details_df, json_payload = _prepare_gradio_data(log_data)
    # Output bucket_df 4 times for the 4 separate bar plots
    return summary_df, bucket_df, bucket_df, bucket_df, bucket_df, details_df, json_payload, out_path


def load_test_logs(file_obj):
    if file_obj is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    with open(file_obj.name, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    summary_df, bucket_df, details_df, json_payload = _prepare_gradio_data(log_data)
    return summary_df, bucket_df, bucket_df, bucket_df, bucket_df, details_df, json_payload


def load_training_graphs(epoch_file, step_file):
    epoch_loss_df = pd.DataFrame()
    epoch_lr_df = pd.DataFrame()
    step_loss_df = pd.DataFrame()

    if epoch_file is not None:
        try:
            df_epoch = pd.read_csv(epoch_file.name)
            if all(col in df_epoch.columns for col in ['epoch', 'train_loss', 'val_loss']):
                epoch_loss_df = df_epoch.melt(
                    id_vars=['epoch'], 
                    value_vars=['train_loss', 'val_loss'], 
                    var_name='loss_type', 
                    value_name='loss'
                )
            if 'epoch' in df_epoch.columns and 'lr' in df_epoch.columns:
                epoch_lr_df = df_epoch[['epoch', 'lr']]
        except Exception as e:
            print(f"Error reading epoch file: {e}")

    if step_file is not None:
        try:
            df_step = pd.read_csv(step_file.name)
            if 'step' in df_step.columns and 'loss' in df_step.columns:
                step_loss_df = df_step[['step', 'loss']]
        except Exception as e:
            print(f"Error reading step file: {e}")

    return epoch_loss_df, epoch_lr_df, step_loss_df


with gr.Blocks() as demo:
    gr.Markdown("# LaTeX Translation App")

    with gr.Tabs():
        # --- TAB 1: CONVERTER & MODEL LOADING ---
        with gr.TabItem("Converter"):
            gr.Markdown("Загрузите изображение математического выражения и получите соответствующий код LaTeX.")
            
            with gr.Accordion("Settings & Model Loading", open=True):
                with gr.Row():
                    arch_dropdown = gr.Dropdown(choices=["SwinMambaTex", "SwiGLiT"], label="Model Architecture", value="SwinMambaTex")
                    upload_vocab = gr.File(label="Upload Vocabulary (vocab.json)", file_types=[".json"])
                    upload_weights = gr.File(label="Upload Weights (.safetensors)", file_types=[".safetensors"])
                with gr.Row():
                    load_model_btn = gr.Button("Load Configured Model", variant="primary")
                    model_status = gr.Textbox(label="Model Status", value="No model loaded. Please configure and load.", interactive=False)
                    
                load_model_btn.click(
                    fn=load_custom_model, 
                    inputs=[arch_dropdown, upload_vocab, upload_weights], 
                    outputs=[model_status]
                )

            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Загрузить изображение")
                    submit_btn = gr.Button("Преобразовать в LaTeX", variant="primary")
                with gr.Column():
                    output_code = gr.Textbox(label="Полученный LaTeX код", lines=4)
                    output_render = gr.Markdown(label="Полученное выражение")

            submit_btn.click(fn=process_image, inputs=[input_img], outputs=[output_code, output_render])

        # --- TAB 2: TEST DATASET STATS ---
        with gr.TabItem("Test dataset stats"):
            gr.Markdown("Запуск теста на test split с логированием и метриками.")
            with gr.Row():
                max_samples = gr.Number(value=50, precision=0, label="Максимум примеров (0 = весь test split)")
                batch_size = gr.Number(value=8, precision=0, label="Batch size")
            run_btn = gr.Button("Start test on dataset", variant="primary")

            summary_table = gr.Dataframe(label="Global Test Results", interactive=False)
            
            with gr.Row():
                plot_samples = gr.BarPlot(x="Length", y="Sample Count", title="Samples per Length Bucket", tooltip=["Length", "Sample Count"])
                plot_em = gr.BarPlot(x="Length", y="Exact Match", title="Exact Match by Length", tooltip=["Length", "Exact Match"])
            with gr.Row():
                plot_bleu = gr.BarPlot(x="Length", y="BLEU", title="BLEU Score by Length", tooltip=["Length", "BLEU"])
                plot_ed = gr.BarPlot(x="Length", y="Avg Edit Distance", title="Avg Edit Distance by Length", tooltip=["Length", "Avg Edit Distance"])
            
            details_table = gr.Dataframe(label="Per-image Drilldown", interactive=False)
            logs_json = gr.JSON(label="Raw logs")
            download_logs = gr.File(label="Download logs JSON")

            run_btn.click(
                fn=run_test_dataset,
                inputs=[max_samples, batch_size],
                outputs=[summary_table, plot_samples, plot_em, plot_bleu, plot_ed, details_table, logs_json, download_logs],
            )

            gr.Markdown("### Load existing test logs")
            upload_logs = gr.File(file_types=[".json"], label="Upload logs JSON")
            load_btn = gr.Button("Load logs")
            
            load_btn.click(
                fn=load_test_logs,
                inputs=[upload_logs],
                outputs=[summary_table, plot_samples, plot_em, plot_bleu, plot_ed, details_table, logs_json],
            )

        # --- TAB 3: TRAINING GRAPHS ---
        with gr.TabItem("Training Graphs"):
            gr.Markdown("Загрузите CSV файлы логов для просмотра графиков обучения.")
            
            with gr.Row():
                upload_epoch_csv = gr.File(label="Epoch Logs (epoch, train_loss, val_loss, lr)", file_types=[".csv"])
                upload_step_csv = gr.File(label="Step Logs (epoch, step, batch_idx, loss, lr)", file_types=[".csv"])
            
            plot_graphs_btn = gr.Button("Построить графики", variant="primary")
            
            with gr.Row():
                epoch_loss_plot = gr.LinePlot(x="epoch", y="loss", color="loss_type", title="Train vs Validation Loss (Epoch)", tooltip=["epoch", "loss_type", "loss"])
                epoch_lr_plot = gr.LinePlot(x="epoch", y="lr", title="Learning Rate Progression", tooltip=["epoch", "lr"])
            
            with gr.Row():
                step_loss_plot = gr.LinePlot(x="step", y="loss", title="Training Loss (Step)", tooltip=["step", "loss"])
                
            plot_graphs_btn.click(
                fn=load_training_graphs,
                inputs=[upload_epoch_csv, upload_step_csv],
                outputs=[epoch_loss_plot, epoch_lr_plot, step_loss_plot]
            )


if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=False, theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink))