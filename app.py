import json
import os

import sys
import re
import wraper 
sys.modules['data'] = wraper 

import tempfile
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torchvision import transforms

import wraper
from interface import beam_search, decode_tokens, predict_latex
from metrics import avg_edit_distance, exact_match, token_accuracy
from image_processing import RandomWidth, ResizePadHW
from model import Im2LatexModel
from wraper import encode_batch

sys.modules["data"] = wraper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS_PATH = "checkpoints\\best_model.pt"

print(f"Loading checkpoint from {MODEL_WEIGHTS_PATH}...")
checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)

vocab_obj = checkpoint["vocab"]
vocab = vocab_obj.itos
VOCAB_SIZE = len(vocab)

print(f"Loaded vocabulary with {VOCAB_SIZE} tokens.")
print("Initializing model...")
model = Im2LatexModel(vocab_size=VOCAB_SIZE, d_model=512).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image_transform = transforms.Compose([
            transforms.Grayscale(),
            RandomWidth(),
            ResizePadHW(*(384, 384)),
            transforms.ToTensor()])


def process_image(img):
    return predict_latex(img, model, DEVICE, vocab)


def normalize_latex(s):
    s = s.replace(" ", "") # Remove spaces
    s = s.replace("_{o}", "_{0}") # Map letter 'o' to zero
    s = re.sub(r'_([a-zA-Z0-9])(?![a-zA-Z0-9])', r'_{\1}', s) # x_2 to x_{2}
    s = re.sub(r'\^([a-zA-Z0-9])(?![a-zA-Z0-9])', r'^{\1}', s) # x^2 to x^{2}
    return s

def _bucket_name(length):
    if length <= 20:
        return "<=20"
    if length <= 40:
        return "21-40"
    if length <= 80:
        return "41-80"
    return ">80"


def _plot_bucket_stats(bucket_stats):
    labels = list(bucket_stats.keys())
    exact_vals = [bucket_stats[k]["exact_match"] for k in labels]
    edit_vals = [bucket_stats[k]["avg_edit_distance"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, exact_vals, color="#4f46e5")
    axes[0].set_title("Exact match by LaTeX length")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Exact match")

    axes[1].bar(labels, edit_vals, color="#059669")
    axes[1].set_title("Avg edit distance by LaTeX length")
    axes[1].set_ylabel("Distance")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def _render_logs(log_data):
    metrics = log_data["metrics"]
    summary = (
        "## Test dataset results\n"
        f"- Samples: **{metrics['samples']}**\n"
        f"- Token accuracy: **{metrics['token_accuracy']:.4f}**\n"
        f"- Exact match: **{metrics['exact_match']:.4f}**\n"
        f"- Avg edit distance: **{metrics['avg_edit_distance']:.4f}**\n"
        f"- Normalized edit distance: **{metrics['normalized_edit_distance']:.4f}**\n"
        f"- Avg absolute LaTeX length error: **{metrics['avg_length_abs_error']:.4f}**"
    )

    details_lines = ["## Per-image drilldown"]
    for item in log_data["items"]:
        details_lines.append(
            f"### Sample #{item['index']}\n"
            f"- GT length: {item['gt_length']}\n"
            f"- Pred length: {item['pred_length']}\n"
            f"- Exact: {item['exact_match']}\n"
            f"- Edit distance: {item['edit_distance']}\n"
            f"- Ground truth: `{item['ground_truth']}`\n"
            f"- Prediction: `{item['prediction']}`\n"
            f"- Top candidates:\n"
            + "\n".join([f"  - `{cand}`" for cand in item["candidates"]])
        )

    fig = _plot_bucket_stats(log_data["bucket_stats"])
    return summary, fig, "\n\n".join(details_lines), log_data


def run_test_dataset(max_samples, batch_size, progress=gr.Progress(track_tqdm=True)):
    max_samples = int(max_samples)
    batch_size = int(batch_size)

    split_name = "test" if max_samples <= 0 else f"test[:{max_samples}]"
    ds = load_dataset("deepcopy/MathWriting-Human", split=split_name)

    all_pred_texts = []
    all_gt_texts = []
    items = []
    bucket_raw = {"<=20": [], "21-40": [], "41-80": [], ">80": []}

    for i in progress.tqdm(range(0, len(ds), batch_size), desc="Running test"):
        batch = ds[i : i + batch_size]
        images = [image_transform(img) for img in batch["image"]]
        image_tensor = torch.stack(images).to(DEVICE)

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

            bucket = _bucket_name(gt_len)
            bucket_raw[bucket].append({"exact_match": em, "edit_distance": ed})

            items.append(
                {
                    "index": i + local_idx,
                    "ground_truth": gt,
                    "prediction": pred_text,
                    "candidates": candidates,
                    "gt_length": gt_len,
                    "pred_length": pred_len,
                    "exact_match": float(em),
                    "edit_distance": float(ed),
                }
            )

    all_pred_texts.extend(pred_texts)
    all_gt_texts.extend(gt_texts)

    pred_tensor = encode_batch(all_pred_texts, vocab_obj)
    tgt_tensor = encode_batch(all_gt_texts, vocab_obj)

    token_acc = token_accuracy(pred_tensor, tgt_tensor)
    em_score = 0
    edit_score = avg_edit_distance(pred_tensor, tgt_tensor)

    norm_ed = 0.0
    len_abs_err = 0.0
    for item in items:
        norm_ed += item["edit_distance"] / max(item["gt_length"], 1)
        len_abs_err += abs(item["pred_length"] - item["gt_length"])

    total = max(len(items), 1)
    norm_ed /= total
    len_abs_err /= total

    bucket_stats = {}
    for bucket, values in bucket_raw.items():
        if not values:
            bucket_stats[bucket] = {"exact_match": 0.0, "avg_edit_distance": 0.0}
        else:
            print(f"{bucket} : {len(values)}" )
            bucket_stats[bucket] = {
                "exact_match": sum(v["exact_match"] for v in values) / len(values),
                "avg_edit_distance": sum(v["edit_distance"] for v in values) / len(values),
            }
            em_score += sum(v["exact_match"] for v in values)
    em_score /= len(items)
    log_data = {
        "created_at": datetime.utcnow().isoformat(),
        "metrics": {
            "samples": len(items),
            "token_accuracy": float(token_acc),
            "exact_match": float(em_score),
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

    summary, fig, details, json_payload = _render_logs(log_data)
    return summary, fig, details, json_payload, out_path


def load_test_logs(file_obj):
    if file_obj is None:
        return "No log file uploaded.", None, "", None

    with open(file_obj.name, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    summary, fig, details, json_payload = _render_logs(log_data)
    return summary, fig, details, json_payload


with gr.Blocks() as demo:
    gr.Markdown("# LaTeX App")

    with gr.Tabs():
        with gr.TabItem("Converter"):
            gr.Markdown("Загрузите изображение математического выражения и получите соответствующий код LaTeX.")
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Загрузить изображение")
                    submit_btn = gr.Button("Преобразовать в LaTeX", variant="primary")
                with gr.Column():
                    output_code = gr.Textbox(label="Полученный LaTeX код", lines=4)
                    output_render = gr.Markdown(label="Полученное выражение")

            submit_btn.click(fn=process_image, inputs=[input_img], outputs=[output_code, output_render])

        with gr.TabItem("Test dataset stats"):
            gr.Markdown("Запуск теста на test split с логированием и метриками.")
            with gr.Row():
                max_samples = gr.Number(value=50, precision=0, label="Максимум примеров (0 = весь test split)")
                batch_size = gr.Number(value=8, precision=0, label="Batch size")
            run_btn = gr.Button("Start test on dataset", variant="primary")

            summary_md = gr.Markdown()
            stats_plot = gr.Plot(label="Length based metrics")
            details_md = gr.Markdown()
            logs_json = gr.JSON(label="Raw logs")
            download_logs = gr.File(label="Download logs JSON")

            run_btn.click(
                fn=run_test_dataset,
                inputs=[max_samples, batch_size],
                outputs=[summary_md, stats_plot, details_md, logs_json, download_logs],
            )

            gr.Markdown("### Load existing test logs")
            upload_logs = gr.File(file_types=[".json"], label="Upload logs JSON")
            load_btn = gr.Button("Load logs")
            load_btn.click(
                fn=load_test_logs,
                inputs=[upload_logs],
                outputs=[summary_md, stats_plot, details_md, logs_json],
            )


if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=False)
