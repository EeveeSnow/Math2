import torch
import json
import os
import re

import pandas as pd
import plotly.express as px

from safetensors.torch import save_file
from torchinfo import summary

import interface.configs as conf
from wraper import predict_latex


class CustomVocab:
    def __init__(self, vocab_data):
        if isinstance(vocab_data, dict):
            sample_key = next(iter(vocab_data.keys()))
            if sample_key.isdigit():
                max_idx = max(int(k) for k in vocab_data.keys())
                self.itos = [""] * (max_idx + 1)
                self.stoi = {}
                for k, v in vocab_data.items():
                    idx = int(k)
                    self.itos[idx] = v
                    self.stoi[v] = idx
            else:
                self.stoi = vocab_data
                max_idx = max(int(v) for v in vocab_data.values())
                self.itos = [""] * (max_idx + 1)
                for token, idx in self.stoi.items():
                    self.itos[int(idx)] = token
        elif isinstance(vocab_data, list):
            self.itos = vocab_data
            self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        else:
            raise ValueError("Словарь должен быть словарем (dict) или списком (list).")

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get("<UNK>", 3))
    
    def tokenize(self, text):
        tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z]+|\d+|[{}()=+\-*/^_]', text)
        return tokens

def process_image(img):
    if conf.model is None or conf.vocab is None:
        return "Ошибка: Сначала загрузите модель и словарь.", ""
    return predict_latex(img, conf.model, conf.DEVICE, conf.vocab)

def _bucket_name(length):
    if length < 10: 
        return "<10"
    if length >= 60: 
        return ">60"
    lower = (length // 10) * 10
    return f"{lower}-{lower+9}"

def tokenize_latex(s):
    return re.findall(r'\\[a-zA-Z]+|.', str(s).replace(" ", ""))

def apply_plot_theme(fig):
    fig.update_layout(
        paper_bgcolor="#fafafa",
        plot_bgcolor="#fafafa",
        font=dict(family="Nunito", color="#0a0f1e"),
        title_font=dict(family="Nunito", color="#0a0f1e"),
        legend_title_font=dict(family="Nunito", color="#0a0f1e"),
    )
    return fig

def _prepare_gradio_data(log_data):
    metrics = log_data["metrics"]
    summary_df = pd.DataFrame([{
        "Примеров": metrics.get("samples", 0),
        "Точность токенов": f"{metrics.get('token_accuracy', 0.0):.4f}",
        "Exact Match": f"{metrics.get('exact_match', 0.0):.4f}",
        "AST/Struct Score": f"{metrics.get('avg_ast_score', 0.0):.4f}",
        "BLEU Score": f"{metrics.get('bleu', 0.0):.4f}",
        "Token CER": f"{metrics.get('avg_cer', 0.0):.4f}",
        "Token ED (Ср.)": f"{metrics.get('avg_token_ed', 0.0):.2f}",
        "Tok/Sec": f"{metrics.get('throughput_tokens_per_sec', 0.0):.2f}",
        "Latency/Token (ms)": f"{metrics.get('latency_per_token_ms', 0.0):.2f}"
    }])
    
    buckets = list(log_data["bucket_stats"].keys())
    stats = log_data["bucket_stats"]
    
    bucket_df = pd.DataFrame({
        "Длина": buckets,
        "Samples": [stats[k].get("sample_count", 0) for k in buckets],
        "Exact Match": [stats[k].get("exact_match", 0.0) for k in buckets],
        "BLEU": [stats[k].get("avg_bleu", 0.0) for k in buckets],
        "Structural Score": [stats[k].get("avg_struct_score", 0.0) for k in buckets],
        "Ins": [stats[k].get("avg_ins", 0.0) for k in buckets],
        "Del": [stats[k].get("avg_del", 0.0) for k in buckets],
        "Sub": [stats[k].get("avg_sub", 0.0) for k in buckets],
        "Throughput": [stats[k].get("tokens_per_sec", 0.0) for k in buckets],
        "Token CER": [stats[k].get("avg_token_cer", 0.0) for k in buckets],
        "Token ED": [stats[k].get("avg_token_ed", 0.0) for k in buckets]
    })
    
    fig_samples = px.bar(bucket_df, x="Длина", y="Samples", text_auto=True, title="Распределение длин")
    fig_em = px.bar(bucket_df, x="Длина", y="Exact Match", text_auto='.3f', title="Exact Match vs Длина")
    fig_errs = px.bar(bucket_df, x="Длина", y=["Ins", "Del", "Sub"], title="Декомпозиция ошибок vs Длина", barmode='stack')
    fig_struct = px.line(bucket_df, x="Длина", y="Structural Score", markers=True, title="Structural/AST Score vs Длина")
    fig_cer = px.line(bucket_df, x="Длина", y="Token CER", markers=True, title="Token CER vs Длина (Меньше - лучше)")
    fig_token_ed = px.line(bucket_df, x="Длина", y="Token ED", markers=True, title="Token Edit Distance vs Длина")
    fig_bleu = px.line(bucket_df, x="Длина", y="BLEU", markers=True, title="BLEU Score vs Длина")
    fig_thr = px.line(bucket_df, x="Длина", y="Throughput", markers=True, title="Пропускная способность (Токены/сек)")
    
    return summary_df, apply_plot_theme(fig_samples), apply_plot_theme(fig_em), apply_plot_theme(fig_errs), apply_plot_theme(fig_struct), apply_plot_theme(fig_cer), apply_plot_theme(fig_token_ed), apply_plot_theme(fig_bleu), apply_plot_theme(fig_thr)

def convert_checkpoint(file, prefix):
    try:
        ckpt = torch.load(file.name, map_location="cpu", weights_only=False)

        state_dict = ckpt["model_state_dict"]
        print(len(state_dict), "original keys")

        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }

        state_dict_bf16 = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    state_dict_bf16[k] = v.to(torch.bfloat16)
                else:
                    state_dict_bf16[k] = v

        print(len(state_dict_bf16), "tensor keys")


        conf.vocab = ckpt["vocab"].itos

        os.makedirs("outputs", exist_ok=True)

        safetensor_path = f"outputs/{prefix}_bf16.safetensors"
        vocab_path = "outputs/vocab.json"

        save_file(state_dict_bf16, safetensor_path)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(conf.vocab, f, ensure_ascii=False, indent=2)

        return safetensor_path, vocab_path, "Conversion successful"
    except Exception as e:
        return None, None, f"Error: {str(e)}"
    
def run_summary(B, T):
    if conf.model is None or conf.vocab_obj is None:
        raise ValueError("Загрузите модель и словарь перед запуском.")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = summary(
            conf.model,
            input_size=[(B, 1, 384, 384), (B, T)],
            dtypes=[torch.bfloat16, torch.long],
            device=device,
            verbose=0
        )

        return str(result), conf.ARCH, conf.VOCAB_SIZE

    except Exception as e:
        return f"Error: {str(e)}"