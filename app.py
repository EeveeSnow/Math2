import json
import os
import sys
import re
import tempfile
import time
from datetime import datetime, timezone

import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from safetensors.torch import load_file
import plotly.express as px
import plotly.graph_objects as go
from safetensors.torch import save_file
from torchinfo import summary

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

from model_conv import SwinGConvTex
from model_transformer import SwinTransformerTex
from model_mamba_1layer import SwinMambaTex


try:
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Внимание: SymPy или antlr4 не установлены. AST-метрика будет использовать fallback.")

sys.modules["data"] = wraper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_obj = None
vocab = None
VOCAB_SIZE = 0
ARCH = None
model = None

STRUCTURAL_TOKENS = {
    '^', '_', '{', '}', '\\frac', '\\sqrt', '\\sum', '\\int', 
    '\\left', '\\right', '(', ')', '[', ']', '\\begin', '\\end'
}


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


image_transform = transforms.Compose([
    transforms.Grayscale(),
    RandomWidth(),
    ResizePadHW(*(384, 384)),
    transforms.ToTensor()
])

def load_custom_model(arch_name, vocab_file, weights_file):
    global model, vocab, vocab_obj, VOCAB_SIZE, ARCH
    if not vocab_file: return "Ошибка: Пожалуйста, загрузите файл vocab.json."
    
    try:
        with open(vocab_file.name, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab_obj = CustomVocab(vocab_data)
        vocab = {idx: token for idx, token in enumerate(vocab_obj.itos)}
        VOCAB_SIZE = len(vocab_obj.itos)
    except Exception as e:
        return f"Ошибка при загрузке словаря: {str(e)}"

    if not weights_file:
        return f"Словарь загружен (Размер: {VOCAB_SIZE}). Загрузите файл .safetensors для инициализации модели."
    if not arch_name:
        return "Ошибка: Выберите архитектуру модели."

    try:
        if arch_name == "SwinMambaTex":
            if DEVICE.type == "cuda":
                new_model = SwinMambaTex(vocab_size=VOCAB_SIZE, d_model=512).to(torch.bfloat16).cuda()
            else:
                return "Ошибка: Mamba доступна только на CUDA."
        elif arch_name == "SwinGConvTex":
            new_model = SwinGConvTex(vocab_size=VOCAB_SIZE, d_model=512).to(DEVICE, torch.bfloat16)
        elif arch_name == "SwinTransformerTex":
            new_model = SwinTransformerTex(vocab_size=VOCAB_SIZE, d_model=512).to(DEVICE, torch.bfloat16)
        else:
            return f"Ошибка: Неизвестная архитектура '{arch_name}'."
        ARCH = arch_name

        state_dict = load_file(weights_file.name)
        new_model.load_state_dict(state_dict)
        new_model.eval()
        model = new_model
        return f"Модель {arch_name} успешно загружена!\nРазмер словаря: {VOCAB_SIZE}\nУстройство: {DEVICE}"
    except Exception as e:
        return f"Ошибка загрузки весов: {str(e)}"

def process_image(img):
    if model is None or vocab is None:
        return "Ошибка: Сначала загрузите модель и словарь.", ""
    return predict_latex(img, model, DEVICE, vocab)


def get_edit_operations(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
    ins, dl, sub = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            i, j = i-1, j-1
        else:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                sub += 1; i, j = i-1, j-1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                dl += 1; i -= 1
            else:
                ins += 1; j -= 1
    return ins, dl, sub

def calc_structural_score(ref_tokens, hyp_tokens):
    ref_struct = [t for t in ref_tokens if t in STRUCTURAL_TOKENS]
    hyp_struct = [t for t in hyp_tokens if t in STRUCTURAL_TOKENS]
    if not ref_struct and not hyp_struct: return 1.0
    if not ref_struct: return 0.0
    ed = nltk.edit_distance(ref_struct, hyp_struct)
    return max(0.0, 1.0 - (ed / len(ref_struct)))

def ast_match_score(gt, pred, gt_tokens, pred_tokens):
    if not SYMPY_AVAILABLE:
        return calc_structural_score(gt_tokens, pred_tokens)
    try:
        ast_ref = parse_latex(gt)
        ast_hyp = parse_latex(pred)
        return 1.0 if str(ast_ref) == str(ast_hyp) else 0.0
    except Exception:
        return calc_structural_score(gt_tokens, pred_tokens)

BUCKET_KEYS = ["<10", "10-19", "20-29", "30-39", "40-49", "50-59", ">60"]

def _bucket_name(length):
    if length < 10: return "<10"
    if length >= 60: return ">60"
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


def run_test_dataset(max_samples, batch_size, progress=gr.Progress(track_tqdm=True)):
    if model is None or vocab_obj is None:
        raise ValueError("Загрузите модель и словарь перед запуском.")
        
    max_samples, batch_size = int(max_samples), int(batch_size)
    split_name = "test" if max_samples <= 0 else f"test[:{max_samples}]"
    ds = load_dataset("deepcopy/MathWriting-Human", split=split_name)

    all_pred_texts, all_gt_texts, items = [], [], []
    bucket_raw = {k: [] for k in BUCKET_KEYS}
    
    total_inference_time = 0.0 
    total_generated_tokens = 0

    for i in progress.tqdm(range(0, len(ds), batch_size), desc="Тестирование"):
        batch = ds[i : i + batch_size]
        images = [image_transform(img) for img in batch["image"]]
        
        image_tensor = torch.stack(images).to(DEVICE, torch.bfloat16)

        t0 = time.time()
        with torch.no_grad():
            predictions = model.generate_beam_search(
                images=image_tensor, start_token_id=1, eos_token_id=2, beam_size=5, max_new_tokens=256)
        batch_time = time.time() - t0
        total_inference_time += batch_time

        gt_texts = batch["latex"]
        for local_idx in range(predictions.shape[0]):
            pred_seq = predictions[local_idx][0].tolist()
            num_tokens = pred_seq.index(2) + 1 if 2 in pred_seq else len(pred_seq)
            total_generated_tokens += num_tokens
            sample_time = batch_time / len(gt_texts) 

            pred_text = decode_tokens(vocab, predictions[local_idx][0])
            gt = gt_texts[local_idx]
            
            gt_t = tokenize_latex(gt)
            pred_t = tokenize_latex(pred_text)
            gt_len = len(gt_t)
            
            p_tensor = encode_batch([pred_text], vocab_obj)
            t_tensor = encode_batch([gt], vocab_obj)
            
            ed = avg_edit_distance(p_tensor, t_tensor)
            em = exact_match(p_tensor, t_tensor)
            
            if gt_len > 0:
                smooth_fn = SmoothingFunction().method4
                if gt_len == 1: w = (1.0, 0, 0, 0)
                elif gt_len == 2: w = (0.5, 0.5, 0, 0)
                elif gt_len == 3: w = (0.33, 0.33, 0.33, 0)
                else: w = (0.25, 0.25, 0.25, 0.25)
                bleu_val = sentence_bleu([gt_t], pred_t, weights=w, smoothing_function=smooth_fn)
            else:
                bleu_val = 0.0
            
            ins, dl, sub = get_edit_operations(gt_t, pred_t)
            token_ed = ins + dl + sub
            token_cer = token_ed / max(1, gt_len)
            
            struct_score = calc_structural_score(gt_t, pred_t)
            ast_score = ast_match_score(gt, pred_text, gt_t, pred_t)

            bucket = _bucket_name(gt_len)
            bucket_raw[bucket].append({
                "exact_match": em, "edit_distance": ed, "bleu": bleu_val,
                "ins": ins, "del": dl, "sub": sub, "struct_score": ast_score,
                "tokens": num_tokens, "time": sample_time,
                "token_ed": token_ed, "token_cer": token_cer
            })

            items.append({
                "index": i + local_idx, "ground_truth": gt, "prediction": pred_text,
                "gt_length": gt_len, "pred_length": len(pred_t), "exact_match": float(em),
                "ins": ins, "del": dl, "sub": sub, "ast_struct_score": float(ast_score),
                "latency_sec": sample_time,
                "bleu": float(bleu_val), "token_ed": int(token_ed), "token_cer": float(token_cer)
            })
            all_pred_texts.append(pred_text)
            all_gt_texts.append(gt)

    pred_tensor = encode_batch(all_pred_texts, vocab_obj)
    tgt_tensor = encode_batch(all_gt_texts, vocab_obj)
    token_acc = token_accuracy(pred_tensor, tgt_tensor)
    
    throughput_tps = total_generated_tokens / total_inference_time if total_inference_time > 0 else 0
    latency_per_token_ms = (total_inference_time / total_generated_tokens * 1000) if total_generated_tokens > 0 else 0

    bucket_stats = {}
    avg_ast = 0.0
    for bucket in BUCKET_KEYS:
        vals = bucket_raw[bucket]
        if not vals:
            bucket_stats[bucket] = {"sample_count": 0}
        else:
            bc_len = len(vals)
            t_toks = sum(v["tokens"] for v in vals)
            t_time = sum(v["time"] for v in vals)
            
            bucket_stats[bucket] = {
                "sample_count": bc_len,
                "exact_match": sum(v["exact_match"] for v in vals) / bc_len,
                "avg_bleu": sum(v["bleu"] for v in vals) / bc_len,
                "avg_ins": sum(v["ins"] for v in vals) / bc_len,
                "avg_del": sum(v["del"] for v in vals) / bc_len,
                "avg_sub": sum(v["sub"] for v in vals) / bc_len,
                "avg_struct_score": sum(v["struct_score"] for v in vals) / bc_len,
                "avg_token_ed": sum(v["token_ed"] for v in vals) / bc_len,
                "avg_token_cer": sum(v["token_cer"] for v in vals) / bc_len,
                "tokens_per_sec": t_toks / t_time if t_time > 0 else 0
            }
            avg_ast += sum(v["struct_score"] for v in vals)
            
    avg_ast /= max(len(items), 1)

    log_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "samples": len(items), "token_accuracy": float(token_acc),
            "exact_match": sum(i["exact_match"] for i in items) / max(len(items), 1),
            "avg_ast_score": float(avg_ast),
            "bleu": sum(i["bleu"] for i in items) / max(len(items), 1),
            "avg_cer": sum(i["token_cer"] for i in items) / max(len(items), 1),
            "avg_token_ed": sum(i["token_ed"] for i in items) / max(len(items), 1),
            "throughput_tokens_per_sec": throughput_tps, "latency_per_token_ms": latency_per_token_ms
        },
        "bucket_stats": bucket_stats,
        "items": items,
    }

    fd, out_path = tempfile.mkstemp(prefix="test_logs_", suffix=".json")
    os.close(fd)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    return _prepare_gradio_data(log_data) + (out_path,)

def load_test_logs(file_obj):
    if file_obj is None: 
        empty = apply_plot_theme(go.Figure())
        return pd.DataFrame(), empty, empty, empty, empty, empty, empty, empty, empty
    with open(file_obj.name, "r", encoding="utf-8") as f:
        log_data = json.load(f)
    return _prepare_gradio_data(log_data)

def load_training_graphs(epoch_file, step_file):
    fig_loss = go.Figure(layout=go.Layout(title="Потери (Loss)"))
    fig_ed = go.Figure(layout=go.Layout(title="Расстояние редактирования (Edit Distance)"))
    fig_acc = go.Figure(layout=go.Layout(title="Точность (Sequence Accuracy)"))
    fig_lr = go.Figure(layout=go.Layout(title="Скорость обучения (LR)"))
    fig_step = go.Figure(layout=go.Layout(title="Потери по шагам (Step Loss)"))
    if epoch_file is not None:
        try:
            df_e = pd.read_csv(epoch_file.name)
            if all(col in df_e.columns for col in ['epoch', 'train_loss', 'val_loss']):
                fig_loss = px.line(df_e, x='epoch', y=['train_loss', 'val_loss'], markers=True, title="Обучение vs Валидация (Loss)")
            if all(col in df_e.columns for col in ['epoch', 'edit_distance', 'norm_edit_distance']):
                fig_ed = px.line(df_e, x='epoch', y=['edit_distance', 'norm_edit_distance'], markers=True, title="Edit Distance")
            if 'sequence_accuracy' in df_e.columns:
                fig_acc = px.line(df_e, x='epoch', y='sequence_accuracy', markers=True, title="Точность (Sequence Accuracy)")
            if 'lr' in df_e.columns:
                fig_lr = px.line(df_e, x='epoch', y='lr', markers=True, title="Скорость обучения (Learning Rate)")
        except Exception as e:
            print(f"Ошибка чтения epoch файла: {e}")

    if step_file is not None:
        try:
            df_s = pd.read_csv(step_file.name)
            if 'step' in df_s.columns and 'loss' in df_s.columns:
                fig_step = px.line(df_s, x='step', y='loss', title="Потери на обучении (по шагам)")
        except Exception as e:
            print(f"Ошибка чтения step файла: {e}")

    return apply_plot_theme(fig_loss), apply_plot_theme(fig_ed), apply_plot_theme(fig_acc), apply_plot_theme(fig_lr), apply_plot_theme(fig_step)

def compare_training_logs(files):
    if not files:
        empty = apply_plot_theme(go.Figure())
        return empty, empty, empty, empty

    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f.name)
            model_name = os.path.basename(f.name).replace(".csv", "")
            df['Модель'] = model_name
            df_list.append(df)
        except Exception as e:
            print(f"Ошибка чтения {f.name}: {e}")

    if not df_list:
        empty = apply_plot_theme(go.Figure())
        return empty, empty, empty, empty

    combined_df = pd.concat(df_list, ignore_index=True)
    cols = combined_df.columns

    fig_tloss = px.line(combined_df, x='epoch', y='train_loss', color='Модель', markers=True, title='Train Loss') if 'train_loss' in cols else go.Figure(layout=go.Layout(title="Train Loss (нет данных)"))
    fig_vloss = px.line(combined_df, x='epoch', y='val_loss', color='Модель', markers=True, title='Validation Loss') if 'val_loss' in cols else go.Figure(layout=go.Layout(title="Validation Loss (нет данных)"))
    
    ed_col = 'edit_distance' if 'edit_distance' in cols else ('norm_edit_distance' if 'norm_edit_distance' in cols else None)
    fig_ed = px.line(combined_df, x='epoch', y=ed_col, color='Модель', markers=True, title='Edit Distance') if ed_col else go.Figure(layout=go.Layout(title="Edit Distance (нет данных)"))
    
    fig_acc = px.line(combined_df, x='epoch', y='sequence_accuracy', color='Модель', markers=True, title='Sequence Accuracy') if 'sequence_accuracy' in cols else go.Figure(layout=go.Layout(title="Sequence Accuracy (нет данных)"))

    return apply_plot_theme(fig_tloss), apply_plot_theme(fig_vloss), apply_plot_theme(fig_ed), apply_plot_theme(fig_acc)

def compare_inference_models(files):
    if not files:
        empty_fig = apply_plot_theme(go.Figure())
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    
    global_records = []
    bucket_records = []

    for f in files:
        try:
            with open(f.name, "r", encoding="utf-8") as file:
                data = json.load(file)

            model_name = os.path.basename(f.name).replace(".json", "")

            metrics = data.get("metrics", {})
            global_records.append({
                "Модель": model_name,
                "Exact Match": metrics.get("exact_match", 0),
                "BLEU Score": metrics.get("bleu", 0),
                "Точность токенов": metrics.get("token_accuracy", 0),
                "AST/Struct Score": metrics.get("avg_ast_score", 0),
                "Token CER": metrics.get("avg_cer", 0),
                "Token ED": metrics.get("avg_token_ed", 0),
                "Скорость (Токены/сек)": metrics.get("throughput_tokens_per_sec", metrics.get("inference_samples_per_sec", 0)),
                "Задержка (мс/токен)": metrics.get("latency_per_token_ms", 0)
            })

            bucket_stats = data.get("bucket_stats", {})
            for bucket, stats in bucket_stats.items():
                bucket_records.append({
                    "Модель": model_name,
                    "Длина": bucket,
                    "Exact Match": stats.get("exact_match", 0),
                    "BLEU Score": stats.get("avg_bleu", 0),
                    "Structural Score": stats.get("avg_struct_score", 0),
                    "Token CER": stats.get("avg_token_cer", 0),
                    "Token ED": stats.get("avg_token_ed", 0),
                    "Скорость (Токены/сек)": stats.get("tokens_per_sec", 0)
                })
        except Exception as e:
            print(f"Ошибка обработки {f.name}: {e}")

    df_global = pd.DataFrame(global_records)
    df_bucket = pd.DataFrame(bucket_records)

    if not df_bucket.empty:
        df_bucket['Длина'] = pd.Categorical(df_bucket['Длина'], categories=BUCKET_KEYS, ordered=True)
        df_bucket = df_bucket.sort_values('Длина')

    df_global_acc = df_global.melt(id_vars=["Модель"], value_vars=["Exact Match", "BLEU Score", "Точность токенов", "AST/Struct Score"], var_name="Метрика", value_name="Значение")
    fig_global_acc = px.bar(df_global_acc, x="Модель", y="Значение", color="Метрика", barmode="group", text_auto='.3f', title="Глобальные метрики: Точность (Больше - лучше)")

    df_global_err = df_global.melt(id_vars=["Модель"], value_vars=["Token CER", "Token ED"], var_name="Метрика", value_name="Значение")
    fig_global_err = px.bar(df_global_err, x="Модель", y="Значение", color="Метрика", barmode="group", text_auto='.3f', title="Глобальные метрики: Ошибки (Меньше - лучше)")

    df_global_perf = df_global.melt(id_vars=["Модель"], value_vars=["Скорость (Токены/сек)", "Задержка (мс/токен)"], var_name="Метрика", value_name="Значение")
    fig_global_perf = px.bar(df_global_perf, x="Модель", y="Значение", color="Метрика", barmode="group", text_auto='.2f', title="Глобальные метрики: Производительность")

    fig_bucket_em = px.line(df_bucket, x="Длина", y="Exact Match", color="Модель", markers=True, title="Сравнение Exact Match по длине")
    fig_bucket_bleu = px.line(df_bucket, x="Длина", y="BLEU Score", color="Модель", markers=True, title="Сравнение BLEU Score по длине")
    fig_bucket_struct = px.line(df_bucket, x="Длина", y="Structural Score", color="Модель", markers=True, title="Сравнение Structural/AST Score по длине")
    fig_bucket_cer = px.line(df_bucket, x="Длина", y="Token CER", color="Модель", markers=True, title="Сравнение Token CER по длине (Меньше - лучше)")
    fig_bucket_ed = px.line(df_bucket, x="Длина", y="Token ED", color="Модель", markers=True, title="Сравнение Token Edit Distance по длине")
    fig_bucket_thr = px.line(df_bucket, x="Длина", y="Скорость (Токены/сек)", color="Модель", markers=True, title="Пропускная способность (Токены/сек) по длине")

    return apply_plot_theme(fig_global_acc), apply_plot_theme(fig_global_err), apply_plot_theme(fig_global_perf), apply_plot_theme(fig_bucket_em), apply_plot_theme(fig_bucket_bleu), apply_plot_theme(fig_bucket_struct), apply_plot_theme(fig_bucket_cer), apply_plot_theme(fig_bucket_ed), apply_plot_theme(fig_bucket_thr)

def get_svg_template(title, block_name, extra_shapes=""):
    return f"""
        <div style="width: 80%; max-width: 1400px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden; margin: auto;">
            {extra_shapes}
        </div>
    """

def get_mamba_svg():
    return get_svg_template("Архетектура SwinMambaTex", "Mamba", 
        '''
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 650" style="background-color:#fafafa;font-family:&quot;Times New Roman&quot;,serif"><rect x="40" y="100" width="100" height="100" rx="4" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><rect x="45" y="105" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="105" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="105" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="105" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="127.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="150" width="20" height="20" rx="2" fill="#2c1e16" stroke="#2c1e16" stroke-width="1.5"/><rect x="90" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="172.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="172.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="172.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="172.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><text x="90" y="225" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Image Input</text><text x="90" y="245" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 1, 384, 384]</text><rect x="65" y="340" width="50" height="200" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="90" cy="360" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="385" r="8" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="410" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="485" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="510" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="90" y="445" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="455" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="465" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="565" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Target Tokens</text><text x="90" y="585" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100]</text><defs><marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path fill="#3d2b1f" d="m0 0 8 3-8 3z"/></marker></defs><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M140 150h70"/><rect x="210" y="80" width="160" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><rect x="230" y="110" width="120" height="40" rx="4" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="240" y="160" width="100" height="40" rx="4" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="290" y="135" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Swin Backbone</text><text x="290" y="185" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Pos2D Encode</text><text x="290" y="70" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">ENCODER</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M370 150h60"/><rect x="430" y="80" width="80" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="445" cy="95" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="460" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="115" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="445" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="135" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="490" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="155" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="475" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="175" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="460" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="195" r="4" fill="#d4af37" stroke="#3d2b1f"/><text x="470" y="245" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Memory (K, V)</text><text x="470" y="265" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 576, 512]</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M115 440h95"/><rect x="210" y="370" width="160" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="235" cy="385" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 385h35"/><rect x="285" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="410" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 410h35"/><rect x="285" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="405" width="10" height="10" rx="8" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="435" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 435h35"/><rect x="285" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="460" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 460h35"/><rect x="285" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="485" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 485h35"/><rect x="285" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="290" y="360" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">EMBEDDING</text><text x="290" y="535" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Token Embeds</text><text x="290" y="555" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100, 512]</text><path stroke="#3d2b1f" stroke-width="2.5" d="M370 440h160m0 0V140"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M530 140h100"/><rect x="610" y="40" width="460" height="530" rx="12" fill="#fff" stroke="#3d2b1f" stroke-width="3"/><text x="840" y="595" font-size="18" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">MAMBA DECODER (× Depth)</text><rect x="630" y="80" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="650" y="115" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="140" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Mamba2</text><text x="685" y="155" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(SSM)</text><rect x="770" y="100" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="120" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 130h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 130h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 142h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 142h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 142h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 154h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 154h12v12h-12z"/><rect x="940" y="115" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="140" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="155" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M720 140h50"/><text x="745" y="132" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 140h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 185h390"/><rect x="630" y="250" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="650" y="285" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="310" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Mamba2</text><text x="685" y="325" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(SSM)</text><rect x="770" y="270" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="290" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 300h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 300h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 312h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 312h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 312h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 324h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 324h12v12h-12z"/><rect x="940" y="285" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="310" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="325" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M720 310h50"/><text x="745" y="302" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 310h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 355h390"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M840 200v50"/><rect x="630" y="420" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="650" y="455" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="480" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Mamba2</text><text x="685" y="495" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(SSM)</text><rect x="770" y="440" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="460" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 470h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 470h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 482h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 482h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 482h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 494h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 494h12v12h-12z"/><rect x="940" y="455" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="480" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="495" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M720 480h50"/><text x="745" y="472" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 480h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 525h390"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M840 370v50"/><path stroke="#d4af37" stroke-width="3" d="M510 150h50m0-45v340"/><defs><marker id="b" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path fill="#d4af37" d="m0 0 8 3-8 3z"/></marker></defs><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 105h210"/><text x="665" y="97" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#d4af37" text-anchor="middle">K, V</text><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 275h210"/><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 445h210"/><path stroke="#3d2b1f" stroke-width="2.5" d="M1010 480h50m0 0V300"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M1060 300h80"/><text x="1100" y="290" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">LINEAR</text><rect x="1140" y="160" width="120" height="280" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="1160" y="315" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="315" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1160" y="330" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="330" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1160" y="345" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="345" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><path stroke="#3d2b1f" d="m1160 200 80-20m-80 20 80 10m-80-10 80 40m-80-40 80 70m-80-70 80 100m-80-100 80 180m-80-180 80 210m-80-170 80-60m-80 60 80-30m-80 30h80m-80 0 80 30m-80-30 80 60m-80-60 80 140m-80-140 80 170m-80-130 80-100m-80 100 80-70m-80 70 80-40m-80 40 80-10m-80 10 80 20m-80-20 80 100m-80-100 80 130m-80-50 80-180m-80 180 80-150m-80 150 80-120m-80 120 80-90m-80 90 80-60m-80 60 80 20m-80-20 80 50m-80-10 80-220m-80 220 80-190m-80 190 80-160m-80 160 80-130m-80 130 80-100m-80 100 80-20m-80 20 80 10"/><circle cx="1160" cy="200" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="240" r="7" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="280" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="360" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="400" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="180" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="210" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="240" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="270" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="300" r="7" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="380" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="410" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="1200" y="145" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">UNEMBED</text><text x="1200" y="470" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Logits</text><text x="1200" y="490" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100, 256]</text></svg>
        ''')

def get_transformer_svg():
    return get_svg_template("Архетектура SwinTransformerTex", "Attention", 
        '''
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 650" style="background-color:#fafafa;font-family:&quot;Times New Roman&quot;,serif"><rect x="40" y="100" width="100" height="100" rx="4" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><rect x="45" y="105" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="105" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="105" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="105" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="127.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="150" width="20" height="20" rx="2" fill="#2c1e16" stroke="#2c1e16" stroke-width="1.5"/><rect x="90" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="172.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="172.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="172.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="172.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><text x="90" y="225" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Image Input</text><text x="90" y="245" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 1, 384, 384]</text><rect x="65" y="340" width="50" height="200" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="90" cy="360" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="385" r="8" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="410" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="485" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="510" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="90" y="445" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="455" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="465" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="565" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Target Tokens</text><text x="90" y="585" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100]</text><defs><marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path fill="#3d2b1f" d="m0 0 8 3-8 3z"/></marker></defs><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M140 150h70"/><rect x="210" y="80" width="160" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><rect x="230" y="110" width="120" height="40" rx="4" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="240" y="160" width="100" height="40" rx="4" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="290" y="135" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Swin Backbone</text><text x="290" y="185" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Pos2D Encode</text><text x="290" y="70" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">ENCODER</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M370 150h60"/><rect x="430" y="80" width="80" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="445" cy="95" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="460" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="115" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="445" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="135" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="490" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="155" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="475" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="175" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="460" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="195" r="4" fill="#d4af37" stroke="#3d2b1f"/><text x="470" y="245" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Memory (K, V)</text><text x="470" y="265" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 576, 512]</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M115 440h95"/><rect x="210" y="370" width="160" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="235" cy="385" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 385h35"/><rect x="285" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="410" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 410h35"/><rect x="285" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="405" width="10" height="10" rx="8" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="435" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 435h35"/><rect x="285" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="460" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 460h35"/><rect x="285" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="485" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 485h35"/><rect x="285" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="290" y="360" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">EMBEDDING</text><text x="290" y="535" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Token Embeds</text><text x="290" y="555" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100, 512]</text><path stroke="#3d2b1f" stroke-width="2.5" d="M370 440h160m0 0V140"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M530 140h100"/><rect x="610" y="40" width="460" height="530" rx="12" fill="#fff" stroke="#3d2b1f" stroke-width="3"/><text x="840" y="595" font-size="18" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">TRANSFORMER DECODER (× Depth)</text><rect x="630" y="80" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="640" y="115" width="90" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="140" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Self-Attn</text><text x="685" y="155" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(Causal)</text><rect x="770" y="100" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="120" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 130h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 130h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 142h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 142h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 142h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 154h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 154h12v12h-12z"/><rect x="940" y="115" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="140" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="155" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M730 140h40"/><text x="750" y="132" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 140h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 185h390"/><rect x="630" y="250" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="640" y="285" width="90" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="310" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Self-Attn</text><text x="685" y="325" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(Causal)</text><rect x="770" y="270" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="290" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 300h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 300h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 312h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 312h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 312h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 324h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 324h12v12h-12z"/><rect x="940" y="285" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="310" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="325" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M730 310h40"/><text x="750" y="302" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 310h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 355h390"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M840 200v50"/><rect x="630" y="420" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="640" y="455" width="90" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="480" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Self-Attn</text><text x="685" y="495" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(Causal)</text><rect x="770" y="440" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="460" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 470h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 470h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 482h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 482h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 482h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 494h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 494h12v12h-12z"/><rect x="940" y="455" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="480" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="495" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M730 480h40"/><text x="750" y="472" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 480h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 525h390"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M840 370v50"/><path stroke="#d4af37" stroke-width="3" d="M510 150h50m0-45v340"/><defs><marker id="b" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path fill="#d4af37" d="m0 0 8 3-8 3z"/></marker></defs><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 105h210"/><text x="665" y="97" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#d4af37" text-anchor="middle">K, V</text><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 275h210"/><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 445h210"/><path stroke="#3d2b1f" stroke-width="2.5" d="M1010 480h50m0 0V300"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M1060 300h80"/><text x="1100" y="290" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">LINEAR</text><rect x="1140" y="160" width="120" height="280" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="1160" y="315" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="315" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1160" y="330" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="330" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1160" y="345" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="345" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><path stroke="#3d2b1f" d="m1160 200 80-20m-80 20 80 10m-80-10 80 40m-80-40 80 70m-80-70 80 100m-80-100 80 180m-80-180 80 210m-80-170 80-60m-80 60 80-30m-80 30h80m-80 0 80 30m-80-30 80 60m-80-60 80 140m-80-140 80 170m-80-130 80-100m-80 100 80-70m-80 70 80-40m-80 40 80-10m-80 10 80 20m-80-20 80 100m-80-100 80 130m-80-50 80-180m-80 180 80-150m-80 150 80-120m-80 120 80-90m-80 90 80-60m-80 60 80 20m-80-20 80 50m-80-10 80-220m-80 220 80-190m-80 190 80-160m-80 160 80-130m-80 130 80-100m-80 100 80-20m-80 20 80 10"/><circle cx="1160" cy="200" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="240" r="7" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="280" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="360" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="400" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="180" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="210" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="240" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="270" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="300" r="7" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="380" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="410" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="1200" y="145" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">UNEMBED</text><text x="1200" y="470" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Logits</text><text x="1200" y="490" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100, 256]</text></svg>
        ''')

def get_gconv_svg():
    return get_svg_template("Архетектура SwinGConvTex", "GConv", 
        '''
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 650" style="background-color:#fafafa;font-family:&quot;Times New Roman&quot;,serif"><rect x="40" y="100" width="100" height="100" rx="4" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><rect x="45" y="105" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="105" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="105" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="105" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="127.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="127.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="150" width="20" height="20" rx="2" fill="#2c1e16" stroke="#2c1e16" stroke-width="1.5"/><rect x="90" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="150" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="45" y="172.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><rect x="67.5" y="172.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="90" y="172.5" width="20" height="20" rx="2" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="112.5" y="172.5" width="20" height="20" rx="2" fill="#2c1e16" stroke="#3d2b1f" stroke-width="1.5"/><text x="90" y="225" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Image Input</text><text x="90" y="245" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 1, 384, 384]</text><rect x="65" y="340" width="50" height="200" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="90" cy="360" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="385" r="8" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="410" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="485" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="90" cy="510" r="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="90" y="445" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="455" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="465" font-size="24" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="90" y="565" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Target Tokens</text><text x="90" y="585" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100]</text><defs><marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path fill="#3d2b1f" d="m0 0 8 3-8 3z"/></marker></defs><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M140 150h70"/><rect x="210" y="80" width="160" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><rect x="230" y="110" width="120" height="40" rx="4" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="240" y="160" width="100" height="40" rx="4" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="290" y="135" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Swin Backbone</text><text x="290" y="185" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Pos2D Encode</text><text x="290" y="70" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">ENCODER</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M370 150h60"/><rect x="430" y="80" width="80" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="445" cy="95" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="460" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="95" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="115" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="115" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="445" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="135" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="490" cy="135" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="155" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="475" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="155" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="175" r="4" fill="#d4af37" stroke="#3d2b1f"/><circle cx="460" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="175" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="445" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="460" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="475" cy="195" r="4" fill="#eae6df" stroke="#3d2b1f"/><circle cx="490" cy="195" r="4" fill="#d4af37" stroke="#3d2b1f"/><text x="470" y="245" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Memory (K, V)</text><text x="470" y="265" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 576, 512]</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M115 440h95"/><rect x="210" y="370" width="160" height="140" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><circle cx="235" cy="385" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 385h35"/><rect x="285" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="380" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="410" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 410h35"/><rect x="285" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="405" width="10" height="10" rx="8" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="405" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="435" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 435h35"/><rect x="285" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="430" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="460" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 460h35"/><rect x="285" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="455" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="235" cy="485" r="6" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><path stroke="#3d2b1f" stroke-width="1.5" marker-end="url(#a)" d="M245 485h35"/><rect x="285" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="305" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="325" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><rect x="345" y="480" width="10" height="10" rx="8" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="290" y="360" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">EMBEDDING</text><text x="290" y="535" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Token Embeds</text><text x="290" y="555" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100, 512]</text><path stroke="#3d2b1f" stroke-width="2.5" d="M370 440h160m0 0V140"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M530 140h100"/><rect x="610" y="40" width="460" height="530" rx="12" fill="#fff" stroke="#3d2b1f" stroke-width="3"/><text x="840" y="595" font-size="18" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">HYBRID DECODER (× Depth)</text><rect x="630" y="80" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="640" y="115" width="90" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="140" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Gated Conv</text><text x="685" y="155" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(1D)</text><rect x="770" y="100" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="120" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 130h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 130h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 142h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 142h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 142h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 154h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 154h12v12h-12z"/><rect x="940" y="115" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="140" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="155" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M730 140h40"/><text x="750" y="132" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 140h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 185h390"/><rect x="630" y="250" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="640" y="285" width="90" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="310" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Gated Conv</text><text x="685" y="325" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(1D)</text><rect x="770" y="270" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="290" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 300h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 300h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 312h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 312h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 312h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 324h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 324h12v12h-12z"/><rect x="940" y="285" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="310" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="325" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M730 310h40"/><text x="750" y="302" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 310h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 355h390"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M840 200v50"/><rect x="630" y="420" width="420" height="120" rx="8" fill="#f5f5f5" stroke="#3d2b1f" stroke-width="2"/><rect x="640" y="455" width="90" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="685" y="480" font-size="12" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Gated Conv</text><text x="685" y="495" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(1D)</text><rect x="770" y="440" width="120" height="80" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="830" y="460" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Cross-Attn</text><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M812 470h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M824 470h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 482h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M824 482h12v12h-12z"/><path fill="#fff" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M836 482h12v12h-12z"/><path fill="#d4af37" opacity=".3" stroke="#3d2b1f" stroke-width=".5" d="M812 494h12v12h-12zm12 0h12v12h-12z"/><path fill="#d4af37" opacity=".8" stroke="#3d2b1f" stroke-width=".5" d="M836 494h12v12h-12z"/><rect x="940" y="455" width="70" height="50" rx="6" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="975" y="480" font-size="13" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">FFN</text><text x="975" y="495" font-size="11" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">(MLP)</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M730 480h40"/><text x="750" y="472" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">Q</text><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M890 480h50"/><path stroke="#3d2b1f" stroke-width="2.5" stroke-dasharray="4,4" d="M640 525h390"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M840 370v50"/><path stroke="#d4af37" stroke-width="3" d="M510 150h50m0-45v340"/><defs><marker id="b" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path fill="#d4af37" d="m0 0 8 3-8 3z"/></marker></defs><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 105h210"/><text x="665" y="97" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#d4af37" text-anchor="middle">K, V</text><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 275h210"/><path stroke="#d4af37" stroke-width="2.5" marker-end="url(#b)" d="M560 445h210"/><path stroke="#3d2b1f" stroke-width="2.5" d="M1010 480h50m0 0V300"/><path stroke="#3d2b1f" stroke-width="2.5" marker-end="url(#a)" d="M1060 300h80"/><text x="1100" y="290" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#3d2b1f" text-anchor="middle">LINEAR</text><rect x="1140" y="160" width="120" height="280" rx="8" fill="#fff" stroke="#3d2b1f" stroke-width="2.5"/><text x="1160" y="315" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="315" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1160" y="330" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="330" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1160" y="345" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><text x="1240" y="345" font-size="18" font-family="Arial, sans-serif" fill="#2c1e16" text-anchor="middle">.</text><path stroke="#3d2b1f" d="m1160 200 80-20m-80 20 80 10m-80-10 80 40m-80-40 80 70m-80-70 80 100m-80-100 80 180m-80-180 80 210m-80-170 80-60m-80 60 80-30m-80 30h80m-80 0 80 30m-80-30 80 60m-80-60 80 140m-80-140 80 170m-80-130 80-100m-80 100 80-70m-80 70 80-40m-80 40 80-10m-80 10 80 20m-80-20 80 100m-80-100 80 130m-80-50 80-180m-80 180 80-150m-80 150 80-120m-80 120 80-90m-80 90 80-60m-80 60 80 20m-80-20 80 50m-80-10 80-220m-80 220 80-190m-80 190 80-160m-80 160 80-130m-80 130 80-100m-80 100 80-20m-80 20 80 10"/><circle cx="1160" cy="200" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="240" r="7" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="280" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="360" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1160" cy="400" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="180" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="210" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="240" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="270" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="300" r="7" fill="#d4af37" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="380" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><circle cx="1240" cy="410" r="7" fill="#eae6df" stroke="#3d2b1f" stroke-width="1.5"/><text x="1200" y="145" font-size="16" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">UNEMBED</text><text x="1200" y="470" font-size="14" font-family="Arial, sans-serif" font-weight="bold" fill="#2c1e16" text-anchor="middle">Logits</text><text x="1200" y="490" font-size="14" font-family="Courier" fill="#2c1e16" text-anchor="middle">[1, 100, 256]</text></svg>
        ''')

def update_model_svg(arch_name):
    if arch_name == "SwinTransformerTex": return get_transformer_svg()
    elif arch_name == "SwinGConvTex": return get_gconv_svg()
    return get_mamba_svg()

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


        vocab = ckpt["vocab"].itos

        os.makedirs("outputs", exist_ok=True)

        safetensor_path = f"outputs/{prefix}_bf16.safetensors"
        vocab_path = "outputs/vocab.json"

        save_file(state_dict_bf16, safetensor_path)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        return safetensor_path, vocab_path, "Conversion successful"
    except Exception as e:
        return None, None, f"Error: {str(e)}"
    
def run_summary(B, T):
    global ARCH, VOCAB_SIZE
    if model is None or vocab_obj is None:
        raise ValueError("Загрузите модель и словарь перед запуском.")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = summary(
            model,
            input_size=[(B, 1, 384, 384), (B, T)],
            dtypes=[torch.bfloat16, torch.long],
            device=device,
            verbose=0  # cleaner output
        )

        return str(result), ARCH, VOCAB_SIZE

    except Exception as e:
        return f"Error: {str(e)}"


theme = gr.themes.Ocean(
    primary_hue="orange",
    secondary_hue="yellow",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Nunito'), gr.themes.GoogleFont('IBM Plex Sans'), 'system-ui', 'IBM Plex Sans'],
)

with gr.Blocks() as demo:
    gr.Markdown("# Переводчик Математики в LaTeX")

    with gr.Tabs():
        with gr.TabItem("Конвертер"):
            with gr.Accordion("Настройки и загрузка модели", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        arch_dropdown = gr.Dropdown(choices=["SwinMambaTex", "SwinTransformerTex", "SwinGConvTex"], label="Архитектура", value="SwinMambaTex")
                        load_model_btn = gr.Button("Загрузить модель", variant="primary")
                    with gr.Column(scale=2):
                        upload_vocab = gr.File(label="vocab.json", file_types=[".json"])
                        upload_weights = gr.File(label="Веса (.safetensors)", file_types=[".safetensors"])
                with gr.Row():
                    with gr.Column():
                        model_status = gr.Textbox(label="Статус", interactive=False)
                        model_svg_display = gr.HTML(value=get_mamba_svg())
                
                arch_dropdown.change(fn=update_model_svg, inputs=[arch_dropdown], outputs=[model_svg_display])
                load_model_btn.click(fn=load_custom_model, inputs=[arch_dropdown, upload_vocab, upload_weights], outputs=[model_status])

            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Изображение формулы")
                    submit_btn = gr.Button("Конвертировать", variant="primary")
                with gr.Column():
                    output_code = gr.Textbox(label="LaTeX Код", lines=4)
                    output_render = gr.Markdown(label="Рендер")
            submit_btn.click(fn=process_image, inputs=[input_img], outputs=[output_code, output_render])

        with gr.TabItem("Статистика датасета"):
            gr.Markdown("Запуск тестирования с подсчетом структурных метрик (AST/SymPy), декомпозиции ошибок и производительности.")
            with gr.Row():
                max_samples = gr.Number(value=50, precision=0, label="Макс. примеров (0 = все)")
                batch_size = gr.Number(value=8, precision=0, label="Размер батча")
            run_btn = gr.Button("Запустить", variant="primary")

            summary_table = gr.Dataframe(label="Глобальные метрики", interactive=False)
            
            with gr.Row():
                plot_samples = gr.Plot(label="Распределение")
                plot_em = gr.Plot(label="Exact Match")
            with gr.Row():
                plot_errs = gr.Plot(label="Error Decomp (Ins/Del/Sub)")
                plot_struct = gr.Plot(label="Structural/AST Score")
            with gr.Row():
                plot_cer = gr.Plot(label="Token CER (TER)")
                plot_token_ed = gr.Plot(label="Token Edit Distance")
            with gr.Row():
                plot_bleu = gr.Plot(label="BLEU Score")
                plot_thr = gr.Plot(label="Throughput (Tokens/s)")
            
            download_logs = gr.File(label="Скачать JSON логи")

            run_btn.click(
                fn=run_test_dataset,
                inputs=[max_samples, batch_size],
                outputs=[summary_table, plot_samples, plot_em, plot_errs, plot_struct, plot_cer, plot_token_ed, plot_bleu, plot_thr, download_logs],
            )

            gr.Markdown("### Загрузить существующий лог")
            upload_logs = gr.File(file_types=[".json"], label="Загрузить JSON")
            load_btn = gr.Button("Показать графики")
            load_btn.click(
                fn=load_test_logs,
                inputs=[upload_logs],
                outputs=[summary_table, plot_samples, plot_em, plot_errs, plot_struct, plot_cer, plot_token_ed, plot_bleu, plot_thr],
            )
        
        with gr.TabItem("Графики обучения"):
            gr.Markdown("Загрузите CSV файлы логов (формат: `epoch, train_loss, val_loss, edit_distance, norm_edit_distance, sequence_accuracy, lr`)")

            with gr.Row():
                upload_epoch_csv = gr.File(label="Логи по Эпохам (Epoch Logs)", file_types=[".csv"])
                upload_step_csv = gr.File(label="Логи по Шагам (Step Logs)", file_types=[".csv"])

            plot_graphs_btn = gr.Button("Построить графики обучения", variant="primary")

            with gr.Row():
                epoch_loss_plot = gr.Plot(label="Потери (Loss)")
                epoch_ed_plot = gr.Plot(label="Расстояние редактирования (Edit Distance)")
            with gr.Row():
                epoch_acc_plot = gr.Plot(label="Точность последовательности (Seq Accuracy)")
                epoch_lr_plot = gr.Plot(label="Скорость обучения (LR)")
            with gr.Row():
                step_loss_plot = gr.Plot(label="Потери по шагам")

            plot_graphs_btn.click(
                fn=load_training_graphs,
                inputs=[upload_epoch_csv, upload_step_csv],
                outputs=[epoch_loss_plot, epoch_ed_plot, epoch_acc_plot, epoch_lr_plot, step_loss_plot]
            )

        with gr.TabItem("Сравнение моделей (Обучение)"):
            gr.Markdown("Загрузите несколько **CSV файлов**, чтобы визуально сравнить графики обучения (наложение друг на друга).")
            upload_multi_csv = gr.File(label="Выберите несколько CSV файлов (Логи по Эпохам)", file_types=[".csv"], file_count="multiple")
            compare_train_btn = gr.Button("Построить графики сравнения обучения", variant="primary")

            with gr.Row():
                comp_train_loss = gr.Plot(label="Train Loss")
                comp_val_loss = gr.Plot(label="Validation Loss")
            with gr.Row():
                comp_train_ed = gr.Plot(label="Edit Distance")
                comp_train_acc = gr.Plot(label="Sequence Accuracy")

            compare_train_btn.click(
                fn=compare_training_logs,
                inputs=[upload_multi_csv],
                outputs=[comp_train_loss, comp_val_loss, comp_train_ed, comp_train_acc]
            )
        

        with gr.TabItem("Сравнение моделей (Инференс)"):
            gr.Markdown("Загрузите несколько **JSON файлов**, полученных во вкладке `Статистика датасета`, чтобы визуально сравнить результаты нескольких моделей по всем новым метрикам.")

            upload_multi_json = gr.File(label="Выберите несколько JSON файлов (Логи статистики)", file_types=[".json"], file_count="multiple")
            compare_btn = gr.Button("Построить графики сравнения моделей", variant="primary")

            gr.Markdown("### Глобальные метрики")
            with gr.Row():
                comp_global_acc = gr.Plot(label="Точность")
                comp_global_err = gr.Plot(label="Ошибки")
            with gr.Row():
                comp_global_perf = gr.Plot(label="Производительность")

            gr.Markdown("### Детальные метрики по бакетам (по длине формул)")
            with gr.Row():
                comp_bucket_em = gr.Plot(label="Exact Match")
                comp_bucket_bleu = gr.Plot(label="BLEU Score")
            with gr.Row():
                comp_bucket_struct = gr.Plot(label="Structural/AST Score")
                comp_bucket_cer = gr.Plot(label="Token CER")
            with gr.Row():
                comp_bucket_ed = gr.Plot(label="Token ED")
                comp_bucket_thr = gr.Plot(label="Throughput")

            compare_btn.click(
                fn=compare_inference_models,
                inputs=[upload_multi_json],
                outputs=[
                    comp_global_acc, comp_global_err, comp_global_perf, 
                    comp_bucket_em, comp_bucket_bleu, comp_bucket_struct,
                    comp_bucket_cer, comp_bucket_ed, comp_bucket_thr
                ]
            )
        
        with gr.TabItem("Вспомогательные утилиты"):
            with gr.Tabs():
                with gr.TabItem(".pt to bf16.safetensor"):
                    gr.Markdown("# Checkpoint → bf16 → SafeTensors Converter")

                    file_input = gr.File(label="Upload .pt checkpoint")
                    name_prefix_input = gr.Textbox(label="model prefix to save", value="model")
                    convert_btn = gr.Button("Convert")
                    output_safetensor = gr.File(label="Download SafeTensors")
                    output_vocab = gr.File(label="Download vocab.json")
                    status = gr.Textbox(label="Status")
                    convert_btn.click(
                        fn=convert_checkpoint,
                        inputs=[file_input,name_prefix_input],
                        outputs=[output_safetensor, output_vocab, status]
                    )


                with gr.TabItem("Model Summary"):
                    gr.Markdown("# Model Architecture Summary (torchinfo)")
                    with gr.Row():
                        model_choice = gr.Textbox(label="Model name")
                        vocab_size = gr.Number(label="Vocab size")

                    with gr.Row():
                        B = gr.Number(value=1, label="Batch size (B)")
                        T = gr.Number(value=100, label="Sequence length (T)")

                    run_btn = gr.Button("Run Summary")

                    output = gr.Textbox(label="Summary Output", lines=30)

                    run_btn.click(
                        fn=run_summary,
                        inputs=[B, T],
                        outputs=[output, model_choice, vocab_size]
                    )

if __name__ == "__main__":
    demo.launch(share=False, theme=theme)