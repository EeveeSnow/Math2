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

# Попытка импорта SymPy для AST-метрик
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
    global model, vocab, vocab_obj, VOCAB_SIZE
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


# --- Новые метрики: Decomp, Structural, AST ---

def get_edit_operations(ref_tokens, hyp_tokens):
    """Декомпозиция Edit Distance на Ins, Del, Sub"""
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
        # Fallback при ошибке парсинга
        return calc_structural_score(gt_tokens, pred_tokens)

# ОБНОВЛЕННЫЕ БАКЕТЫ
BUCKET_KEYS = ["<10", "10-19", "20-29", "30-39", "40-49", "50-59", ">60"]

def _bucket_name(length):
    if length < 10: return "<10"
    if length >= 60: return ">60"
    lower = (length // 10) * 10
    return f"{lower}-{lower+9}"

def tokenize_latex(s):
    return re.findall(r'\\[a-zA-Z]+|.', str(s).replace(" ", ""))

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
    
    # Использование ключей из логов (совместимость со старыми логами)
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
    
    # Графики
    fig_samples = px.bar(bucket_df, x="Длина", y="Samples", text_auto=True, title="Распределение длин")
    fig_em = px.bar(bucket_df, x="Длина", y="Exact Match", text_auto='.3f', title="Exact Match vs Длина")
    
    fig_errs = px.bar(bucket_df, x="Длина", y=["Ins", "Del", "Sub"], title="Декомпозиция ошибок (Ins/Del/Sub) vs Длина", barmode='stack')
    fig_struct = px.line(bucket_df, x="Длина", y="Structural Score", markers=True, title="Structural/AST Score vs Длина")
    
    # НОВЫЕ ГРАФИКИ
    fig_cer = px.line(bucket_df, x="Длина", y="Token CER", markers=True, title="Token CER vs Длина (Меньше - лучше)")
    fig_token_ed = px.line(bucket_df, x="Длина", y="Token ED", markers=True, title="Token Edit Distance vs Длина")
    fig_bleu = px.line(bucket_df, x="Длина", y="BLEU", markers=True, title="BLEU Score vs Длина")
    fig_thr = px.line(bucket_df, x="Длина", y="Throughput", markers=True, title="Пропускная способность (Токены/сек)")
    
    return summary_df, fig_samples, fig_em, fig_errs, fig_struct, fig_cer, fig_token_ed, fig_bleu, fig_thr


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
            if 2 in pred_seq:
                num_tokens = pred_seq.index(2) + 1
            else:
                num_tokens = len(pred_seq)
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
            
            # ИСПРАВЛЕННЫЙ BLEU SCORE
            if gt_len > 0:
                smooth_fn = SmoothingFunction().method4
                # Используем меньшие n-граммы для коротких формул, чтобы избежать BLEU = 0.0
                if gt_len == 1: w = (1.0, 0, 0, 0)
                elif gt_len == 2: w = (0.5, 0.5, 0, 0)
                elif gt_len == 3: w = (0.33, 0.33, 0.33, 0)
                else: w = (0.25, 0.25, 0.25, 0.25)
                bleu_val = sentence_bleu([gt_t], pred_t, weights=w, smoothing_function=smooth_fn)
            else:
                bleu_val = 0.0
            
            ins, dl, sub = get_edit_operations(gt_t, pred_t)
            token_ed = ins + dl + sub
            token_cer = token_ed / max(1, gt_len) # Token CER (TER)
            
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

    # Глобальные метрики
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
        empty = go.Figure()
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

    return fig_loss, fig_ed, fig_acc, fig_lr, fig_step

def compare_inference_models(files):
    if not files:
        empty_fig = go.Figure()
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

    return fig_global_acc, fig_global_err, fig_global_perf, fig_bucket_em, fig_bucket_bleu, fig_bucket_struct, fig_bucket_cer, fig_bucket_ed, fig_bucket_thr


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Переводчик Математики в LaTeX")

    with gr.Tabs():
        with gr.TabItem("Конвертер"):
            with gr.Accordion("Настройки и загрузка модели", open=True):
                with gr.Row():
                    arch_dropdown = gr.Dropdown(choices=["SwinMambaTex", "SwinTransformerTex", "SwinGConvTex"], label="Архитектура", value="SwinMambaTex")
                    upload_vocab = gr.File(label="vocab.json", file_types=[".json"])
                    upload_weights = gr.File(label="Веса (.safetensors)", file_types=[".safetensors"])
                with gr.Row():
                    load_model_btn = gr.Button("Загрузить модель", variant="primary")
                    model_status = gr.Textbox(label="Статус", interactive=False)
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

if __name__ == "__main__":
    demo.launch(share=False)