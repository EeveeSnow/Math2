import json
import os
import sys
import re
import tempfile
import time
from datetime import datetime

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

from model import SwiGLiT 
from model_mamba_1layer import SwinMambaTex

sys.modules["data"] = wraper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_obj = None
vocab = None
VOCAB_SIZE = 0
model = None


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
    
    if not vocab_file:
        return "Ошибка: Пожалуйста, загрузите файл vocab.json."
    
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
            if DEVICE == torch.device("cuda"):
                new_model = SwinMambaTex(vocab_size=VOCAB_SIZE, d_model=512).to(torch.bfloat16).cuda()
            else:
                return "Ошибка: Mamba доступна только на CUDA."
        elif arch_name == "SwiGLiT":
            new_model = SwiGLiT(vocab_size=VOCAB_SIZE, d_model=512).to(DEVICE)
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
    if length < 10: return "<10"
    if length > 100: return ">100"
    if length == 100: return "90-100"
    lower = (length // 10) * 10
    upper = lower + 9
    if upper == 99: upper += 1
    return f"{lower}-{upper}"

def tokenize_latex(s):
    s = str(s).replace(" ", "")
    return re.findall(r'\\[a-zA-Z]+|.', s)

def calc_bleu(gt, pred):
    smoother = SmoothingFunction().method1
    gt_tokens = tokenize_latex(gt)
    pred_tokens = tokenize_latex(pred)
    if not gt_tokens: return 0.0
    return sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)

def _prepare_gradio_data(log_data):
    metrics = log_data["metrics"]
    summary_df = pd.DataFrame([{
        "Примеров": metrics.get("samples", 0),
        "Точность токенов": f"{metrics.get('token_accuracy', 0.0):.4f}",
        "Exact Match": f"{metrics.get('exact_match', 0.0):.4f}",
        "BLEU Score": f"{metrics.get('bleu', 0.0):.4f}",
        "CER (Симв. ошибка)": f"{metrics.get('avg_cer', 0.0):.4f}", 
        "Скорость (штук/сек)": f"{metrics.get('inference_samples_per_sec', 0.0):.2f}",
        "Ср. Edit Dist": f"{metrics.get('avg_edit_distance', 0.0):.4f}",
        "Норм. Edit Dist": f"{metrics.get('normalized_edit_distance', 0.0):.4f}",
        "Ошибка длины (Абс)": f"{metrics.get('avg_length_abs_error', 0.0):.4f}"
    }])
    
    buckets = list(log_data["bucket_stats"].keys())
    bucket_df = pd.DataFrame({
        "Длинна": buckets,
        "Количество изображений": [log_data["bucket_stats"][k].get("sample_count", 0) for k in buckets],
        "Полное совпадение": [log_data["bucket_stats"][k].get("exact_match", 0.0) for k in buckets],
        "BLEU": [log_data["bucket_stats"][k].get("avg_bleu", 0.0) for k in buckets],
        "Средняя дистанция ошибки": [log_data["bucket_stats"][k].get("avg_edit_distance", 0.0) for k in buckets]
    })
    
    # Генерация графиков Plotly (с цифрами на корзинах)
    fig_samples = px.bar(bucket_df, x="Длинна", y="Количество изображений", text_auto=True, title="Количество примеров по бакетам (длине)")
    fig_em = px.bar(bucket_df, x="Длинна", y="Полное совпадение", text_auto='.3f', title="Точное совпадение (Exact Match) по бакетам")
    fig_bleu = px.bar(bucket_df, x="Длинна", y="BLEU", text_auto='.3f', title="BLEU Score по бакетам")
    fig_ed = px.bar(bucket_df, x="Длинна", y="Средняя дистанция ошибки", text_auto='.3f', title="Ср. расстояние редактирования по бакетам")
    
    return summary_df, fig_samples, fig_em, fig_bleu, fig_ed


def run_test_dataset(max_samples, batch_size, progress=gr.Progress(track_tqdm=True)):
    if model is None or vocab_obj is None:
        raise ValueError("Загрузите модель и словарь перед запуском.")
        
    max_samples = int(max_samples)
    batch_size = int(batch_size)

    split_name = "test" if max_samples <= 0 else f"test[:{max_samples}]"
    ds = load_dataset("deepcopy/MathWriting-Human", split=split_name)

    all_pred_texts, all_gt_texts, items = [], [], []
    bucket_raw = {k: [] for k in BUCKET_KEYS}
    total_inference_time = 0.0 

    for i in progress.tqdm(range(0, len(ds), batch_size), desc="Идет тестирование"):
        batch = ds[i : i + batch_size]
        images = [image_transform(img) for img in batch["image"]]
        
        if DEVICE == torch.device("cuda"):
            image_tensor = torch.stack(images).to(torch.bfloat16).cuda()
        else:
            image_tensor = torch.stack(images).to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            predictions = model.generate_beam_search(
                images=image_tensor, start_token_id=1, eos_token_id=2, beam_size=5, max_new_tokens=256)
        total_inference_time += (time.time() - t0)

        gt_texts = batch["latex"]
        for local_idx in range(predictions.shape[0]):
            candidates = [decode_tokens(vocab, predictions[local_idx][k]) for k in range(len(predictions[local_idx]))]
            pred_text = candidates[0]
            
            gt = gt_texts[local_idx]
            gt_len = len(''.join(gt.split()))
            pred_len = len(''.join(pred_text.split()))
            
            p_tensor = encode_batch([pred_text], vocab_obj)
            t_tensor = encode_batch([gt], vocab_obj)
            
            ed = avg_edit_distance(p_tensor, t_tensor)
            em = exact_match(p_tensor, t_tensor)
            bleu_val = calc_bleu(gt, pred_text)
            
            gt_chars, pred_chars = gt.replace(" ", ""), pred_text.replace(" ", "")
            char_ed = nltk.edit_distance(gt_chars, pred_chars)
            cer_val = char_ed / max(len(gt_chars), 1)

            bucket = _bucket_name(gt_len)
            bucket_raw[bucket].append({"exact_match": em, "edit_distance": ed, "bleu": bleu_val})

            items.append({
                "index": i + local_idx,
                "ground_truth": gt,
                "prediction": pred_text,
                "gt_length": gt_len,
                "pred_length": pred_len,
                "exact_match": float(em),
                "edit_distance": float(ed),
                "bleu": float(bleu_val),
                "cer": float(cer_val)
            })
            all_pred_texts.append(pred_text)
            all_gt_texts.append(gt)

    pred_tensor = encode_batch(all_pred_texts, vocab_obj)
    tgt_tensor = encode_batch(all_gt_texts, vocab_obj)

    token_acc = token_accuracy(pred_tensor, tgt_tensor)
    em_score, avg_bleu, avg_cer, norm_ed, len_abs_err = 0, 0.0, 0.0, 0.0, 0.0
    edit_score = avg_edit_distance(pred_tensor, tgt_tensor)

    for item in items:
        norm_ed += item["edit_distance"] / max(item["gt_length"], 1)
        len_abs_err += abs(item["pred_length"] - item["gt_length"])
        avg_bleu += item["bleu"]
        avg_cer += item["cer"]

    total = max(len(items), 1)
    norm_ed /= total
    len_abs_err /= total
    avg_bleu /= total
    avg_cer /= total
    samples_per_sec = total / total_inference_time if total_inference_time > 0 else 0.0

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
            "samples": len(items), "token_accuracy": float(token_acc), "exact_match": float(em_score),
            "bleu": float(avg_bleu), "avg_cer": float(avg_cer), "inference_samples_per_sec": float(samples_per_sec),
            "avg_edit_distance": float(edit_score), "normalized_edit_distance": float(norm_ed), "avg_length_abs_error": float(len_abs_err),
        },
        "bucket_stats": bucket_stats,
        "items": items,
    }

    fd, out_path = tempfile.mkstemp(prefix="test_logs_", suffix=".json")
    os.close(fd)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    summary_df, f_samples, f_em, f_bleu, f_ed = _prepare_gradio_data(log_data)
    return summary_df, f_samples, f_em, f_bleu, f_ed, out_path


def load_test_logs(file_obj):
    if file_obj is None:
        return pd.DataFrame(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

    with open(file_obj.name, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    summary_df, f_samples, f_em, f_bleu, f_ed = _prepare_gradio_data(log_data)
    return summary_df, f_samples, f_em, f_bleu, f_ed


def load_raw_data_chunk(file_obj, start_idx, chunk_size):
    if file_obj is None: return pd.DataFrame(), "Файл не загружен."
    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        total = len(items)
        end_idx = min(start_idx + chunk_size, total)
        df = pd.DataFrame(items[start_idx:end_idx])
        return df, f"Показаны строки с {start_idx} по {end_idx - 1} из {total}."
    except Exception as e:
        return pd.DataFrame(), f"Ошибка чтения логов: {str(e)}"


# --- TRAINING GRAPHS LOGIC ---
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
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
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
                "CER": metrics.get("avg_cer", 0),
                "Нормализованная дистанция ошибки": metrics.get("normalized_edit_distance", 0),
                "Средняя дистанция ошибки": metrics.get("avg_edit_distance", 0),
                "Скорость (штук/сек)": metrics.get("inference_samples_per_sec", 0)
            })

            bucket_stats = data.get("bucket_stats", {})
            for bucket, stats in bucket_stats.items():
                bucket_records.append({
                    "Модель": model_name,
                    "Длина": bucket,
                    "Exact Match": stats.get("exact_match", 0),
                    "BLEU Score": stats.get("avg_bleu", 0),
                    "Edit Distance": stats.get("avg_edit_distance", 0)
                })
        except Exception as e:
            print(f"Ошибка обработки {f.name}: {e}")

    df_global = pd.DataFrame(global_records)
    df_bucket = pd.DataFrame(bucket_records)

    if not df_bucket.empty:
        df_bucket['Длина'] = pd.Categorical(df_bucket['Длина'], categories=BUCKET_KEYS, ordered=True)
        df_bucket = df_bucket.sort_values('Длина')

    df_em_bleu = df_global.melt(id_vars=["Модель"], value_vars=["Exact Match", "BLEU Score"], var_name="Метрика", value_name="Значение")
    fig_global_1 = px.bar(df_em_bleu, x="Модель", y="Значение", color="Метрика", barmode="group", text_auto='.3f', title="Глобальные метрики: Точность")

    df_err = df_global.melt(id_vars=["Модель"], value_vars=["CER", "Нормализованная дистанция ошибки", "Средняя дистанция ошибки"], var_name="Метрика", value_name="Значение")
    fig_global_2 = px.bar(df_err, x="Модель", y="Значение", color="Метрика", barmode="group", text_auto='.3f', title="Глобальные метрики: Ошибки (меньше - лучше)")

    fig_speed = px.bar(df_global, x="Модель", y="Скорость (штук/сек)", color="Модель", text_auto='.2f', title="Скорость генерации (FPS)")

    fig_bucket_em = px.line(df_bucket, x="Длина", y="Exact Match", color="Модель", markers=True, title="Сравнение Exact Match по длине формул")

    fig_bucket_bleu = px.line(df_bucket, x="Длина", y="BLEU Score", color="Модель", markers=True, title="Сравнение BLEU Score по длине формул")

    fig_bucket_ed = px.line(df_bucket, x="Длина", y="Edit Distance", color="Модель", markers=True, title="Сравнение Edit Distance по длине формул (меньше - лучше)")

    return fig_global_1, fig_global_2, fig_speed, fig_bucket_em, fig_bucket_bleu, fig_bucket_ed


with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("Приложение для перевода математики в LaTeX")

    with gr.Tabs():
        with gr.TabItem("Конвертер"):
            gr.Markdown("Загрузите изображение математического выражения и получите соответствующий код LaTeX.")
            
            with gr.Accordion("Настройки и загрузка модели", open=True):
                with gr.Row():
                    arch_dropdown = gr.Dropdown(choices=["SwinMambaTex", "SwiGLiT"], label="Архитектура модели", value="SwinMambaTex")
                    upload_vocab = gr.File(label="Загрузить словарь (vocab.json)", file_types=[".json"])
                    upload_weights = gr.File(label="Загрузить веса (.safetensors)", file_types=[".safetensors"])
                with gr.Row():
                    load_model_btn = gr.Button("Загрузить выбранную модель", variant="primary")
                    model_status = gr.Textbox(label="Статус модели", value="Модель не загружена.", interactive=False)
                    
                load_model_btn.click(fn=load_custom_model, inputs=[arch_dropdown, upload_vocab, upload_weights], outputs=[model_status])

            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Загрузить изображение")
                    submit_btn = gr.Button("Преобразовать в LaTeX", variant="primary")
                with gr.Column():
                    output_code = gr.Textbox(label="Полученный LaTeX код", lines=4)
                    output_render = gr.Markdown(label="Отформатированный вывод")

            submit_btn.click(fn=process_image, inputs=[input_img], outputs=[output_code, output_render])

        with gr.TabItem("Статистика датасета"):
            gr.Markdown("Запуск инференса на тестовом наборе данных с подсчетом метрик и построением графиков.")
            with gr.Row():
                max_samples = gr.Number(value=50, precision=0, label="Макс. примеров (0 = весь test split)")
                batch_size = gr.Number(value=8, precision=0, label="Размер батча (Batch size)")
            run_btn = gr.Button("Запустить тестирование", variant="primary")

            summary_table = gr.Dataframe(label="Глобальные результаты тестирования", interactive=False)
            
            with gr.Row():
                plot_samples = gr.Plot(label="Количество примеров")
                plot_em = gr.Plot(label="Exact Match")
            with gr.Row():
                plot_bleu = gr.Plot(label="BLEU Score")
                plot_ed = gr.Plot(label="Edit Distance")
            
            download_logs = gr.File(label="Скачать сырые JSON логи")

            run_btn.click(
                fn=run_test_dataset,
                inputs=[max_samples, batch_size],
                outputs=[summary_table, plot_samples, plot_em, plot_bleu, plot_ed, download_logs],
            )

            gr.Markdown("### Или загрузите уже существующий лог JSON")
            upload_logs = gr.File(file_types=[".json"], label="Загрузить JSON лог")
            load_btn = gr.Button("Загрузить логи")
            
            load_btn.click(
                fn=load_test_logs,
                inputs=[upload_logs],
                outputs=[summary_table, plot_samples, plot_em, plot_bleu, plot_ed],
            )

        with gr.TabItem("Сырые данные тренировки"):
            gr.Markdown("Здесь можно детально просмотреть данные для каждого изображения. **Данные загружаются частями (chunks), чтобы не перегружать RAM.**")
            raw_json_input = gr.File(label="Загрузите JSON файл, сгенерированный во вкладке Статистики", file_types=[".json"])
            
            with gr.Row():
                start_idx = gr.Number(value=0, precision=0, label="Начальный индекс")
                chunk_size = gr.Number(value=50, precision=0, label="Количество строк (Chunk size)")
            
            load_chunk_btn = gr.Button("Показать строки", variant="primary")
            chunk_status = gr.Markdown("Файл не загружен")
            raw_dataframe = gr.Dataframe(label="Детальная таблица по картинкам", interactive=False, wrap=True)
            
            load_chunk_btn.click(
                fn=load_raw_data_chunk,
                inputs=[raw_json_input, start_idx, chunk_size],
                outputs=[raw_dataframe, chunk_status]
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
            gr.Markdown("Загрузите несколько **JSON файлов**, полученных во вкладке `Статистика датасета`, чтобы визуально сравнить результаты нескольких моделей.")
            
            upload_multi_json = gr.File(label="Выберите несколько JSON файлов (Логи статистики)", file_types=[".json"], file_count="multiple")
            compare_btn = gr.Button("Построить графики сравнения моделей", variant="primary")
            
            gr.Markdown("### 📊 Глобальные метрики")
            with gr.Row():
                comp_global_1 = gr.Plot()
                comp_global_2 = gr.Plot()
            with gr.Row():
                comp_speed = gr.Plot()
                
            gr.Markdown("### 📈 Детальные метрики по бакетам (по длине формул)")
            with gr.Row():
                comp_bucket_em = gr.Plot()
                comp_bucket_bleu = gr.Plot()
            with gr.Row():
                comp_bucket_ed = gr.Plot()
            
            compare_btn.click(
                fn=compare_inference_models,
                inputs=[upload_multi_json],
                outputs=[
                    comp_global_1, comp_global_2, comp_speed, 
                    comp_bucket_em, comp_bucket_bleu, comp_bucket_ed
                ]
            )

if __name__ == "__main__":
    print("Запуск интерфейса Gradio...")
    demo.launch(share=False)