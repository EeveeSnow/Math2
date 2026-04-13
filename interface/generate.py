import torch
import tempfile
import time
import json
import os

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime, timezone
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import interface.configs as conf
from wraper import encode_batch, decode_tokens
from interface.metrics import (
    avg_edit_distance,
    exact_match,
    token_accuracy,
    ast_match_score,
    get_edit_operations,
)
from interface.helpers import _bucket_name, tokenize_latex, _prepare_gradio_data


def run_test_dataset(
    max_samples, batch_size, generation_type, progress=gr.Progress(track_tqdm=True)
):
    if conf.model is None or conf.vocab_obj is None:
        raise ValueError("Загрузите модель и словарь перед запуском.")

    max_samples, batch_size = int(max_samples), int(batch_size)
    split_name = "test" if max_samples <= 0 else f"test[:{max_samples}]"
    ds = load_dataset("deepcopy/MathWriting-Human", split=split_name)

    all_pred_texts, all_gt_texts, items = [], [], []
    bucket_raw = {k: [] for k in conf.BUCKET_KEYS}

    total_inference_time = 0.0
    total_generated_tokens = 0

    for i in progress.tqdm(range(0, len(ds), batch_size), desc="Тестирование"):
        batch = ds[i : i + batch_size]
        images = [conf.image_transform(img) for img in batch["image"]]
        gt_texts = batch["latex"]  # Ground truth strings for the batch

        image_tensor = torch.stack(images).to(conf.DEVICE, torch.bfloat16)

        t0 = time.time()
        with torch.no_grad():
            if generation_type == "Beam Search":
                # Returns (B, beam_size, seq_len)
                predictions = conf.model.generate_beam_search(
                    images=image_tensor,
                    start_token_id=1,
                    eos_token_id=2,
                    beam_size=5,
                    max_new_tokens=256,
                )
            else:
                # Returns (B, seq_len)
                predictions = conf.model.generate(
                    images=image_tensor,
                    start_token_id=1,
                    eos_token_id=2,
                    max_new_tokens=256,
                )

        batch_time = time.time() - t0
        total_inference_time += batch_time
        batch_len = len(gt_texts)

        # Loop through the batch using an integer index
        for b_idx in range(batch_len):
            # 1. Identify the specific sequence for this batch item
            if generation_type == "Beam Search":
                # Take the top beam (index 0) for the current image
                pred_tokens = predictions[b_idx][0]
            else:
                # Take the generated sequence for the current image
                pred_tokens = predictions[b_idx]

            # 2. Extract text and count tokens
            pred_seq_list = pred_tokens.tolist()
            num_tokens = (
                pred_seq_list.index(2) + 1 if 2 in pred_seq_list else len(pred_seq_list)
            )
            total_generated_tokens += num_tokens

            pred_text = decode_tokens(conf.vocab, pred_tokens)
            gt = gt_texts[b_idx]  # This is now safe because b_idx is an integer

            sample_time = batch_time / batch_len

            # 3. Metrics Calculation
            gt_t = tokenize_latex(gt)
            pred_t = tokenize_latex(pred_text)
            gt_len = len(gt_t)

            p_tensor = encode_batch([pred_text], conf.vocab_obj)
            t_tensor = encode_batch([gt], conf.vocab_obj)

            ed = avg_edit_distance(p_tensor, t_tensor)
            em = exact_match(p_tensor, t_tensor)

            # BLEU calculation
            if gt_len > 0:
                smooth_fn = SmoothingFunction().method4
                if gt_len == 1:
                    w = (1.0, 0, 0, 0)
                elif gt_len == 2:
                    w = (0.5, 0.5, 0, 0)
                elif gt_len == 3:
                    w = (0.33, 0.33, 0.33, 0)
                else:
                    w = (0.25, 0.25, 0.25, 0.25)
                bleu_val = sentence_bleu(
                    [gt_t], pred_t, weights=w, smoothing_function=smooth_fn
                )
            else:
                bleu_val = 0.0

            ins, dl, sub = get_edit_operations(gt_t, pred_t)
            token_ed = ins + dl + sub
            token_cer = token_ed / max(1, gt_len)
            ast_score = ast_match_score(gt, pred_text, gt_t, pred_t)

            # 4. Storage
            bucket = _bucket_name(gt_len)
            bucket_raw[bucket].append(
                {
                    "exact_match": em,
                    "edit_distance": ed,
                    "bleu": bleu_val,
                    "ins": ins,
                    "del": dl,
                    "sub": sub,
                    "struct_score": ast_score,
                    "tokens": num_tokens,
                    "time": sample_time,
                    "token_ed": token_ed,
                    "token_cer": token_cer,
                }
            )

            items.append(
                {
                    "index": i + b_idx,
                    "ground_truth": gt,
                    "prediction": pred_text,
                    "gt_length": gt_len,
                    "pred_length": len(pred_t),
                    "exact_match": float(em),
                    "ins": ins,
                    "del": dl,
                    "sub": sub,
                    "ast_struct_score": float(ast_score),
                    "latency_sec": sample_time,
                    "bleu": float(bleu_val),
                    "token_ed": int(token_ed),
                    "token_cer": float(token_cer),
                }
            )
            all_pred_texts.append(pred_text)
            all_gt_texts.append(gt)

    # --- Final calculations ---
    pred_tensor = pad_sequence(
        encode_batch(all_pred_texts, conf.vocab_obj), batch_first=True, padding_value=0
    )
    tgt_tensor = pad_sequence(
        encode_batch(all_gt_texts, conf.vocab_obj), batch_first=True, padding_value=0
    )
    token_acc = token_accuracy(pred_tensor, tgt_tensor)

    throughput_tps = (
        total_generated_tokens / total_inference_time if total_inference_time > 0 else 0
    )
    latency_per_token_ms = (
        (total_inference_time / total_generated_tokens * 1000)
        if total_generated_tokens > 0
        else 0
    )

    bucket_stats = {}
    avg_ast = 0.0
    for bucket in conf.BUCKET_KEYS:
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
                "tokens_per_sec": t_toks / t_time if t_time > 0 else 0,
            }
            avg_ast += sum(v["struct_score"] for v in vals)

    avg_ast /= max(len(items), 1)

    log_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "samples": len(items),
            "token_accuracy": float(token_acc),
            "exact_match": sum(i["exact_match"] for i in items) / max(len(items), 1),
            "avg_ast_score": float(avg_ast),
            "bleu": sum(i["bleu"] for i in items) / max(len(items), 1),
            "avg_cer": sum(i["token_cer"] for i in items) / max(len(items), 1),
            "avg_token_ed": sum(i["token_ed"] for i in items) / max(len(items), 1),
            "throughput_tokens_per_sec": throughput_tps,
            "latency_per_token_ms": latency_per_token_ms,
        },
        "bucket_stats": bucket_stats,
        "items": items,
    }

    fd, out_path = tempfile.mkstemp(prefix="test_logs_", suffix=".json")
    os.close(fd)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    return _prepare_gradio_data(log_data) + (out_path,)


def generate_and_visualize(
    max_samples, batch_size, temperature, progress=gr.Progress(track_tqdm=True)
):
    conf.routing_history.clear()

    if conf.ARCH != "SwinMoETex":
        return "Ошибка: Доступно только для MoE"
    split_name = "test" if max_samples <= 0 else f"test[:{max_samples}]"
    ds = load_dataset("deepcopy/MathWriting-Human", split=split_name)

    for i in progress.tqdm(range(0, len(ds), batch_size), desc="Тестирование"):
        batch = ds[i : i + batch_size]
        images = [conf.image_transform(img) for img in batch["image"]]

        image_tensor = torch.stack(images).to(conf.DEVICE, torch.bfloat16)

        prediction = conf.model.generate(
            images=image_tensor,
            start_token_id=1,
            eos_token_id=2,
            beam_size=5,
            max_new_tokens=256,
        )
        decoded_tokens = decode_tokens(conf.vocab, prediction[0])

        processed_history = {}
        for layer, history_list in conf.routing_history.items():
            processed_history[layer] = np.concatenate(history_list, axis=0)

        all_expert_selections = []
        for layer, assignments in processed_history.items():
            all_expert_selections.extend(assignments.flatten())

        expert_counts = np.bincount(all_expert_selections, minlength=conf.NUM_EXPERTS)

        fig_balance = px.bar(
            x=[f"Expert {i}" for i in range(conf.NUM_EXPERTS)],
            y=expert_counts,
            labels={"x": "Expert ID", "y": "Number of Tokens Handled"},
            title="Global Expert Load Balancing",
            color=expert_counts,
            color_continuous_scale="Blues",
        )

        layer_names = list(processed_history.keys())
        primary_experts = np.zeros((len(decoded_tokens), len(layer_names)))
        hover_text = np.empty((len(decoded_tokens), len(layer_names)), dtype=object)

        for col_idx, layer in enumerate(layer_names):
            assignments = processed_history[layer]
            for row_idx in range(len(decoded_tokens)):
                if row_idx < len(assignments):
                    primary = assignments[row_idx, 0]
                    secondary = assignments[row_idx, 1] if conf.TOP_K > 1 else "N/A"
                    primary_experts[row_idx, col_idx] = primary
                    hover_text[row_idx, col_idx] = (
                        f"Token: '{decoded_tokens[row_idx]}'<br>Primary: Exp {primary}<br>Secondary: Exp {secondary}"
                    )
                else:
                    primary_experts[row_idx, col_idx] = -1
                    hover_text[row_idx, col_idx] = "No Data"

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=primary_experts,
                x=layer_names,
                y=decoded_tokens,
                text=hover_text,
                hoverinfo="text",
                colorscale="Turbo",
                zmin=0,
                zmax=conf.NUM_EXPERTS - 1,
                colorbar=dict(title="Expert ID"),
            )
        )

        fig_heatmap.update_layout(
            title="Token-by-Token Routing Behavior",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(side="top"),
            height=max(
                400, len(decoded_tokens) * 20
            ),
        )

        return decoded_tokens, fig_balance, fig_heatmap
