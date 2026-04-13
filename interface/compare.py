import os
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from interface.helpers import apply_plot_theme
import interface.configs as conf

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
        df_bucket['Длина'] = pd.Categorical(df_bucket['Длина'], categories=conf.BUCKET_KEYS, ordered=True)
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
