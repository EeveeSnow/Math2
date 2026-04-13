import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from safetensors.torch import load_file
from torch import bfloat16

import interface.configs as conf
from interface.helpers import apply_plot_theme, _prepare_gradio_data, CustomVocab
from models.model_conv import SwinGConvTex
from models.model_transformer import SwinTransformerTex

from models.model_mamba import SwinMambaTex
from models.model_MOE import SwinMoETex


def load_custom_model(arch_name, vocab_file, weights_file):
    if not vocab_file: 
        return "Ошибка: Пожалуйста, загрузите файл conf.vocab.json."
    
    try:
        with open(vocab_file.name, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        conf.vocab_obj = CustomVocab(vocab_data)
        conf.vocab = {idx: token for idx, token in enumerate(conf.vocab_obj.itos)}
        conf.VOCAB_SIZE = len(conf.vocab_obj.itos)
    except Exception as e:
        return f"Ошибка при загрузке словаря: {str(e)}"

    if not weights_file:
        return f"Словарь загружен (Размер: {conf.VOCAB_SIZE}). Загрузите файл .safetensors для инициализации модели."
    if not arch_name:
        return "Ошибка: Выберите архитектуру модели."

    try:
        if arch_name == "SwinMambaTex":
            if conf.DEVICE.type == "cuda":
                new_model = SwinMambaTex(vocab_size=conf.VOCAB_SIZE, d_model=512).to(bfloat16).cuda()
            else:
                return "Ошибка: Mamba доступна только на CUDA."
        elif arch_name == "SwinMoETex":
            if conf.DEVICE.type == "cuda":
                new_model = SwinMoETex(vocab_size=conf.VOCAB_SIZE, d_model=512).to(bfloat16).cuda()
            else:
                return "Ошибка: MoE доступна только на CUDA."
        elif arch_name == "SwinGConvTex":
            new_model = SwinGConvTex(vocab_size=conf.VOCAB_SIZE, d_model=512).to(conf.DEVICE, bfloat16)
        elif arch_name == "SwinTransformerTex":
            new_model = SwinTransformerTex(vocab_size=conf.VOCAB_SIZE, d_model=512).to(conf.DEVICE, bfloat16)
        else:
            return f"Ошибка: Неизвестная архитектура '{arch_name}'."
        conf.ARCH = arch_name

        state_dict = load_file(weights_file.name)
        new_model.load_state_dict(state_dict)
        new_model.eval()
        #if arch_name == "SwinMoETex":
            #new_model.decoder = compile(new_model.decoder, mode="reduce-overhead")
        conf.model = new_model
        return f"Модель {arch_name} успешно загружена!\nРазмер словаря: {conf.VOCAB_SIZE}\nУстройство: {conf.DEVICE}"
    except Exception as e:
        return f"Ошибка загрузки весов: {str(e)}"

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