import sys
import wraper 
import gradio as gr

import interface.configs

from interface.helpers import convert_checkpoint, run_summary, process_image
from interface.svg import update_model_svg, get_mamba_svg
from interface.generate import run_test_dataset
from interface.loaders import load_custom_model, load_test_logs, load_training_graphs
from interface.compare import compare_training_logs, compare_inference_models

sys.modules["data"] = wraper



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
                        arch_dropdown = gr.Dropdown(choices=["SwinMambaTex", "SwinTransformerTex", "SwinGConvTex", "SwinMoETex"], label="Архитектура", value="SwinMambaTex")
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
                generation_type = gr.Dropdown(choices=["Beam Search", "Gready"], label="Вид генерации", value="SwinMambaTex")
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
                inputs=[max_samples, batch_size, generation_type],
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