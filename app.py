from cProfile import label
from email.policy import default

import numpy as np
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from crowbar import *
from preprocess_data import *
matplotlib.use('Agg')


def plot_forecast(file):
    start_year = 2020
    x = np.arange(start_year, 2025 + 1)
    year_count = x.shape[0]
    plt_format = ({"cross": "X", "line": "-", "circle": "o--"})["circle"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(5):
        series = np.arange(0, year_count, dtype=float)
        series = series ** 2 * (i + 1)
        series += np.random.rand(year_count) * 1
        ax.plot(x, series, plt_format)
    return fig



def update_panel_visibility(method_type):
    if method_type == "Unsupervised":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks() as demo:
    file_input = gr.File(label="Upload your file")
    gr.Examples([["example_data/first_100_dataset_session-based_isoweekofday_hour"]], inputs=[file_input])
    with gr.Row(variant="panel"):
        with gr.Accordion("Method selection", open=True):
            train_test_split_slider = gr.Slider(minimum=1, maximum=100, value=20, step=1, interactive=True,
                                                label="% Testset Size")

            train_test_split_slider.change(lambda x: x, [train_test_split_slider])
            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Accordion("Prefetching Type", open=True):
                        prefetching_type = gr.Radio(["User Data Analysis Based Prefetching", "Collective Data Analysis Based Prefetching"],
                                             label="Prefetching Data Type")
                        supervised_or_unsupervised = gr.Radio(
                            ["Unsupervised", "Supervised"],
                            label=" Prefetching Method Type"
                        )
            with gr.Row(variant="panel", visible=False) as panel_unsupervised:
                with gr.Column():
                    with gr.Accordion("Configuration of unsupervised methods", open=True):
                        unsupervised_methods = gr.Radio(["PrefixSpan"], label="Unsupervised Methods",value="PrefixSpan")
                        unsupervised_methods_time_dependency = gr.Radio(["Time-dependent", "Time-independent"], label="Time Dependency", value="Time-independent")
                        unsupervised_method_conf = gr.Radio(
                             ["Real-time processing", "Batch-processing", "Hybrid-Processing"],
                             label="Unsupervised Prefetching Methods", value="Batch-processing")

            with gr.Row(variant="panel", visible=False) as panel_supervised:
                with gr.Column():
                    with gr.Accordion("Configuration of supervised methods", open=True):
                        supervised_methods = gr.Radio(["KNN", "DecisionTree", "MLP", "RandomForest", "LSTM", "BiLSTM"],
                                             label="Supervised Methods", value="RandomForest")
                        vector_presentation = gr.Radio(["Word2Vec", "Node2Vec", "Deep Walk", "LSTM Encoder",
                                                        "Transformers"], label="Vector Presentation Method", value="LSTM Encoder")
            supervised_or_unsupervised.change(
                fn=update_panel_visibility,
                inputs=supervised_or_unsupervised,
                outputs=[panel_unsupervised, panel_supervised]
            )
    with gr.Row(variant="panel"):
        with gr.Column():
            btn_train = gr.Button(value="Train", icon="files/train_icon.png")
            progress_bar = gr.Progress()
            training_result = gr.Textbox(label="Training Output")
            cr   = gr.DataFrame(show_label=True, label="Classification Report", headers=["precision", "recall", "f1-score", "support"], )
            cm_fig = gr.Plot(label="Confusion Matrix")
        btn_train.click(
            fn=train_model,
            inputs=[file_input, train_test_split_slider, prefetching_type, supervised_or_unsupervised,
                    unsupervised_methods,
                    unsupervised_methods_time_dependency, unsupervised_method_conf, supervised_methods,
                    vector_presentation],
            outputs=[training_result, cr, cm_fig], show_progress="full")


    with gr.Row(variant="panel"):
        cache_size = gr.Slider(minimum=0, maximum=1000, value=200, step=1, interactive=True,
                                            label="Cache Size Setting")
    with gr.Row(variant="panel", show_progress=True):
        btn_simulation = gr.Button(value="Simulate Cache Performance", icon="files/simulation.png")
        btn_clear = gr.Button(value="Clear Results", icon="files/simulation.png")
    with gr.Row(variant="panel"):
        with gr.Column():
            dataframe = gr.Dataframe(
                headers=["Conf", "Cache hit", "Cache miss", "Total"],
                datatype=["str", "number", "number", "number"],
                label="Cache Performance"
            )
            btn_simulation.click(
                fn=cache_hit_miss,
                inputs=[prefetching_type, supervised_or_unsupervised, unsupervised_methods,
                        unsupervised_methods_time_dependency, unsupervised_method_conf, supervised_methods,
                        vector_presentation, cache_size],
                outputs=[dataframe, gr.Plot(label="Cache Performance")],
                show_progress="full"
            )
            btn_clear.click(
                fn=clear_cache_performance_results,
                outputs=[dataframe],
                show_progress="minimal"
            )
    with gr.Row(variant="panel"):
        btn_download_embeddings = gr.Button(value="Download Embedding", icon="files/simulation.png")
        btn_download_model = gr.Button(value="Download Model", icon="files/simulation.png")
    with gr.Row(variant="panel"):
        btn_download_embeddings.click(
            fn=download_embedding_file,
            outputs=[gr.File()],
            show_progress="minimal"
        )
        btn_download_model.click(
            fn=download_model_file,
            outputs=[gr.File()],
            show_progress="minimal"
        )

if __name__ == "__main__":
    print(file_input.value)
    demo.launch(share=False)
