from email.policy import default

import numpy as np
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from train_models import *
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


df = pd.DataFrame({
    "Method": ["Real-time unsupervised", "Batch unsupervised", "Hybrid unsupervised", "...", "..."],
    "Cache hit": [5, 2, 54, 3, 2],
    "Cache miss": [20, 20, 7, 3, 8],
    "Total": [14, 3, 6, 2, 6]
})
styler = df.style.highlight_max(color='lightgreen', axis=0)

with gr.Blocks() as demo:
    file_input = gr.File(label="Upload your file")
    gr.Examples([["example_data/first_100_dataset_session-based_isoweekofday_hour"]], inputs=[file_input])
    with gr.Row(variant="panel"):
        with gr.Accordion("Method selection", open=True):
            train_test_split_slider = gr.Slider(minimum=0, maximum=100, value=20, step=1, interactive=True,
                                                label="% Testset Size")

            train_test_split_slider.change(lambda x: x, [train_test_split_slider])
            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Accordion("Prefetching Type", open=True):
                        prefetching_type = gr.Radio(["User Data Analysis Based Prefetching", "Collective Data Analysis Based Prefetching"],
                                             label="Prefetching Type")
            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Accordion("Configuration of unsupervised methods", open=True):
                        unsupervised_methods = gr.Radio(["PrefixSpan"], label="Unsupervised Methods",value=0)
                        unsupervised_methods_time_dependency = gr.Radio(["Time-dependent", "Time-independent"], label="Time Dependency", value=1)
                        unsupervised_method_conf = gr.Radio(
                             ["Real-time processing", "Batch-processing", "Hybrid-Processing"],
                             label="Unsupervised Prefetching Methods", value=1)

            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Accordion("Configuration of supervised methods", open=True):
                        supervised_methods = gr.Radio(["KNN", "DecisionTree", "MLP", "RandomForest", "LSTM", "BiLSTM"],
                                             label="Supervised Methods", value=3)
                        vector_presentation = gr.Radio(["Word2Vec", "Node2Vec", "Deep Walk", "LSTM Encoder",
                                                        "Transformers"], label="Vector Presentation Method", value=3)
    with gr.Row(variant="panel"):
        with gr.Column():
            btn_train = gr.Button(value="Train", icon="files/train_icon.png")
            progress_bar = gr.Progress()
            btn_train.click(
                fn=train_model,
                inputs=[file_input, train_test_split_slider, prefetching_type, unsupervised_methods,
                        unsupervised_methods_time_dependency, unsupervised_method_conf, supervised_methods,
                        vector_presentation],
                outputs=[gr.Textbox(label="Training Output")], show_progress="full")

    with gr.Row(variant="panel"):
        cache_size = gr.Slider(minimum=0, maximum=1000, value=200, step=1, interactive=True,
                                            label="Cache Size Setting")
    with gr.Row(variant="panel"):
        btn_simulation = gr.Button(value="Simulation", icon="files/simulation.png")
    with gr.Row(variant="panel"):
        with gr.Column():
            gr.Dataframe(styler)
        with gr.Column():
            gr.Plot()

if __name__ == "__main__":
    print(file_input.value)
    demo.launch(share=False)
