from UserBehaviourDeepLearningBasedEmbeddings import UserBehaviourDeepLearningBasedEmbeddings
from UserBehaviourGraphBasedEmbeddings import UserBehaviourGraphBasedEmbeddings
from preprocess_data import create_dataset, read_file
import MachineLearningModels as mlm
import DeepLearningModels as dlm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from prefixspan import PrefixSpan
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay

X, y, X_train, X_test, y_train, y_test = None, None, None, None, None, None
model = None
y_pred = []
def train_unsupervised_methods(unsupervised_methods_time_dependency, unsupervised_method_conf):
    global X, y, X_train, X_test, y_train, y_test
    ps = PrefixSpan(X_train)
    min_support = 1
    frequent_patterns = ps.frequent(min_support)
    patterns = [pattern for pattern in frequent_patterns if len(pattern[1]) == 3]
    print(patterns)
    sorted_patterns = sorted(patterns, key=lambda x: x[0], reverse=True)
    return sorted_patterns


def train_model(file_input, train_test_split_slider, prefetching_type, supervised_or_unsupervised, unsupervised_methods,
                unsupervised_methods_time_dependency, unsupervised_method_conf, supervised_methods,
                vector_presentation):
    global X, y, X_train, X_test, y_train, y_test
    global model
    if prefetching_type == ["User Data Analysis Based Prefetching"]:
        X, y = create_dataset(file_input, user_id=1, max_len=4)
    elif unsupervised_methods_time_dependency == ["Time-dependent"]:
        X, y = create_dataset(file_input, time_dependency=True)
    else:
        X, y = create_dataset(file_input, user_id=None, max_len=4)

    if vector_presentation == "Word2Vec" and supervised_or_unsupervised == "Supervised":
        userBehaviourEmbedding = UserBehaviourGraphBasedEmbeddings(X)
        word2vec_model = userBehaviourEmbedding.forwardWord2Vec()
        X = userBehaviourEmbedding.createEmbeddigns(word2vec_model)
    elif vector_presentation == "Node2Vec" and supervised_or_unsupervised == "Supervised":
        userBehaviourEmbedding = UserBehaviourGraphBasedEmbeddings(X)
        node2vec_model = userBehaviourEmbedding.forwardNode2Vec()
        X = userBehaviourEmbedding.createEmbeddigns(node2vec_model)
    elif vector_presentation == "Deep Walk" and supervised_or_unsupervised == "Supervised":
        userBehaviourEmbedding = UserBehaviourGraphBasedEmbeddings(X)
        deepwalk_model = userBehaviourEmbedding.forwardDeepWalk()
        X = userBehaviourEmbedding.createEmbeddigns(deepwalk_model)
    elif vector_presentation == "LSTM Encoder" and supervised_or_unsupervised == "Supervised":
        userBehaviourEmbedding = UserBehaviourDeepLearningBasedEmbeddings(X)
        lstm_encoder_model = userBehaviourEmbedding.forwardLSTMEncoder()
        # Prepare data for the LSTM
        X = np.expand_dims(X, axis=1)  # Make it (samples, timesteps, features)
        X = lstm_encoder_model.predict(X)
    elif vector_presentation == "Transformers" and supervised_or_unsupervised == "Supervised":
        userBehaviourEmbedding = UserBehaviourDeepLearningBasedEmbeddings(X)
        transformer_model = userBehaviourEmbedding.forwardAutoEncoder()
        X = transformer_model.predict(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider/100, random_state=42)
    if unsupervised_methods in ["PrefixSpan"] and supervised_or_unsupervised == "Unsupervised":
        model = train_unsupervised_methods(unsupervised_methods_time_dependency, unsupervised_method_conf)
    elif supervised_methods in ["LSTM", "BiLSTM"] and supervised_or_unsupervised == "Supervised":
        print(np.unique(y).shape[0])
        model  = dlm.train_models(X_train, y_train, np.unique(y).shape[0], supervised_methods)
    elif supervised_or_unsupervised == "Supervised":
        model = mlm.train_models(X_train, y_train, supervised_methods)

    if supervised_or_unsupervised == "Supervised":
        y_pred, accuracy, cm, cr = test_model(supervised_methods)
    else:
        y_pred, accuracy, cm, cr = test_unsupervised_model(unsupervised_methods)
        model = "PrefixSpan"
    plt.figure(figsize=(15, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Oranges)
    fig = plt.gcf()
    plt.close(fig)
    return f"{model} Model trained successfully! Accuracy:{accuracy}", pd.DataFrame(cr).transpose(),fig

def test_model(supervised_methods):
    global model, X_test, y_test
    if supervised_methods in ["LSTM", "BiLSTM"]:
        y_pred, accuracy, cm, cr = dlm.test_models(model, X_test, y_test)
    else:
        y_pred, accuracy, cm, cr = mlm.test_models(model, X_test, y_test)
    return y_pred, accuracy, cm, cr

def test_unsupervised_model(unsupervised_methods):
    y_pred.clear()
    global model, X_test, y_test
    X_train_ps = set(tuple(i[1][:-1]) for i in model)
    y_train_ps = [i[1][-1] for i in model]
    correct = 0
    for i in X_test:
        if tuple(i[:-1]) in X_train_ps:
            y_pred.append(y_train_ps[list(X_train_ps).index(tuple(i[:-1]))])
        else:
            y_pred.append(-1)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    # roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    return y_pred, accuracy, cm, cr

df = pd.DataFrame(columns=["Conf", "Cache hit", "Cache miss", "Total"])

def cache_hit_miss(prefetching_type, supervised_or_unsupervised, unsupervised_methods,
                        unsupervised_methods_time_dependency, unsupervised_method_conf, supervised_methods,
                        vector_presentation, cache_size):
    if supervised_or_unsupervised == "Unsupervised":
        conf = (prefetching_type, unsupervised_methods, unsupervised_methods_time_dependency, unsupervised_method_conf,
                cache_size)
    else:
        conf = (supervised_methods, vector_presentation, cache_size)
    global model, X_test, y_test, df
    cache = []
    cache_miss = 0
    cache_hit = 0
    # calculate cache hits and misses
    for i in range(len(X_test)):
        if y_test[i] in cache:
            cache_hit += 1
        else:
            if supervised_methods in ["LSTM", "BiLSTM"] and supervised_or_unsupervised == "Supervised":
                print(X_test[i].shape)
                pred = np.argmax(model.predict(np.array(X_test[i], dtype='float32').reshape(-1, 1, X_test.shape[1])))
            elif supervised_or_unsupervised == "Supervised":
                pred = model.predict([X_test[i]])
            else:
                pred = y_pred[i]
                if unsupervised_method_conf in ["Batch-processing", "Hybrid-Processing"]:
                    cache.append(pred)
            if pred == y_test[i]:
                cache_hit += 1
            else:
                cache_miss += 1
            if len(cache) < cache_size:
                cache.append(pred)
            else:
                cache.pop(0)
                cache.append(pred)
    print(cache_hit, cache_miss)
    new_row = pd.DataFrame([{
        "Conf": conf,
        "Cache hit": cache_hit,
        "Cache miss": cache_miss,
        "Total": cache_hit + cache_miss
    }])
    df = df[df["Conf"] != conf]
    df = pd.concat([df, new_row], ignore_index=True)
    styled_df = df.style.applymap(
            lambda x: "color: red;" if isinstance(x, int) and x < 50 else "color: blue;"
        ).set_table_attributes('style="width:100%; border-collapse: collapse;"')

    # create plot of the styled dataframe
    ax = df.plot(kind='bar', x='Conf', y=['Cache hit', 'Cache miss'], title="Cache Hit vs Cache Miss")

    # Customize the plot
    ax.set_ylabel('Count')
    ax.set_xlabel('Config')
    ax.set_xticklabels(df['Conf'], rotation=90)

    # Save the plot to a file
    plt.tight_layout()
    fig = plt.gcf()
    plt.close(fig)
    return styled_df, fig


def download_model_file():
    model_file = "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    return model_file

def download_embedding_file():
    embedding_file = "embedding.pkl"
    with open(embedding_file, "wb") as f:
        pickle.dump(X, f)
    return embedding_file
def clear_cache_performance_results():
    global dataframe
    dataframe = pd.DataFrame(columns=["Conf", "Cache hit", "Cache miss", "Total"])
    return dataframe
