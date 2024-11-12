from UserBehaviourDeepLearningBasedEmbeddings import UserBehaviourDeepLearningBasedEmbeddings
from UserBehaviourGraphBasedEmbeddings import UserBehaviourGraphBasedEmbeddings
from preprocess_data import create_dataset
import MachineLearningModels as mlm
import DeepLearningModels as dlm
from sklearn.model_selection import train_test_split

from test import userBehaviourEmbedding, lstm_encoder_model

X_train, X_test, y_train, y_test = None, None, None, None

def train_unsupervide_methods(unsupervised_methods_time_dependency, unsupervised_method_conf):
    if unsupervised_methods_time_dependency == ["Time-dependent"]:
            if unsupervised_method_conf == ["Real-time processing"]:
                pass
            elif unsupervised_method_conf == ["Batch-processing"]:
                pass
            else:
                pass
    return None


def train_model(file_input, train_test_split_slider, prefetching_type, unsupervised_methods,
                unsupervised_methods_time_dependency, unsupervised_method_conf, supervised_methods,
                vector_presentation):
    X, y, X_train, X_test, y_train, y_test = None, None, None, None, None, None
    if prefetching_type == ["User Data Analysis Based Prefetching"]:
        pass
    else:
        X, y = create_dataset(file_input, user_id=None, max_len=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider/100, random_state=42)

    if vector_presentation == "Word2Vec":
        userBehaviourEmbedding = UserBehaviourGraphBasedEmbeddings(X_train)
        word2vec_model = userBehaviourEmbedding.forwardWord2Vec()
        X_train = userBehaviourEmbedding.createEmbeddigns(word2vec_model)
    elif vector_presentation == "Node2Vec":
        userBehaviourEmbedding = UserBehaviourGraphBasedEmbeddings(X_train)
        node2vec_model = userBehaviourEmbedding.forwardNode2Vec()
        X_train = userBehaviourEmbedding.createEmbeddigns(node2vec_model)
    elif vector_presentation == "Deep Walk":
        userBehaviourEmbedding = UserBehaviourGraphBasedEmbeddings(X_train)
        deepwalk_model = userBehaviourEmbedding.forwardDeepWalk()
        X_train = userBehaviourEmbedding.createEmbeddigns(deepwalk_model)
    elif vector_presentation == "LSTM Encoder":
        userBehaviourEmbedding = UserBehaviourDeepLearningBasedEmbeddings(X_train)
        lstm_encoder_model = userBehaviourEmbedding.forwardLSTMEncoder()
        X_train = lstm_encoder_model.predict(X_train)
    else:
        userBehaviourEmbedding = UserBehaviourDeepLearningBasedEmbeddings(X_train)
        transformer_model = userBehaviourEmbedding.forwardTransformer()
        X_train = transformer_model.predict(X_train)

    if unsupervised_methods == ["PrefixSpan"]:
        unsupervised_models = train_unsupervide_methods(unsupervised_methods_time_dependency, unsupervised_method_conf)
    model  = mlm.train_models(X_train, y_train, supervised_methods)
    return mlm.test_models(model, X_test, y_test)