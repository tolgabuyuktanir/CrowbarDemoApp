from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_models(X_train, y_train, model_name):
    if model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "DecisionTree":
        model = DecisionTreeClassifier()
    elif model_name == "MLP":
        model = MLPClassifier()
    elif model_name == "RandomForest":
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid model name")
    model.fit(X_train, y_train)
    return model

def test_models(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
     # roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    return y_pred,accuracy, cm, cr