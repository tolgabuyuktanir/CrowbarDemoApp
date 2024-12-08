import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay

#train lstm and bilstm
def train_models(X_train, y_train, unique_y_shape, model_name, epoch=10):
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    y_train_cat = to_categorical(y_train, num_classes=unique_y_shape+1)
    if model_name == "LSTM":
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(32),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(y_train_cat.shape[1], activation='sigmoid')  # Adjust activation if necessary
        ])
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train_cat, epochs=epoch)
    elif model_name == "BiLSTM":
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(y_train_cat.shape[1], activation='sigmoid')  # Adjust activation if necessary
        ])
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train_cat, epochs=epoch)
    return model

def test_models(model, X_test, y_test):
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    predictions = model.predict(X_test)
    y_pred = []
    for prediction in predictions:
        y_pred.append(np.argmax(prediction))
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred, output_dict=True)
    # roc_auc = roc_auc_score(y_test, y_pred)
    return y_pred, accuracy, confusion, classification

