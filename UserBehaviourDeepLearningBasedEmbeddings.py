import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
import setup_logger
import logging
logger = logging.getLogger('UserBehaviourDeepLearningBasedEmbeddings')
class UserBehaviourDeepLearningBasedEmbeddings():
    def __init__(self, data):
        self.data = data

    def forwardAutoEncoder(self):
        logger.info("AutoEncoder")
        X  = self.data
        # Parameters
        input_dim = X.shape[1]
        encoding_dim = 64

        # Define the autoencoder model
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)  # Embedding layer
        encoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        # Train the autoencoder
        autoencoder.fit(X, X,
                        epochs=10,
                        batch_size=128,
                        shuffle=True)

        return encoder

    def forwardLSTMEncoder(self):
        logger.info("LSTMEncoder")
        X = self.data
        # Define the model
        input_dim = X.shape[1]
        latent_dim = 64
        timesteps = 1  # Since each data point is a single timestep

        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(latent_dim, return_state=False)(inputs)
        encoder_model = Model(inputs, encoded)

        return encoder_model