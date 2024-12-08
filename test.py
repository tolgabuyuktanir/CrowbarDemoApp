import pandas as pd
from UserBehaviourDeepLearningBasedEmbeddings import *
from UserBehaviourGraphBasedEmbeddings import *
from preprocess_data import *

X, y = create_dataset("example_data/first_100_dataset_session-based_isoweekofday_hour", max_len=4)
auto_encoder_model  = UserBehaviourDeepLearningBasedEmbeddings(X).forwardAutoEncoder()
lstm_encoder_model = UserBehaviourDeepLearningBasedEmbeddings(X).forwardLSTMEncoder()
#X = np.expand_dims(X, axis=1)  # Make it (samples, timesteps, features)
X = auto_encoder_model.predict(X)
