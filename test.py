import pandas as pd
from UserBehaviourDeepLearningBasedEmbeddings import *
from UserBehaviourGraphBasedEmbeddings import *
from preprocess_data import *

X, y = create_dataset("example_data/first_100_dataset_session-based_isoweekofday_hour", max_len=4)
auto_encoder_model  = UserBehaviourDeepLearningBasedEmbeddings(X).forwardAutoEncoder()
lstm_encoder_model = UserBehaviourDeepLearningBasedEmbeddings(X).forwardLSTMEncoder()
word2vec_model = UserBehaviourGraphBasedEmbeddings(X).forwardWord2Vec()
# node2vec_model = UserBehaviourGraphBasedEmbeddings(X).forwardNode2Vec()
deepwalk_model = UserBehaviourGraphBasedEmbeddings(X).forwardDeepWalk()
userBehaviourEmbedding = UserBehaviourGraphBasedEmbeddings(X)
node2vec_model = userBehaviourEmbedding.forwardNode2Vec()
X = userBehaviourEmbedding.createEmbeddigns(node2vec_model)

