import numpy as np
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import setup_logger
import logging
logger = logging.getLogger('UserBehaviourGraphBasedEmbeddings')

class UserBehaviourGraphBasedEmbeddings():
    def __init__(self, data):
        self.data = data

    def createGraph(self):
        logger.info("Creating graph")
        G = nx.Graph()
        # Add edges to the graph
        for row in self.data:
            for i in range(len(row) - 1):
                G.add_edge(row[i], row[i + 1])
        return G

    def forwardNode2Vec(self):
        logger.info("Node2Vec model")
        G = self.createGraph()
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
        node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
        return node2vec_model

    def forwardDeepWalk(self):
        logger.info("DeepWalk model")
        G = self.createGraph()
        deepwalk = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, p=1, q=1)
        deepwalk_model = deepwalk.fit(window=10, min_count=1, batch_words=4)
        return deepwalk_model

    def forwardWord2Vec(self):
        logger.info("Word2Vec model")
        sentences = [[word for word in row] for row in self.data.tolist()]
        word2vec_model = Word2Vec(sentences, vector_size=64, window=10, min_count=1, workers=4)
        return word2vec_model

    def createEmbeddigns(self, model):
        logger.info("Creating embeddings")
        embeddings = []
        for i in self.data:
            embeddings.append(np.mean(model.wv[i], axis=0).tolist())
        return embeddings

