from collections import deque
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

def read_file(path):
    lines = []
    with open(path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def parse_line(line):
    # Split the line by the delimiter -1
    sequences = line.replace(" ","").split("-1")
    # Remove the last element (which is just "-2\n")
    sequences = sequences[:-1]
    
    return sequences

def create_dataset(path, max_len):
    lines = read_file(path)
    parsed_data = []
    for line in lines:
        parsed_data.append([l[7:10] for l in parse_line(line)])
    queue = deque([0, 0, 0, 0, 0, 0, 0], maxlen=max_len)
    padded_sequences = []
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(parsed_data)
    sequences = tokenizer.texts_to_sequences(parsed_data)
    print(sequences[0])
    padded_sequences = pad_sequences(sequences, padding='pre', maxlen=4)
    # for i in sequences:
    #    for j in i:
    #        queue.append(j)
    #        padded_sequences.append(list(queue))
    #    queue = deque([0, 0, 0, 0, 0, 0, 0], maxlen=4)
    print(padded_sequences[:5])
    # Split data into features and labels (if applicable)
    X = np.array([seq[:-1] for seq in padded_sequences])  # Example only, adjust as per your actual data
    y = np.array([seq[-1] for seq in padded_sequences])  # Example only, adjust as per your actual data

    return X, y
