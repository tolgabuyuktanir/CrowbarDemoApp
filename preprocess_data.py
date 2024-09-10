import pandas as pd

lines = []

def read_file():
    with open('/kaggle/input/ieee-bigdata-demai/dataset_pageid', 'r') as file:
        lines = [line.strip() for line in file.readlines()]


def parse_line(line):
    # Split the line by the delimiter -1
    sequences = line.replace(" ","").split("-1")
    
    # Remove the last element (which is just "-2\n")
    sequences = sequences[:-1]
    
    return sequences

def create_data_frame(lines):
    parsed_data = []
    for line in lines:
        parsed_data.extend(parse_line(line))
    return pd.DataFrame(parsed_data, columns=["pid"])

df = create_data_frame(lines)