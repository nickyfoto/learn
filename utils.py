import pandas as pd

def load_data(fn, path='./datasets/', sep=','):
    data = pd.read_csv(path + fn, sep=sep)
    return data