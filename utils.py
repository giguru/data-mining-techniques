import pandas


def read_data(**kwargs):
    return pandas.read_csv('dataset_mood_smartphone.csv', **kwargs)
