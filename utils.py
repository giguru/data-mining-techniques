import pandas
from pandas import DataFrame


def read_data(**kwargs):
    return pandas.read_csv('dataset_mood_smartphone.csv', **kwargs)


def get_subset_by_variable(variable_name: str, df: DataFrame):
    return df[df['variable'] == variable_name]
