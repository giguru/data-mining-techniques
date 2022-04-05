from typing import List
from pandas import DataFrame
from matplotlib import pyplot as plt
from utils import read_data
import numpy as np


def print_values_bar(variableName: str, df: DataFrame):
    """
    :param variableName: Variable name of the variable column in the CSV
    :param df:
    """
    subset = df[df['variable'] == variableName]
    count_dict = subset['value'].value_counts().to_dict()
    plt.bar(
        list(count_dict.keys()),
        list(count_dict.values()),
    )
    plt.title(f"Values of '{variableName}'")
    plt.show()


def print_values_hist(variableName: str, df: DataFrame, bins: List):
    """
    :param variableName: Variable name of the variable column in the CSV
    :param df:
    """
    subset = df[df['variable'] == variableName]
    plt.hist(x=subset['value'].to_list(), bins=bins)
    plt.title(f"Values of '{variableName}'")
    plt.show()


print_values_bar('mood', read_data())
print_values_bar('circumplex.arousal', read_data())
print_values_bar('circumplex.valence', read_data())
print_values_hist('activity', read_data(), list(np.linspace(0, 1, 20)))