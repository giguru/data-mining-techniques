from typing import List
from pandas import DataFrame
from matplotlib import pyplot as plt
from utils import read_data, get_subset_by_variable, get_temporal_records, SECONDS_IN_DAY
import numpy as np


def print_values_bar(variable_name: str, df: DataFrame):
    """
    :param variable_name: Variable name of the variable column in the CSV
    :param df:
    """
    subset = get_subset_by_variable(variable_name, df)
    count_dict = subset['value'].value_counts().to_dict()
    plt.bar(
        list(count_dict.keys()),
        list(count_dict.values()),
    )
    plt.title(f"Values of '{variable_name}'")
    plt.show()


def print_values_hist(variable_name: str, df: DataFrame, bins: List):
    """
    :param variable_name: Variable name of the variable column in the CSV
    :param df:
    """
    subset = get_subset_by_variable(variable_name, df)
    plt.hist(x=subset['value'].to_list(), bins=bins)
    plt.title(f"Values of '{variable_name}'")
    plt.show()


data = read_data()

get_temporal_records(data, SECONDS_IN_DAY)

print_values_bar('mood', data)
print_values_bar('circumplex.arousal', data)
print_values_bar('circumplex.valence', data)
print_values_hist('activity', data, list(np.linspace(0, 1, 20)))

# The range of screen time is variable
variables_with_unfixed_range = [
    'screen',
    'appCat.builtin',
    'appCat.communication',
    'appCat.entertainment',
    'appCat.finance',
    'appCat.game',
    'appCat.office',
    'appCat.other',
    'appCat.social',
    'appCat.travel',
    'appCat.unknown',
    'appCat.utilities',
    'appCat.weather'
]
for variable_name in variables_with_unfixed_range:
    variable_subset = get_subset_by_variable('screen', data)['value']
    min_value = min(variable_subset)
    max_value = max(variable_subset)

    print_values_hist(variable_name, data, list(np.linspace(min_value, max_value, 100)))

print(f"Occurences of variable='call': {len(get_subset_by_variable('call', data))}")
print(f"Occurences of variable='sms': {len(get_subset_by_variable('sms', data))}")