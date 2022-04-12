from typing import List
from pandas import DataFrame
from matplotlib import pyplot as plt
from utils import read_data, get_subset_by_variable, VARIABLES_WITH_UNFIXED_RANGE
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
    :param bins:
    """
    subset = get_subset_by_variable(variable_name, df)
    plt.hist(x=subset['value'].to_list(), bins=bins)
    plt.title(f"Values of '{variable_name}'")
    plt.show()


def print_unique_values(key, df):
    print(f"Unique values for variable='{key}': {get_subset_by_variable(key, df)['value'].unique()}")


data = read_data()

unique_users = list(data['id'].unique())
records_per_user = {}
for user_id in unique_users:
    records_per_user[user_id] = len(data[data.id == user_id])
print(f"There are {len(unique_users)} unique users with number of records "
      f"mu={np.mean(list(records_per_user.values()))}, "
      f"sigma={np.var(list(records_per_user.values()))}")

print_unique_values('mood', data)
print_unique_values('activity', data)
print_unique_values('circumplex.arousal', data)
print_unique_values('circumplex.valence', data)

print_values_bar('mood', data)
print_values_bar('circumplex.arousal', data)
print_values_bar('circumplex.valence', data)
print_values_hist('activity', data, list(np.linspace(0, 1, 20)))

for variable_name in VARIABLES_WITH_UNFIXED_RANGE:
    variable_subset = get_subset_by_variable('screen', data)['value']
    min_value = min(variable_subset)
    max_value = max(variable_subset)

    print_values_hist(variable_name, data, list(np.linspace(min_value, max_value, 100)))

print(f"Occurences of variable='call': {len(get_subset_by_variable('call', data))}")
print(f"Occurences of variable='sms': {len(get_subset_by_variable('sms', data))}")
