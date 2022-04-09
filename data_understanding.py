from typing import List
from pandas import DataFrame
from matplotlib import pyplot as plt
from utils import read_data, get_subset_by_variable, get_temporal_records, SECONDS_IN_DAY, VARIABLES_WITH_UNFIXED_RANGE
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import os

OUTPUT_PATH = './output/'

def check_existing_folder(this_path):
    MYDIR = (this_path)
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)


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


def plot_temporal_effects(df: DataFrame, time_window: str, variable_name: str, **kwargs):
    """
    :param df: DataFrame with time as index and variable_name as value.
                Example: t=0,1,...,24 (hour)
                         mood = 2,3,...,6
    :param time_window: label of horizontal axis. time window considered.
                        E.g. hour  (during the day), date,...
    :param variable_name: mood, or any other attribute
    :return: Plots  the time series
    """
    this_path = os.path.join(OUTPUT_PATH, 'temporal_dependency')
    check_existing_folder(this_path)
    plt.plot(df.index,
             df['skipna_mean'],
             label=time_window
    )
    plt.fill_between(df.index, df['lower'], df['upper'], alpha=0.2)
    plt.title(f"Time evolution mean of '{variable_name}'")
    plt.xlabel(time_window)
    plt.ylabel(variable_name)
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig(os.path.join(this_path, f'{time_window}.pdf'), bbox_inches='tight')
    plt.close()

data = read_data()

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

#
# check periodicity
data['hour'] = data.time.dt.hour
data['date'] = data.time.dt.date
data['weekday'] = data.time.dt.strftime('%A')
data['month'] = data.time.dt.month
data['year'] = data.time.dt.year

item = 'mood'

def skipna_mean(df):
    return df.mean(skipna=True)
cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def skipna_std(df):
    return df.std(skipna=True)/np.sqrt(len(df)-sum(df.isna()))

for time_window  in  ['hour', 'date', 'weekday', 'month']:
    this_mean = data.loc[data['variable'] == item].groupby(by=[time_window])['value'].agg([skipna_mean,
                                                                                           skipna_std])
    this_mean['lower'] = this_mean['skipna_mean'] - 1.96 * this_mean['skipna_std']
    this_mean['upper'] = this_mean['skipna_mean'] + 1.96 * this_mean['skipna_std']

    if time_window == 'weekday':
        this_mean = this_mean.reindex(cats)
    plot_temporal_effects(this_mean, time_window,  'mood')
