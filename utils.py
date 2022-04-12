import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Dict, Callable, Union
from tqdm import tqdm
import math


__all__ = [
    'SECONDS_IN_DAY', 'VARIABLES_WITH_UNFIXED_RANGE', 'read_data', 'get_temporal_records',
    'get_subset_by_variable', 'fill_defaults'
]


SECONDS_IN_DAY = 3600*24

# When building a features-target combination, some attributes may not have any data. If that's the case, this var is used
MISSING_VALUE = 0

VARIABLES_WITH_UNFIXED_RANGE = [
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


def read_data(**kwargs):
    dtypes = {}
    df = pd.read_csv('dataset_mood_smartphone.csv', dtype=dtypes, parse_dates=['time'], **kwargs)

    # Added timestamp for computational optimization
    df['timestamp'] = df['time'].values.astype(np.int64) // 10 ** 9  # divide by 10^9, because value is in nanoseconds

    # Added day of the week, because people have biases towards e.g. Mondays
    df['week_day'] = df['time'].dt.dayofweek

    invalid_rows = []
    indices_to_drop = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Removing records with invalid values"):
        if math.isnan(row['value']):
            indices_to_drop.append(index)
            invalid_rows.append((index, row['variable'], row['value']))
            # Do not add rows with invalid values
            continue

    df = df.drop(indices_to_drop)
    print(invalid_rows)
    print(f"There are {len(invalid_rows)} invalid rows for {list({item[1] for item in invalid_rows})}")
    return df


def get_subset_by_variable(variable_name: str, df: DataFrame) -> DataFrame:
    return df[df['variable'] == variable_name]


def do_agg_func(agg_func, variable_values):
    return agg_func(variable_values) if len(variable_values) > 0 else MISSING_VALUE


def aggregate_actions_per_user_per_day(df: DataFrame, variable_key, agg_func):
    df_only_variable = df[df['variable'] == variable_key]
    df_aggregate_variable_per_date_and_user = df_only_variable.groupby(
        [df_only_variable['time'].dt.date, df_only_variable['id']]
    ).agg({
        'value': agg_func,
        'timestamp': 'max',
        'time': 'max',
        'id': 'first',  # Retain the value
        'variable': 'first'  # Retain the value
    })

    # Data without the variable column
    df_without_variable = df[df['variable'] != variable_key]
    df = pd.concat([df_aggregate_variable_per_date_and_user, df_without_variable])
    return df, df_aggregate_variable_per_date_and_user


def get_temporal_records(df: DataFrame,
                         history: int,
                         aggregation_actions_per_user_per_day: Dict[str, str] = None,
                         aggregation_actions_total_history: Dict[str, Union[Callable, List[Callable]]] = None
                         ):
    """

    :param df:
    :param history: In seconds
    :param aggregation_actions_per_user_per_day:
    :param aggregation_actions_total_history:
    :return:
    """
    if aggregation_actions_per_user_per_day is None:
        aggregation_actions_per_user_per_day = {}

    # The mean must be always average per day
    aggregation_actions_per_user_per_day['mood'] = 'mean'

    # some variable keys you would like to have average per day
    for variable_key, agg_func in aggregation_actions_per_user_per_day.items():
        df, _ = aggregate_actions_per_user_per_day(df, variable_key, agg_func)

    # Now create one data frame with the mean mood data and the non-mood data
    sorted_df = df.sort_values(by=['timestamp'])

    records = []  # type: List[List[Dict, float, str]]
    running_window = []  # type: List[pd.Series]

    for index, row in tqdm(sorted_df.iterrows(), total=len(sorted_df), desc="Formatting records"):
        # Build a features-target combination
        if row['variable'] == 'mood':
            # Remove records in running list before time frame
            first_index_in_frame = 0
            for idx, running_row in enumerate(running_window):
                if running_row['timestamp'] < row['timestamp'] - history:
                    first_index_in_frame = idx
                else:
                    # Since the list is sorted by time, if you reach an item in the
                    # frame, you can assume the rest is also in the frame.
                    break

            if first_index_in_frame > 0:
                running_window = running_window[first_index_in_frame:]

            # Only build it when there is data for the features.
            if len(running_window) > 0:
                # Other variables you would like to have aggregated over the entire time period
                if aggregation_actions_total_history is None:
                    # Only append data of the user, but remove the ID.
                    # The ID is irrelevant for every machine learning model.
                    features = [r.drop('id') for r in running_window if r['id'] == row['id']]
                else:
                    features = {key: [] for key in aggregation_actions_total_history.keys()}
                    for r in running_window:
                        if r['id'] == row['id'] and r['variable'] in aggregation_actions_total_history:
                            features[r['variable']].append(r['value'])

                    features = dict(features)
                    keys = list(features.keys())
                    for variable_key in keys:
                        variable_values = features[variable_key]
                        del features[variable_key]

                        agg_func = aggregation_actions_total_history[variable_key]
                        if callable(agg_func):
                            name = getattr(agg_func, '__name__')
                            features[f"{variable_key}_{name}"] = do_agg_func(agg_func, variable_values)
                        elif type(agg_func) == list:
                            for func in agg_func:
                                name = getattr(func, '__name__')
                                features[f"{variable_key}_{name}"] = do_agg_func(func, variable_values)

                records.append([
                    features,
                    row['value'],  # the target
                    row['id']
                ])

        running_window.append(row)
    return records

def fill_defaults(items: List, wanted_count: int, default_value):
    if len(items) < wanted_count:
        return items + [default_value] * (wanted_count - len(items))
    else:
        return items