import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Dict
from tqdm import tqdm


SECONDS_IN_DAY = 3600*24


def read_data(**kwargs):
    dtypes = {}
    df = pd.read_csv('dataset_mood_smartphone.csv', dtype=dtypes, parse_dates=['time'],**kwargs)

    df['timestamp'] = df['time'].values.astype(np.int64) // 10 ** 9 # divide by 10^9, because the value is in nanoseconds
    return df


def get_subset_by_variable(variable_name: str, df: DataFrame):
    return df[df['variable'] == variable_name]


def get_temporal_records(df: DataFrame, history: int, aggregation_actions: Dict[str, str] = None):
    """

    :param df:
    :param history: In seconds
    :return:
    """
    if aggregation_actions is None:
        aggregation_actions = {}
    # The mean must be always average
    aggregation_actions['mood'] = 'mean'

    for variable_key, agg_func in aggregation_actions.items():
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

    # Now create one data frame with the mean mood data and the non-mood data
    sorted_df = df.sort_values(by=['timestamp'])

    records = []
    running_window = []  # type: List[pd.Series]

    for index, row in tqdm(sorted_df.iterrows(), total=len(sorted_df), desc="Formatting records"):
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

            records.append([
                # Only append data of the user
                [r for r in running_window if r['id'] == row['id']],
                # the target
                row['value']
            ])

        running_window.append(row)
    return records
