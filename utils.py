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


def get_temporal_records(df: DataFrame, history: int):
    """

    :param df:
    :param history: In seconds
    :return:
    """
    sorted_df = df.sort_values(by=['timestamp'])
    records = []

    running_list = []  # type: List[Dict]
    for index, row in tqdm(sorted_df.iterrows(), total=len(sorted_df), desc="Formatting records"):
        if row['variable'] == 'mood':
            # Remove records in running list before time frame
            first_index_in_frame = 0
            for idx, running_row in enumerate(running_list):
                if running_row['timestamp'] < row['timestamp'] - history:
                    first_index_in_frame = idx
                else:
                    # Since the list is sorted by time, if you reach an item in the
                    # frame, you can assume the rest is also in the frame.
                    break
            if first_index_in_frame > 0:
                running_list = running_list[first_index_in_frame:]

            records.append([
                # Only append data of the user
                [r for r in running_list if r['id'] == row['id']],
                row['value']
            ])

        running_list.append(row)
    return records
