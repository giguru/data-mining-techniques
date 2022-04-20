from collections import defaultdict
import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Dict, Callable, Union, Any
from tqdm import tqdm
from datetime import datetime
import math
from copy import copy
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


__all__ = [
    'SECONDS_IN_DAY', 'VARIABLES_WITH_UNFIXED_RANGE', 'read_data', 'process_data',
    'get_subset_by_variable', 'fill_defaults', 'keep_per_day', 'mean', 'check_existing_folder',
    'create_temporal_input', 'dataframe_to_dict_per_day', 'MAX_ATTRIBUTE', 'get_normalising_constants',
    'FIXED_STD_ATTRIBUTE', 'apply_normalisation_constants', 'compute_baseline_metrics'
]

DATE_FORMAT = '%Y-%m-%d'
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
    'appCat.weather',
    'amount_screen',
    'screenrest'
]
FIXED_STD_ATTRIBUTE = ['mood_scaled', 'mood_before_mean', 'screen_mean', 'screen_len', 'amount_screen_mean',
                       'amount_screen_len',
                       'screenrest_mean', 'screenrest_len']
MAX_ATTRIBUTE = ['circumplex.arousal_custom', 'circumplex.valence_custom', 'activity_mean', 'call_custom', 'sms_custom',
                 'appCat.builtin_mean', 'appCat.builtin_len', 'appCat.communication_mean', 'appCat.communication_len',
                 'appCat.entertainment_mean',
                 'appCat.entertainment_len', 'appCat.finance_mean', 'appCat.finance_len', 'appCat.game_mean',
                 'appCat.game_len', 'appCat.office_mean', 'appCat.office_len', 'appCat.other_mean', 'appCat.other_len',
                 'appCat.social_mean', 'appCat.social_len', 'appCat.travel_mean', 'appCat.travel_len',
                 'appCat.unknown_mean', 'appCat.unknown_len', 'appCat.utilities_mean', 'appCat.utilities_len',
                 'appCat.weather_mean', 'appCat.weather_len']


def compute_baseline_metrics(y_true, y_pred, title: str, scaled: bool = False):
    mood_labels = [-3, -2, -1, 0, 1, 2, 3] if scaled else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"r2 - {title}", r2_score(y_true=y_true, y_pred=y_pred))
    cm = confusion_matrix(y_true=[round(v) for v in y_true],
                          y_pred=[round(v) for v in y_pred],
                          labels=mood_labels)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mood_labels).plot()
    plt.title(f"Baselines - {title}")
    plt.show()


def apply_normalisation_constants(X_train, normalisation_constants: Dict[str, Dict[str, Dict[str, float]]]):
    for idx in range(len(X_train)):
        user_id, batched_user_input_records = X_train[idx]
        for batch_input_records in batched_user_input_records:
            for record in batch_input_records:
                for feature_key, feature_value in record.items():
                    if feature_key in MAX_ATTRIBUTE:
                        new_value = feature_value / normalisation_constants[user_id][feature_key]['max']
                        record[feature_key] = new_value
                    elif feature_key in FIXED_STD_ATTRIBUTE:
                        new_value = (feature_value - normalisation_constants[user_id][feature_key]['mean']) / normalisation_constants[user_id][feature_key]['std']
                        record[feature_key] = new_value


def read_data(**kwargs):
    dtypes = {}
    df = pd.read_csv('newdata.csv', dtype=dtypes, parse_dates=['time'], **kwargs)

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


def mean(values_list):
    """
    Custom mean function that defaults to zero if there are no values.
    This function was necessary, because np.mean crashes when there are no values.
    """
    return np.mean(values_list) if len(values_list) > 0 else 0


def build_extended_key(variable_key: str, agg_func):
    name = getattr(agg_func, '__name__') if callable(agg_func) else agg_func  # Assume it's a string otherwise1q1q
    return f"{variable_key}_{name}"


def aggregate_actions_per_user_per_day(df: DataFrame, variable_key, agg_func, rename_variable:bool=False):
    df_only_variable = df[df['variable'] == variable_key]
    df_without_variable = df[df['variable'] != variable_key]

    func_list = agg_func if type(agg_func) == list else [agg_func]

    for func in func_list:
        df_aggregate_variable_per_date_and_user = df_only_variable.groupby(
            [df_only_variable['time'].dt.date, df_only_variable['id']]
        ).agg({
            'value': func,
            'timestamp': 'max',
            'time': 'max',
            'id': 'first',  # Retain the value
            'variable': 'first',  # Retain the value
            'week_day': 'first'  # Retain the value
        })

        if rename_variable:
            df_aggregate_variable_per_date_and_user.replace(
                # Replace e.g. 'mean' with 'mean_mood'
                to_replace={variable_key: build_extended_key(variable_key, func)},
                inplace=True
            )
        df = pd.concat([df_aggregate_variable_per_date_and_user, df_without_variable])
    return df


def keep_per_day(default: float):
    def inner_keep_per_day(values_per_day: Dict[str, Any], row: Dict[str, Any], day_window: int, prefix: str):
        result_dict = {}
        for days_go in range(1, day_window + 1):
            date_ago = datetime.fromtimestamp(row['timestamp'] - days_go * SECONDS_IN_DAY).strftime(DATE_FORMAT)
            value_to_use = values_per_day[date_ago] if date_ago in values_per_day else default
            result_dict[f"{prefix}_{days_go}_day_before"] = value_to_use
        return result_dict
    return inner_keep_per_day


def start_of_day(timestamp: int):
    """
    You want to aggregate over the data of the entire day.
    """
    dt_object = datetime.fromtimestamp(timestamp).replace(hour=3,
                                                          minute=0)
    return datetime.timestamp(dt_object)


def to_date_string(date_object: datetime):
    return date_object.strftime(DATE_FORMAT)


def process_data(df: DataFrame,
                 day_window: int,
                 aggregation_actions_per_user_per_day: Dict[str, str] = None,
                 aggregation_actions_total_history: Dict[str, Union[Callable, List[Callable]]] = None,
                 rename_variable_per_day: bool = False
                 ):
    """

    :param df:
    :param day_window: Enter the number of days
    :param aggregation_actions_per_user_per_day:
    :param aggregation_actions_total_history:
    :return:
    """
    if aggregation_actions_per_user_per_day is None:
        aggregation_actions_per_user_per_day = {}

    # some variable keys you would like to have average per day
    for variable_key, agg_func in aggregation_actions_per_user_per_day.items():
        df = aggregate_actions_per_user_per_day(df, variable_key, agg_func, rename_variable=rename_variable_per_day)

    # Now create one data frame with the mean mood data and the non-mood data
    df.sort_values(by=['timestamp'], inplace=True)

    records = []  # type: List[List[Dict, float, str]]
    running_window = []  # type: List[pd.Series]

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Formatting records"):
        # Build a features-target combination for each mood. The mean variable may have been rewritten in e.g. mood_mean
        if row['variable'] == 'mood' or 'mood' in row['variable']:
            # Remove records in running list before time frame
            first_index_in_frame = 0
            for idx, running_row in enumerate(running_window):
                if running_row['timestamp'] < start_of_day(row['timestamp'] - day_window * SECONDS_IN_DAY):
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
                    features = [r.drop('id').to_dict() for r in running_window if r['id'] == row['id']]
                else:
                    features = {key: [] for key in aggregation_actions_total_history.keys()}
                    for r in running_window:
                        if r['id'] == row['id'] and r['variable'] in aggregation_actions_total_history:
                            features[r['variable']].append(r)

                    features = dict(features)
                    keys = list(features.keys())
                    for variable_key in keys:
                        agg_func = aggregation_actions_total_history[variable_key]
                        if callable(agg_func) and getattr(agg_func, '__name__') == 'inner_keep_per_day':
                            values_per_day = {to_date_string(r['time']): r['value'] for r in features[variable_key]}
                            dict_per_day = agg_func(values_per_day=values_per_day,
                                                    row=row,
                                                    day_window=day_window,
                                                    prefix=variable_key)
                            features = {**features, **dict_per_day}
                        elif callable(agg_func) and getattr(agg_func, '__name__') == '<lambda>':
                            variable_values = [r['value'] for r in features[variable_key]]
                            features[f"{variable_key}_custom"] = agg_func(variable_values, row)
                        elif callable(agg_func):
                            extended_key = build_extended_key(variable_key, agg_func)
                            variable_values = [r['value'] for r in features[variable_key]]
                            features[extended_key] = agg_func(variable_values)
                        elif type(agg_func) == list:
                            variable_values = [r['value'] for r in features[variable_key]]
                            for func in agg_func:
                                extended_key = build_extended_key(variable_key, func)
                                features[extended_key] = func(variable_values)

                        del features[variable_key]
                records.append([
                    features,
                    to_date_string(row['time']),
                    row['value'],  # the target
                    row['id'],
                ])

        running_window.append(row)
    return records


def get_normalising_constants(df):

    per_user_per_feature = {}
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Creating aggregated index for normalisation..."):
        variable, user_id = r['variable'], r['id']
        if user_id not in per_user_per_feature:
            per_user_per_feature[user_id] = defaultdict(list)

        per_user_per_feature[user_id][variable].append(r['value'])

    user_feature_normalisation_constants = {}
    for user_id, user_features in tqdm(per_user_per_feature.items(),
                                       total=len(per_user_per_feature),
                                       desc="Compute normalisation constants for each user-feature combination"):
        user_feature_normalisation_constants[user_id] = {}

        for feature_key, feature_values in user_features.items():
            computed_std = np.std(feature_values)

            user_feature_normalisation_constants[user_id][feature_key] = {
                'max': max(feature_values),
                'mean': mean(feature_values),
                'std': computed_std if computed_std > 0 else 1
            }
    return user_feature_normalisation_constants


def dataframe_to_dict_per_day(df: DataFrame, default_callables: Dict[str, Callable]):
    """

    :param df:
    :param default_callables:
    :return: returns a dict of the format:
        {
            "user id 1" : {
                "yyyy-mm-dd": {
                    "feature 1": value 1,
                    "feature 2": value 2,
                    ...
                },
                ...
            },
            ...
        }
    """
    per_user_per_day = {}
    for _, r in df.iterrows():
        date = to_date_string(r['time'])
        variable = r['variable']
        user_id = r['id']
        if user_id not in per_user_per_day:
            per_user_per_day[user_id] = {}

        if date not in per_user_per_day[user_id]:
            per_user_per_day[user_id][date] = {}

        assert variable not in per_user_per_day[user_id][date], "The records must have already been aggregated per user per day"

        per_user_per_day[user_id][date][variable] = r['value']

    for user_id, features_per_day in dict(per_user_per_day).items():
        prev_day_features = {}
        for date, found_features_dict in features_per_day.items():
            defaults = {}
            for key, func in default_callables.items():
                defaults[key] = func(
                    found_features_dict[key] if key in found_features_dict else None,
                    prev_day_features[key] if key in prev_day_features else None
                )

            per_user_per_day[user_id][date] = {
                **defaults,
                # Overwrite with found features
                **found_features_dict
            }
            prev_day_features = copy(per_user_per_day[user_id][date])

    return per_user_per_day


def fill_defaults(items: List, wanted_count: int, default_value):
    if len(items) < wanted_count:
        return items + [default_value] * (wanted_count - len(items))
    else:
        return items


def check_existing_folder(this_path):
    my_dir = this_path
    check_folder = os.path.isdir(my_dir)

    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(my_dir)
        print("created folder : ", my_dir)


def create_temporal_input(per_user_per_day: Dict[str, dict],
                          min_sequence_len: int,
                          max_sequence_len: int = 1000,
                          mood_key: str = 'mood_mean',
                          train_size: float = 0.8):
    """
    This method assumes the records are ordered by ascending date.
    """

    total_x_train, total_y_train, total_x_test, total_y_test = [], [], [], []

    for user_id, all_user_records in per_user_per_day.items():
        all_user_records = np.array(list(all_user_records.values()))
        user_inputs, user_targets = [], []

        for current_index in range(min_sequence_len, len(all_user_records)):
            start_input_index = max(0, current_index - max_sequence_len)
            last_input_index = current_index-1

            if mood_key in all_user_records[current_index]:
                input_records = list(all_user_records[start_input_index:last_input_index])
                target = all_user_records[current_index][mood_key]

                user_inputs.append(input_records)
                user_targets.append(target)

        user_targets = np.array(user_targets)
        user_inputs = np.array(user_inputs)
        idx = int(len(user_inputs) * train_size)

        total_x_train.append((user_id, list(user_inputs[:idx])))
        total_y_train.append((user_id, list(user_targets[:idx])))

        total_x_test.append((user_id, list(user_inputs[idx:])))
        total_y_test.append((user_id, list(user_targets[idx:])))

    return total_x_train, total_y_train, total_x_test, total_y_test

