import pandas as pd
from utils import process_data, read_data, VARIABLES_WITH_UNFIXED_RANGE, mean, \
    check_existing_folder, create_temporal_input, fill_defaults, keep_per_day, aggregate_by_day
import numpy as np
import os
from matplotlib.pyplot import figure
figure(figsize=(20, 20), dpi=80)


def normalize(means):
    computed_std = np.std(means)
    std = computed_std if computed_std > 0 else 1
    return np.mean(means) / std


OUTPUT_PATH = './output/'

DEFAULT_CALL = 0
DEFAULT_SMS = 0
DEFAULT_AROUSAL = 0
DEFAULT_VALENCE = 1
DEFAULT_MOOD = 7.0

MOOD_INDEX = -2
ID_INDEX = -1
DATE_INDEX = -3
N_NON_FEATURES = len([MOOD_INDEX, ID_INDEX, DATE_INDEX])

N_DAY_WINDOW = 3

data = read_data()
aggregation_actions_per_user_per_day = {
    'mood': 'mean',  # The mean must be always average per day
    'circumplex.arousal': mean,
    'circumplex.valence': mean,
    'activity': sum,
    'sms': sum,
    'call': sum,
    **({key: [mean, sum, len] for key in VARIABLES_WITH_UNFIXED_RANGE})
}
records = process_data(data,
                       N_DAY_WINDOW,
                       aggregation_actions_per_user_per_day=aggregation_actions_per_user_per_day,
                       aggregation_actions_total_history=None,
                       rename_variable_per_day=True)

print("example records", records[0])
for i in range(len(records)):
    records[i][0] = aggregate_by_day(records[i][0],
                                     records[i][DATE_INDEX],
                                     N_DAY_WINDOW,
                                     defaults={
                                          'circumplex.arousal': DEFAULT_AROUSAL,
                                          'circumplex.valence': DEFAULT_VALENCE,
                                          'activity': 0,
                                          'call': DEFAULT_CALL,
                                          'sms': DEFAULT_SMS,
                                          'mood': DEFAULT_MOOD,
                                          **({key: 0 for key in VARIABLES_WITH_UNFIXED_RANGE})
                                     })


# TODO Vincenzo: Decide normalisation constants using training set only


# TODO Vincenzo: Feature selection using training set only: e.g. PCA

# Create temporal dataset
X_train, y_train, X_test, y_test = create_temporal_input(records,
                                                         mood_index=MOOD_INDEX,
                                                         id_index=ID_INDEX,
                                                         min_sequence_len=10)


# TODO Bram: train a temporal model, e.g. LSTM, RNN, etc.


# TODO Giguru: compute two base lines. Simply take the mood the day before and take the average mood.


# TODO Evaluation: confusion matrix, mean squared error, qualitative prediction power per user.

