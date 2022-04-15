import pandas as pd
from utils import process_data, read_data, VARIABLES_WITH_UNFIXED_RANGE, mean, \
    check_existing_folder, create_temporal_input, fill_defaults, keep_per_day
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

N_DAY_WINDOW = 1
save_file_path = os.path.join(OUTPUT_PATH, f'feature_temporal_{N_DAY_WINDOW}.csv')

if os.path.exists(save_file_path):
    df = pd.read_csv(save_file_path)
    feature_labels = df.columns.tolist()[:-N_NON_FEATURES]
    feature_matrix = np.array(df.values.tolist())
else:
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
                           aggregation_actions_total_history=None)
    print("example records", records[0])
    feature_matrix = np.array([list(r[0].values()) + r[1:] for r in records])

    # Save data frame
    feature_labels = list(aggregation_actions_per_user_per_day.keys())
    df = pd.DataFrame(feature_matrix, columns=feature_labels + ['date', 'mood', 'id'])
    check_existing_folder(OUTPUT_PATH)
    df.to_csv(save_file_path, index=False)


# TODO Vincenzo: Decide normalisation constants using training set only


# TODO Vincenzo: Feature selection using training set only: e.g. PCA

# Create temporal dataset
X_train, y_train, X_test, y_test = create_temporal_input(feature_matrix,
                                                         mood_index=MOOD_INDEX,
                                                         id_index=ID_INDEX,
                                                         min_sequence_len=10)


# TODO Bram: train a temporal model, e.g. LSTM, RNN, etc.


# TODO Giguru: compute two base lines. Simply take the mood the day before and take the average mood.


# TODO Evaluation: confusion matrix, mean squared error, qualitative prediction power per user.

