from utils import read_data, VARIABLES_WITH_UNFIXED_RANGE, mean, \
    create_temporal_input, \
    aggregate_actions_per_user_per_day, dataframe_to_dict_per_day
import numpy as np
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

data_df = read_data()
aggregation_actions_per_user_per_day = {
    'mood': 'mean',  # The mean must be always average per day
    'circumplex.arousal': mean,
    'circumplex.valence': mean,
    'activity': sum,
    'sms': sum,
    'call': sum,
    **({key: [mean, sum, len] for key in VARIABLES_WITH_UNFIXED_RANGE})
}

for variable_key, agg_func in aggregation_actions_per_user_per_day.items():
    data_df = aggregate_actions_per_user_per_day(data_df, variable_key, agg_func, rename_variable=True)

data_df.sort_values(by=['timestamp'], inplace=True)

per_user_per_day_index = dataframe_to_dict_per_day(data_df,
                                                   default_callables={
                                        'mood_mean': lambda current, prev: current or prev or DEFAULT_MOOD,
                                        'circumplex.arousal_mean': lambda current, prev: current or DEFAULT_AROUSAL,
                                        'circumplex.valence_mean': lambda current, prev: current or DEFAULT_VALENCE,
                                        'activity_sum': lambda current, prev: current or 0,
                                        'call_sum': lambda current, prev: current or DEFAULT_CALL,
                                        'sms_sum': lambda current, prev: current or DEFAULT_SMS,
                                        **({f"{key}_mean": lambda current, prev: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE}),
                                        **({f"{key}_sum": lambda current, prev: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE}),
                                        **({f"{key}_len": lambda current, prev: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE})
                                    })

# TODO Vincenzo: Decide normalisation constants using training set only


# TODO Vincenzo: Feature selection using training set only: e.g. PCA

# Create temporal dataset
per_user = { # TODO for normalisation
    'user A': [ # only training records
        {'featureA': ...},
        {'featureA': ...},
        {'featureA': ...},
        {'featureA': ...},
    ]
}
X_train, y_train, X_test, y_test = create_temporal_input(per_user_per_day_index, min_sequence_len=10)
print("Example input: ", X_train[0])
print("Example target: ", y_train[0])

# TODO Bram: train a temporal model, e.g. LSTM, RNN, etc.


# Compute two base lines. Simply take the mood the day before and take the average mood.
predictions_last_mood_train = [r[len(r)-1]['mood_mean'] for r in X_train]
mood_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
score_last_mood_train = r2_score(y_true=y_train, y_pred=predictions_last_mood_train)
cm = confusion_matrix(y_true=[round(v) for v in y_train],
                      y_pred=[round(v) for v in predictions_last_mood_train],
                      labels=mood_labels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mood_labels).plot()
plt.title("Baselines - train data")
plt.show()

predictions_last_mood_test = [r[len(r)-1]['mood_mean'] for r in X_test]
score_last_mood_test = r2_score(y_true=y_test, y_pred=predictions_last_mood_test)
cm = confusion_matrix(y_true=[round(v) for v in y_test],
                      y_pred=[round(v) for v in predictions_last_mood_test],
                      labels=mood_labels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mood_labels).plot()
plt.title("Baselines - test data")
plt.show()

# TODO Evaluation: qualitative prediction power per user.
