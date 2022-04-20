from utils import read_data, VARIABLES_WITH_UNFIXED_RANGE, mean, \
    create_temporal_input, \
    aggregate_actions_per_user_per_day, dataframe_to_dict_per_day, get_normalising_constants, \
    apply_normalisation_constants, compute_baseline_metrics
from matplotlib.pyplot import figure

figure(figsize=(20, 20), dpi=80)

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

# TODO Vincenzo: Feature selection using training set only: e.g. PCA

# Create temporal dataset

X_train, y_train, X_test, y_test = create_temporal_input(per_user_per_day_index, min_sequence_len=10)
print("Example input: ", X_train[0])
print("Example target: ", y_train[0])

# normalize X_train, X_test
normalisation_constants = get_normalising_constants(data_df)
X_train = apply_normalisation_constants(X_train, normalisation_constants)
X_test = apply_normalisation_constants(X_test, normalisation_constants)

# Try to predict how much a user deviate from the USER's mean. So shift and shift back

# TODO Bram: train a temporal model, e.g. LSTM, RNN, etc.


# Compute two baseline. Simply take the mood the day before and take the average mood.
predictions_last_mood_train = [r[len(r)-1]['mood_mean'] for r in X_train]
compute_baseline_metrics(y_true=y_train,
                         y_pred=predictions_last_mood_train,
                         title="train data")

predictions_last_mood_test = [r[len(r)-1]['mood_mean'] for r in X_test]
compute_baseline_metrics(y_true=y_train,
                         y_pred=predictions_last_mood_test,
                         title="test data")

# TODO Evaluation: qualitative prediction power per user.
