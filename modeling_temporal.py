from utils import read_data, VARIABLES_WITH_UNFIXED_RANGE, mean, \
    create_temporal_input, \
    aggregate_actions_per_user_per_day, dataframe_to_dict_per_day, get_normalising_constants, \
    apply_normalisation_constants, compute_metrics, get_selected_attributes, convert_to_list
from matplotlib.pyplot import figure
import torch
from model_classes import LSTM
from tqdm import tqdm


figure(figsize=(20, 20), dpi=80)

OUTPUT_PATH = './output/'

DEFAULT_CALL = 0
DEFAULT_SMS = 0
DEFAULT_AROUSAL = 0
DEFAULT_VALENCE = 1
DEFAULT_MOOD = 7.0
DEFAULT_SCREENREST = 0
N_EPOCHS = 900

data_df = read_data()
aggregation_actions_per_user_per_day = {
    'mood': 'mean',  # The mean must be always average per day
    'screenrest': [mean, len],
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

per_user_per_day_index = dataframe_to_dict_per_day(
    data_df,
    default_callables={
        'mood_mean': lambda current, prev: current or prev or DEFAULT_MOOD,
        'circumplex.arousal_mean': lambda current, prev: current or DEFAULT_AROUSAL,
        'circumplex.valence_mean': lambda current, prev: current or DEFAULT_VALENCE,
        'activity_sum': lambda current, prev: current or 0,
        'call_sum': lambda current, prev: current or DEFAULT_CALL,
        'sms_sum': lambda current, prev: current or DEFAULT_SMS,
        'screenrest_mean': lambda current, prev: current or DEFAULT_SCREENREST,
        'screenrest_len': lambda current, prev: current or DEFAULT_SCREENREST,
        **({f"{key}_mean": lambda current, prev: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE}),
        **({f"{key}_sum": lambda current, prev: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE}),
        **({f"{key}_len": lambda current, prev: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE})
    },
    keep_features=get_selected_attributes(OUTPUT_PATH)
)

# Create temporal dataset
X_train, y_train, X_test, y_test = create_temporal_input(per_user_per_day_index, min_sequence_len=10)
print("Example input: ", X_train[0])
print("Example target: ", y_train[0])

# normalize X_train, X_test
normalisation_constants = get_normalising_constants(data_df)
X_train = apply_normalisation_constants(X_train, normalisation_constants)
X_test = apply_normalisation_constants(X_test, normalisation_constants)

# convert dicts to lists of floats
X_train = convert_to_list(X_train)
X_test = convert_to_list(X_test)

# Try to predict how much a user deviate from the USER's mean. So shift and shift back
lstm = LSTM()

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

# Train the model
for epoch in range(N_EPOCHS):
    for user_input_data, user_target_data in tqdm(zip(X_train, y_train), desc=f"Epoch {epoch}"):
        _, inputs = user_input_data
        _, targets = user_target_data

        for input, target in zip(inputs, targets):
            trainX = torch.tensor(input)
            trainY = torch.tensor(target)
            outputs = lstm(trainX)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, trainY)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

print(lstm(X_train))
print(y_train)

# Compute two baseline. Simply take the mood the day before and take the average mood.
predictions_last_mood_train = [r[len(r)-1]['mood_mean'] for r in X_train]
compute_metrics(y_true=y_train,
                y_pred=predictions_last_mood_train,
                title="train data")

predictions_last_mood_test = [r[len(r)-1]['mood_mean'] for r in X_test]
compute_metrics(y_true=y_train,
                y_pred=predictions_last_mood_test,
                title="test data")

# TODO Evaluation: qualitative prediction power per user.
