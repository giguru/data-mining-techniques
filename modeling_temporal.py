from utils import read_data, VARIABLES_WITH_UNFIXED_RANGE, mean, \
    create_temporal_input, \
    aggregate_actions_per_user_per_day, dataframe_to_dict_per_day, get_normalising_constants, \
    apply_normalisation_constants, compute_metrics, get_selected_attributes, convert_to_list
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import torch
from model_classes import LSTM
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import mean_squared_error


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

figure(figsize=(20, 20), dpi=80)

OUTPUT_PATH = './output/'

DEFAULT_CALL = 0.
DEFAULT_SMS = 0.
DEFAULT_AROUSAL = 0.
DEFAULT_VALENCE = 1.
DEFAULT_MOOD = 0.
DEFAULT_SCREENREST = 0.


N_EPOCHS = 140
LEARNING_RATE = 1e-3
DROPOUT = 0.2
KEEP_FEATURES = get_selected_attributes(OUTPUT_PATH, ['mood_mean'])
INPUT_SIZE = 20
HIDDEN_SIZE = 10
ACCUM_ITER = 4
print("Lr: ", LEARNING_RATE,
      'Epochs: ', N_EPOCHS,
      "Dropout:", DROPOUT,
      "Hidden size:", HIDDEN_SIZE,
      "Input size:", INPUT_SIZE,
      "Gradient accum.:", ACCUM_ITER,
      "Features (not necessarily used): ", len(KEEP_FEATURES), KEEP_FEATURES)

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

normalisation_constants = get_normalising_constants(data_df)

per_user_per_day_index = dataframe_to_dict_per_day(
    data_df,
    default_callables={
        'mood_mean': lambda current, prev, user_id: current or normalisation_constants[user_id]['mood_mean']['mean'],
        'circumplex.arousal_mean': lambda current, prev, _: current or DEFAULT_AROUSAL,
        'circumplex.valence_mean': lambda current, prev, _: current or DEFAULT_VALENCE,
        'activity_sum': lambda current, prev, _: current or 0,
        'call_sum': lambda current, prev, _: current or DEFAULT_CALL,
        'sms_sum': lambda current, prev, _: current or DEFAULT_SMS,
        'screenrest_mean': lambda current, prev, _: current or DEFAULT_SCREENREST,
        'screenrest_len': lambda current, prev, _: current or DEFAULT_SCREENREST,
        **({f"{key}_mean": lambda current, prev, _: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE}),
        **({f"{key}_sum": lambda current, prev, _: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE}),
        **({f"{key}_len": lambda current, prev, _: current or 0 for key in VARIABLES_WITH_UNFIXED_RANGE})
    },
    keep_features=KEEP_FEATURES
)

# Create temporal dataset
X_train, y_train, X_test, y_test = create_temporal_input(per_user_per_day_index,
                                                         min_sequence_len=10,
                                                         max_sequence_len=10)

# normalize X_train, X_test
X_train = apply_normalisation_constants(X_train, normalisation_constants)
X_test = apply_normalisation_constants(X_test, normalisation_constants)

# convert dicts to lists of floats
feature_labels = X_train[0][1][0][0].keys()
X_train = convert_to_list(X_train)
X_test = convert_to_list(X_test)

print("Example features: ", feature_labels)
print("Example input: ", X_train[0][1][0])
print("Example target: ", y_train[0][1][0])

# Try to predict how much a user deviate from the USER's mean. So shift and shift back
lstm = LSTM(input_size=INPUT_SIZE, dropout_rate=DROPOUT, hidden_size=HIDDEN_SIZE).double()

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
MOOD_INDEX = 0

best_score = 10000
best_model = None

def do_eval(x_inputs, y_test):
    y_pred_test = []
    y_true_test = []
    y_pred_last_mood_test = []
    total_loss = 0
    lstm.eval()
    for user_input_data, user_target_data in tqdm(zip(x_inputs, y_test), total=len(x_inputs), desc=f"Evaluating"):
        test_user_id, inputs = user_input_data
        _, targets = user_target_data
        for input, target in zip(inputs, targets):
            normalised_target = (target - normalisation_constants[test_user_id]['mood_mean']['mean']) / normalisation_constants[test_user_id]['mood_mean']['std']
            input_tensor = torch.tensor([input]).double()
            outputs = lstm(input_tensor)
            pred = float(outputs[0]) * normalisation_constants[test_user_id]['mood_mean']['std'] + normalisation_constants[test_user_id]['mood_mean']['mean']

            y_pred_test.append(pred)
            y_true_test.append(target)
            y_pred_last_mood_test.append(input[-1][MOOD_INDEX] * normalisation_constants[test_user_id]['mood_mean']['std'] + normalisation_constants[test_user_id]['mood_mean']['mean'])

            trainY = torch.tensor([normalised_target]).double()
            total_loss += criterion(outputs[0], trainY)
    return y_pred_test, y_true_test, y_pred_last_mood_test, total_loss


def normalise(t, u_id):
    return (t - normalisation_constants[u_id]['mood_mean']['mean']) / normalisation_constants[u_id]['mood_mean']['std']

# Train the model
train_losses, scores = [], []
for epoch in range(N_EPOCHS):
    lstm.train()
    for user_input_data, user_target_data in tqdm(zip(X_train, y_train), total=len(X_train), desc=f"Epoch {epoch}"):
        user_id, inputs = user_input_data
        _, targets = user_target_data

        for batch_idx, (input, target) in enumerate(zip(inputs, targets)):
            trainX = torch.tensor([input]).double()
            normalised_target = normalise(target, user_id)
            trainY = torch.tensor([normalised_target]).double()
            outputs = lstm(trainX)[0]

            # obtain the loss function
            loss = criterion(outputs, trainY)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if epoch % 10 == 0:
        y_pred_eval, y_true_eval, _, total_loss = do_eval(X_train, y_train)
        score = mean_squared_error(y_true=y_true_eval, y_pred=y_pred_eval)
        print("Loss: ", total_loss.item(), " Score:", score)
        train_losses.append(total_loss.item())
        scores.append(score)
        if score < best_score:
            best_model = lstm.state_dict()

# load best model
lstm.load_state_dict(best_model)

plt.plot(train_losses)
plt.title("Training loss")
plt.show()

plt.plot(scores)
plt.title("Training MSE score")
plt.show()

y_pred_test, y_true_test, y_pred_last_mood_test, _ = do_eval(X_test, y_test)
y_pred_train, y_true_train, _, _ = do_eval(X_train, y_train)

print("Example y_true: ", y_true_test[:10])
print("Example y_pred: ", y_pred_test[:10])
compute_metrics(y_true=y_true_test,
                y_pred=y_pred_test,
                scaled=False,
                title="temporal test data")

compute_metrics(y_true=y_true_train,
                y_pred=y_pred_train,
                scaled=False,
                title="temporal train data")

# Compute two baseline. Simply take the mood the day before and take the average mood.
compute_metrics(y_true=y_true_test,
                y_pred=y_pred_last_mood_test,
                scaled=False,
                title="baseline test data")

predictions_last_mood_train, y_train_flattened = [], []
for user_input_data, user_target_data in tqdm(zip(X_train, y_train), total=len(X_train), desc=f"Evaluating"):
    user_id, inputs = user_input_data
    _, targets = user_target_data

    for input, target in zip(inputs, targets):
        denormalised_target = input[-1][MOOD_INDEX] * normalisation_constants[user_id]['mood_mean']['std'] + normalisation_constants[user_id]['mood_mean']['mean']
        predictions_last_mood_train.append(denormalised_target)
        y_train_flattened.append(target)

compute_metrics(y_true=y_train_flattened,
                y_pred=predictions_last_mood_train,
                scaled=False,
                title="baseline train data")
