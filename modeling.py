import pandas as pd
from utils import process_data, read_data, VARIABLES_WITH_UNFIXED_RANGE, fill_defaults, keep_per_day, mean, \
    check_existing_folder, MAX_ATTRIBUTE, FIXED_STD_ATTRIBUTE, compute_metrics
import numpy as np
import os
from matplotlib.pyplot import figure
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

figure(figsize=(20, 20), dpi=80)


def scale_features_is_values(df, test_size=0.2, save_csv=True):
    """
    This function takes a DataFrame with features and target ('mood') and normalize the features with
    the insample mean/max std. Features can be classified in 3 types:
    -std: to be scaled by x->(x-x_mean)/x_std
    -max: to be scaled by x->x/x_max
    -no_change: x-> x
    Mean, std, max are computed IS and per person.
    Currently which features belong to which class it is hardcoded.

    :param df: DataFrame with
    :param test_size: % of entries to place  in test set.
    :return:
    X_train: np.array with scaled features (with is_values)
    y_train: vector with mood [0, 10]
    y_train_scaled: vector with mood in (0,1), scaled by x->(x-x_mean)/x_std
    X_test: np.array with out-of-sample features, scaled with is_values
    y_test: vector with out-of-sample mood
    y_test_scaled: vector with mood in (0,1), scaled by x->(x-x_mean)/x_std
    feature_labels: list with corresponding  feature name  for X_train/X_test
    scale_back_df: pd.DataFrame with index the id and corresponding mean and  std for y_target
    """
    # format DF
    mood_cols = [f'mood_{1+x}_day_before' for x in range(N_DAY_WINDOW)]
    mood_cols_scaled = [f'mood_{1+x}_day_before_scaled' for x in range(N_DAY_WINDOW)]
    cols_2_drop = [col for col in df.columns if col[-3:] == 'sum']  #  these are redundant
    std_attribute = mood_cols_scaled + FIXED_STD_ATTRIBUTE
    # take DF, assign IS/OS lable, compute mean, return X_train, ...
    df_copy = df.copy()

    # Add new columns
    for i in range(N_DAY_WINDOW):
        df_copy[f'mood_{1+i}_day_before_scaled'] = df_copy[f'mood_{1+i}_day_before']

    df_copy = df_copy.loc[:, mood_cols_scaled + df_copy.columns.to_list()[:-N_DAY_WINDOW]]

    exclude_index = -3  # Minus three, such that the date columns is not cast to a float
    df_copy.iloc[:, :exclude_index] = df_copy.iloc[:, :exclude_index].astype(float)
    df_copy.loc[:, 'mood_scaled'] = df_copy.loc[:, 'mood']
    df_copy.drop(columns=cols_2_drop, inplace=True)
    df_copy['mood_before_mean'] = df_copy.loc[:, mood_cols].mean(axis=1)
    to_sort = [col for col in df_copy.columns.to_list() if col not in ['mood_before_mean', 'id']]
    df_copy = df_copy.loc[:, to_sort + ['mood_before_mean', 'id']] #reorder
    # select is
    df_copy['rand'] = np.random.random_sample(len(df_copy))
    no_change = [cols for cols in df_copy.columns if cols not in MAX_ATTRIBUTE + std_attribute + ['id']]
    this_df_is = df_copy.loc[df_copy.rand > test_size]

    # parameters for std
    mean_dict = dict(zip(std_attribute, ['mean']*len(std_attribute)))
    scaling_dict = dict(zip(std_attribute + MAX_ATTRIBUTE, ['std'] * len(std_attribute) + ['max'] * len(MAX_ATTRIBUTE)))

    is_mean = this_df_is.loc[:, std_attribute+['id']].groupby(['id']).agg(mean_dict)
    is_mean.loc[:, MAX_ATTRIBUTE + no_change] = 0

    # -1 to prevent computing std of
    is_std = this_df_is.iloc[:, :-1].groupby(['id']).agg(scaling_dict)
    is_std = is_std.replace(0, 1)
    is_std.loc[:, no_change] = 1

    # Save the scaling factors for the y_rain_scale
    scale_back_df = pd.DataFrame(index=is_std.index)
    scale_back_df['std'] = is_std['mood_scaled']
    scale_back_df['mean'] = is_mean['mood_scaled']

    # Transform on the whole dataset
    date_column = df_copy['date']
    df_copy = df_copy.drop('date', axis=1).set_index('id').subtract(is_mean.drop('date', axis=1))
    df_copy['date'] = date_column
    df_copy = df_copy.divide(is_std).reset_index()
    # sort columns
    sorted_cols = [cols for cols in df_copy.columns if cols not in ['id', 'mood', 'mood_scaled', 'rand'] + mood_cols]
    sorted_cols = ['mood', 'mood_scaled', 'id', 'rand'] + mood_cols + sorted(sorted_cols)
    df_copy = df_copy.loc[:, sorted_cols]

    #split in-sample/out-sample
    df_is = df_copy.loc[df_copy.rand > test_size].drop(columns='rand')
    df_os = df_copy.loc[df_copy.rand <= test_size].drop(columns='rand')
    #save
    X_train, y_train, y_train_scaled = df_is.iloc[:, 2:].values, df_is.iloc[:, 0].values, df_is.iloc[:, 1].values
    X_test, y_test, y_test_scaled = df_os.iloc[:, 2:].values, df_os.iloc[:, 0].values, df_os.iloc[:, 1].values
    feature_labels = df_is.iloc[:, 2:].columns.to_list()
    if save_csv:
        df_copy.to_csv(os.path.join(OUTPUT_PATH, 'scaled_dataset.csv'), index=False)
        scale_back_df.to_csv(os.path.join(OUTPUT_PATH, 'scaled_factors.csv'))
    return X_train, y_train, y_train_scaled, X_test, y_test, y_test_scaled, feature_labels, scale_back_df


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
save_file_path = os.path.join(OUTPUT_PATH, f'feature_non_temporal_{N_DAY_WINDOW}.csv')

if os.path.exists(save_file_path):
    df = pd.read_csv(save_file_path)
    feature_labels = df.columns.tolist()[:-N_NON_FEATURES]
    feature_matrix = np.array(df.values.tolist())
else:
    data = read_data()
    records = process_data(data,
                           N_DAY_WINDOW,
                           {'circumplex.arousal': mean,
                            'circumplex.valence': mean,
                            'activity': sum,
                            'sms': sum,
                            'call': sum,
                            'mood': 'mean', # The mean must be always average per day
                           },
                           {
                            'circumplex.arousal': lambda daily_mean, _: np.mean(fill_defaults(daily_mean, N_DAY_WINDOW, DEFAULT_AROUSAL)),
                            'circumplex.valence': lambda daily_means, _: np.mean(fill_defaults(daily_means, N_DAY_WINDOW, DEFAULT_VALENCE)),
                            'activity': mean,
                            'call': lambda n_calls, _: np.mean(n_calls) / np.max(n_calls) if len(n_calls) > 0 else DEFAULT_CALL,
                            'sms': lambda n_sms, _: np.mean(n_sms) / np.max(n_sms) if len(n_sms) > 0 else DEFAULT_SMS,
                            'week_day': lambda _, row: row['week_day'] % 7 - 3,
                            'mood': keep_per_day(default=DEFAULT_MOOD),
                            **({ key: [mean, sum, len] for key in VARIABLES_WITH_UNFIXED_RANGE })
                            })

    feature_matrix = np.array([list(r[0].values()) + r[1:] for r in records])
    # Save data frame
    feature_labels = list(records[0][0].keys())
    df = pd.DataFrame(feature_matrix, columns=feature_labels + ['date', 'mood', 'id'])
    check_existing_folder(OUTPUT_PATH)
    df.to_csv(save_file_path, index=False)

# Decide normalisation constants using training set only
X_train, y_train, \
y_train_scaled, \
X_test, y_test, \
y_test_scaled, feature_labels, scale_back_df = scale_features_is_values(df, test_size=0.2, save_csv=True)
print("Example row:", X_train[0])
print("Example target:", y_train[0])

# Feature selection using training set only: e.g. PCA
attributes = pd.read_csv(os.path.join(OUTPUT_PATH, 'selected_attributes.csv'))
n_cols = len(attributes.columns)
this_attr = []
for i in range(n_cols):
    this_attr = this_attr + list(attributes[str(i)].dropna().values)
this_attr = list(set(this_attr))

feat_index = [idx for idx in range(len(feature_labels)) if feature_labels[idx] in this_attr]
feat_index_labels = [feature_labels[idx] for idx in range(len(feature_labels)) if feature_labels[idx] in this_attr]
X_train = X_train[:, feat_index]
X_test = X_test[:, feat_index]

print("\nTraining model...")
print("Example train input:\n  Labels:\n", feat_index_labels, "\n  Input:\n", X_train[0])
print("Example train target:", y_train_scaled[0], "\n")

mdl = LinearRegression()
mdl.fit(X=X_train, y=y_train_scaled)
# plot_tree(mdl)

# Evaluate on test set
y_pred = mdl.predict(X_test)
print("Example test predictions:", y_pred[:5])
print("Example test truths:", y_test_scaled[:5])
compute_metrics(y_true=y_test_scaled, y_pred=y_pred, scaled=True, title="test")

# Compute baseline
mood_idx = this_attr.index('mood_1_day_before')
predictions_last_mood_train = [r[mood_idx] for r in X_train]
compute_metrics(y_true=y_train_scaled,
                y_pred=predictions_last_mood_train,
                scaled=True,
                title="baseline train data")

predictions_last_mood_test = [r[mood_idx] for r in X_test]
compute_metrics(y_true=y_test_scaled,
                y_pred=predictions_last_mood_test,
                scaled=True,
                title="baseline test data")
