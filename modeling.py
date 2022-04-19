import pandas as pd
from utils import process_data, read_data, VARIABLES_WITH_UNFIXED_RANGE, fill_defaults, keep_per_day, mean, \
    check_existing_folder
import numpy as np
import os
from matplotlib.pyplot import figure
from sklearn.tree import DecisionTreeRegressor, plot_tree

figure(figsize=(20, 20), dpi=80)


def scale_features_is_values(df, test_size=0.2, save_csv=True):
    """
    This function takes a DataFrame with features and target ('mood') and normalize the features with
    the insample mean/max std. Features can be classified in 3 types:
    -std: to be scaled by x->(x-x_mean)/x_std
    -max: to be scaled by x->x/x_max
    -no_change: x-> x
    Mean, std, max are computed IS and per person.
    Currently which features belong to which class it is hardcoded

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
    cols_2_drop = [col for col in df.columns if col[-3:] == 'sum'] #  these are redundant
    std_attribute = mood_cols_scaled + ['mood_scaled', 'mood_before_mean', 'screen_mean', 'screen_len', 'amount_screen_mean', 'amount_screen_len',
                     'screenrest_mean', 'screenrest_len']
    max_attribute = ['circumplex.arousal_custom', 'circumplex.valence_custom', 'activity_mean', 'call_custom', 'sms_custom',
                     'appCat.builtin_mean', 'appCat.builtin_len', 'appCat.communication_mean', 'appCat.communication_len',
                     'appCat.entertainment_mean',
                     'appCat.entertainment_len', 'appCat.finance_mean', 'appCat.finance_len', 'appCat.game_mean',
                     'appCat.game_len', 'appCat.office_mean', 'appCat.office_len', 'appCat.other_mean', 'appCat.other_len',
                     'appCat.social_mean', 'appCat.social_len', 'appCat.travel_mean', 'appCat.travel_len',
                     'appCat.unknown_mean', 'appCat.unknown_len', 'appCat.utilities_mean', 'appCat.utilities_len',
                     'appCat.weather_mean', 'appCat.weather_len']

    #take DF, assign IS/OS lable, compute mean, return X_train, ...
    this_df = df.copy()
    for i in range(N_DAY_WINDOW):
        this_df[f'mood_{1+i}_day_before_scaled'] = this_df[f'mood_{1+i}_day_before']
    this_df = this_df.loc[:, mood_cols_scaled + this_df.columns.to_list()[:-N_DAY_WINDOW]]
    this_df.iloc[:, :-1] = this_df.iloc[:, :-1].astype(float)
    this_df.loc[:, 'mood_scaled'] = this_df.loc[:, 'mood']
    this_df.drop(columns=cols_2_drop, inplace=True)
    this_df['mood_before_mean'] = this_df.loc[:, mood_cols].mean(axis=1)
    to_sort = [col for col in this_df.columns.to_list() if col not in ['mood_before_mean', 'id']]
    this_df = this_df.loc[:, to_sort + ['mood_before_mean', 'id']] #reorder
    #select is
    this_df['rand'] = np.random.random_sample(len(this_df))
    no_change = [cols for cols in this_df.columns if cols not in max_attribute+std_attribute+['id']]
    this_df_is = this_df.loc[this_df.rand > test_size]

    #parameters for std
    mean_dict = dict(zip(std_attribute, ['mean']*len(std_attribute)))
    scaling_dict = dict(zip(std_attribute + max_attribute, ['std']*len(std_attribute) + ['max']*len(max_attribute)))

    is_mean = this_df_is.loc[:, std_attribute+['id']].groupby(['id']).agg(mean_dict)
    is_mean.loc[:, max_attribute+no_change] = 0
    is_std = this_df_is.iloc[:, :-1].groupby(['id']).agg(scaling_dict)
    is_std = is_std.replace(0, 1)
    is_std.loc[:, no_change] = 1

    # save  the scaling factors for the y_rain_scale
    scale_back_df = pd.DataFrame(index=is_std.index)
    scale_back_df['std'] = is_std['mood_scaled']
    scale_back_df['mean'] = is_mean['mood_scaled']

    # Transform on the whole dataset
    this_df = this_df.set_index('id').subtract(is_mean)
    this_df = this_df.divide(is_std).reset_index()
    # sort columns
    sorted_cols = [cols for cols in this_df.columns if cols not in ['id', 'mood', 'mood_scaled', 'rand'] + mood_cols]
    sorted_cols = ['mood', 'mood_scaled', 'id', 'rand'] + mood_cols + sorted(sorted_cols)
    this_df = this_df.loc[:, sorted_cols]

    #split is/os
    df_is = this_df.loc[this_df.rand > test_size].drop(columns='rand')
    df_os = this_df.loc[this_df.rand <= test_size].drop(columns='rand')
    #save
    X_train, y_train, y_train_scaled = df_is.iloc[:, 2:].values, df_is.iloc[:, 0].values, df_is.iloc[:, 1].values
    X_test, y_test, y_test_scaled = df_os.iloc[:, 2:].values, df_os.iloc[:, 0].values, df_os.iloc[:, 1].values
    feature_labels = df_is.iloc[:, 2:].columns.to_list()
    if save_csv:
        this_df.to_csv(os.path.join(OUTPUT_PATH, 'scaled_dataset.csv'), index=False)
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
N_NON_FEATURES = len([MOOD_INDEX, ID_INDEX])

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
    df = pd.DataFrame(feature_matrix, columns=feature_labels + ['mood', 'id'])
    check_existing_folder(OUTPUT_PATH)
    df.to_csv(save_file_path, index=False)

# TODO Vincenzo: Decide normalisation constants using training set only and split train/test
X_train, y_train, \
y_train_scaled, \
X_test, y_test, \
y_test_scaled, feature_labels, scale_back_df = scale_features_is_values(df, test_size=0.2, save_csv=True)
print("Example row:", X_train[0])
print("Example target:", y_train[0])

# TODO Vincenzo: Feature selection using training set only: e.g. PCA
attributes = pd.read_csv(os.path.join(OUTPUT_PATH, 'selected_attributes.csv'))
n_cols = len(attributes.columns)
this_attr = []
for i in range(n_cols):
    this_attr = this_attr + list(attributes[str(i)].dropna().values)
this_attr = list(set(this_attr))

feat_index = [idx for idx in range(len(feature_labels)) if feature_labels[idx] in this_attr]
X_train = X_train[:, feat_index]
X_test = X_test[:, feat_index]

print("Training model...")
mdl = DecisionTreeRegressor()
mdl = mdl.fit(X=X_train, y=y_train)
plot_tree(mdl)
print("Score:", mdl.predict(X_test), y_test)

# TODO Bram: train a temporal model, e.g. LSTM, RNN, etc.


# TODO Giguru: compute two base lines. Simply take the mood the day before and take the average mood.


# TODO Evaluation: confusion matrix, mean squared error, qualitative prediction power per user.

