import pandas as pd
from pandas import DataFrame
from utils import get_temporal_records, read_data, VARIABLES_WITH_UNFIXED_RANGE, fill_defaults, keep_per_day, mean,\
    check_existing_folder
import seaborn as sn
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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

N_DAY_WINDOW = 3
save_file_path = os.path.join(OUTPUT_PATH, f'feature_tab_{N_DAY_WINDOW}.csv')

if os.path.exists(save_file_path):
    df = pd.read_csv(save_file_path)
    feature_labels = df.columns.tolist()[:-2]
    feature_matrix = np.array(df.values.tolist())
else:
    data = read_data()
    records = get_temporal_records(data,
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


X = feature_matrix[:, :-2]  # The data for
y = feature_matrix[:, -2]
print("Example row:", X[0])
print("Example target:", y[0])

# TODO Normalization


# TODO Feature selection: e.g. PCA


# TODO split into train and validation set


# Print correlation matrix
corrMatrix = DataFrame(X, columns=feature_labels).apply(pd.to_numeric).corr(method='pearson')
sn.heatmap(corrMatrix, annot=True)
plt.show()

# A simple decision tree
print("Training model...")
mdl = DecisionTreeClassifier()
mdl = mdl.fit(X=X, y=y)
print("Score:", mdl.score(X, y))
