import pandas as pd
from pandas import DataFrame
from utils import get_temporal_records, read_data, VARIABLES_WITH_UNFIXED_RANGE, fill_defaults, keep_per_day, mean
import seaborn as sn
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
figure(figsize=(20, 20), dpi=80)


OUTPUT_PATH = './output/'


def check_existing_folder(this_path):
    MYDIR = (this_path)
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)


DEFAULT_CALL = 0
DEFAULT_SMS = 0
DEFAULT_AROUSAL = 0
DEFAULT_VALENCE = 1
DEFAULT_MOOD = 7.0

data = read_data()

nday_window = 3
records = get_temporal_records(data,
                               nday_window,
                               {'circumplex.arousal': mean,
                                'circumplex.valence': mean,
                                'activity': sum,
                                'sms': sum,
                                'call': sum,
                               },
                               {
                                'circumplex.arousal': lambda daily_mean, _: np.mean(fill_defaults(daily_mean, nday_window, DEFAULT_AROUSAL))/np.std(fill_defaults(daily_mean, nday_window, DEFAULT_AROUSAL)),
                                'circumplex.valence': lambda daily_means, _: np.mean(fill_defaults(daily_means, nday_window, DEFAULT_VALENCE)),
                                'activity': mean,
                                'call': lambda n_calls, _: np.mean(n_calls) / np.max(n_calls) if len(n_calls) > 0 else DEFAULT_CALL,
                                'sms': lambda n_sms, _: np.mean(n_sms) / np.max(n_sms) if len(n_sms) > 0 else DEFAULT_SMS,
                                'week_day': lambda _, row: row['week_day'] % 7 - 3,
                                'mood': keep_per_day(default=DEFAULT_MOOD),
                                **({ key: [mean, sum, len] for key in VARIABLES_WITH_UNFIXED_RANGE })
                                })

feature_matrix = np.array([list(r[0].values()) + r[1:] for r in records])
print(feature_matrix[0])
labels = list(records[0][0].keys()) + ['mood', 'id']
df = pd.DataFrame(feature_matrix, columns=labels)
check_existing_folder(OUTPUT_PATH)
df.to_csv(os.path.join(OUTPUT_PATH, f'feature_tab_{nday_window}.csv'), index=False)

X = feature_matrix[:, :-2]  # The data for
y = feature_matrix[:, -2]

# Print correlation matrix
corrMatrix = DataFrame(X, columns=records[0][0].keys()).apply(pd.to_numeric).corr(method='pearson')
sn.heatmap(corrMatrix, annot=True)
plt.show()

# A simple decision tree
mdl = DecisionTreeClassifier(n_neighbors=10)
mdl = mdl.fit(X=X, y=y)
print(mdl.score(X, y))
