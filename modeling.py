import pandas as pd
from pandas import DataFrame
import os
from utils import get_temporal_records, read_data, SECONDS_IN_DAY, VARIABLES_WITH_UNFIXED_RANGE, fill_defaults
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



data = read_data()

nday_window = 1
records = get_temporal_records(data,
                               SECONDS_IN_DAY * nday_window,
                               {'circumplex.arousal': np.mean,
                                'call': sum,
                                'week_day': lambda x: x[0]},
                               {# Just naively take the means
                                'circumplex.arousal': lambda daily_mean: np.mean(fill_defaults(daily_mean, nday_window, 0))/np.std(fill_defaults(daily_mean, nday_window, 0)),
                                'circumplex.valence': np.mean,
                                'activity': np.mean,
                                'call': lambda n_calls: np.mean(n_calls)/np.max(n_calls),
                                'sms': sum,
                                'week_day': lambda x: (x[-1] + 1)%7 - 3
                                **({ key: [np.mean, sum, len] for key in VARIABLES_WITH_UNFIXED_RANGE })
                                })

feature_matrix = np.array([list(r[0].values()) + r[1:] for r in records])
print(feature_matrix)
labels = ['circumplex.arousal', 'circumplex.valence', 'activity', 'call', 'sms'] + \
         VARIABLES_WITH_UNFIXED_RANGE + ['mood', 'id']
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
