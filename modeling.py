from pandas import DataFrame
import pandas as pd
from utils import get_temporal_records, read_data, SECONDS_IN_DAY, VARIABLES_WITH_UNFIXED_RANGE
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure

figure(figsize=(20, 20), dpi=80)

data = read_data()

records = get_temporal_records(data,
                               SECONDS_IN_DAY * 1,
                               None,
                               {# Just naively take the means
                                'circumplex.arousal': np.mean,
                                'circumplex.valence': np.mean,
                                'activity': np.mean,
                                'call': sum,
                                'sms': sum,
                                **({key: np.mean for key in VARIABLES_WITH_UNFIXED_RANGE})
                                })

feature_matrix = np.array([list(r[0].values()) + r[1:] for r in records])
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