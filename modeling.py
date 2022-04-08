from utils import get_temporal_records, read_data, SECONDS_IN_DAY, VARIABLES_WITH_UNFIXED_RANGE
import numpy as np
from sklearn.tree import DecisionTreeRegressor

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
                                **({ key: np.mean for key in VARIABLES_WITH_UNFIXED_RANGE })
                                })

print(records)