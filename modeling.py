import pandas as pd
import os
from utils import get_temporal_records, read_data, SECONDS_IN_DAY, VARIABLES_WITH_UNFIXED_RANGE
import numpy as np
import os

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
                               None,
                               {# Just naively take the means
                                'circumplex.arousal': np.mean,
                                'circumplex.valence': np.mean,
                                'activity': np.mean,
                                'call': sum,
                                'sms': sum,
                                **({ key: np.mean for key in VARIABLES_WITH_UNFIXED_RANGE })
                                })

feature_matrix = np.array([list(r[0].values()) + r[1:] for r in records])
print(feature_matrix)
labels = ['circumplex.arousal', 'circumplex.valence', 'activity', 'call', 'sms'] + \
         VARIABLES_WITH_UNFIXED_RANGE + ['mood', 'id']
df = pd.DataFrame(feature_matrix, columns=labels)
check_existing_folder(OUTPUT_PATH)
df.to_csv(os.path.join(OUTPUT_PATH, f'feature_tab_{nday_window}.csv'), index=False)