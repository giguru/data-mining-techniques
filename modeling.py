import pandas as pd
from pandas import DataFrame
from utils import process_data, read_data, VARIABLES_WITH_UNFIXED_RANGE, fill_defaults, keep_per_day, mean, \
    check_existing_folder, temporal_input_generator
import seaborn as sn
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
os.chdir(r"C:\Users\Bram\data-mining-techniques")
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
    records = process_data(
                                   data,
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


X = feature_matrix[:, :-N_NON_FEATURES]  # The data for
y = feature_matrix[:, MOOD_INDEX]
print("Example row:", X[0])
print("Example target:", y[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TODO Vincenzo: Decide normalisation constants using training set only


# TODO Vincenzo: Feature selection using training set only: e.g. PCA


# Print correlation matrix
# corrMatrix = DataFrame(X, columns=feature_labels).apply(pd.to_numeric).corr(method='pearson')
# sn.heatmap(corrMatrix, annot=True)
# plt.show()


# TODO training a non-temporal model
#print("Training model...")
#mdl = DecisionTreeRegressor()
#mdl = mdl.fit(X=X_train, y=y_train)
#plot_tree(mdl)
#print("Score:", mdl.predict(X_test), y_test)
seq_length = 10

# Create temporal dataset


# TODO Bram: train a temporal model, e.g. LSTM, RNN, etc.
traiX=[]
traiY=[]
for trX, trY in temporal_input_generator(feature_matrix,mood_index=MOOD_INDEX,id_index=ID_INDEX,min_sequence_len=10, max_sequence_len=10):
    trX=[[float(i) for i in x]for x in trX]
    a=np.array(trX)

    trY=float(trY)
    traiX.append(trX)
    traiY.append(trY)

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
num_epochs = 900
learning_rate = 0.01

input_size = 47
hidden_size = 2
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):

    trainY = torch.Tensor(traiY)
    
    trainX = torch.Tensor(traiX)
    outputs = lstm(trainX)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



print(lstm(trainX))
print(traiY)
# TODO Giguru: compute two base lines. Simply take the mood the day before and take the average mood.


# TODO Evaluation: confusion matrix, mean squared error, qualitative prediction power per user.

