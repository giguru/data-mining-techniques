import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self,
                 dropout_rate,
                 num_classes=1,  # Single output, since we are doing regression
                 input_size=20,
                 hidden_size=14,
                 num_layers=1,
                 ):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        batch_size = 1
        self.h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).double())
        torch.nn.init.normal_(self.h_0)

        self.c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).double())
        torch.nn.init.normal_(self.c_0)

        self.fc = nn.Linear(hidden_size, num_classes)
        torch.nn.init.normal_(self.fc.weight)

    def forward(self, x):
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (self.h_0, self.c_0))

        h_out = h_out.view(-1, self.hidden_size)
        out = self.dropout(h_out)
        out = self.fc(out)

        return out