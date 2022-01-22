from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers and sizes
        #XXX need to figure out input dimensions --> I think 3 per sample??
        # (when do I flatten the 3 differnt measures of relaxation per reading?)
        # how many features do we expect to have in the middle
        self.fc1 = nn.Linear(3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)
        # four output probabilities, corresponding to choices a, b, c, d

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # CE loss does its own softmax
        # m = nn.Softmax(dim=1)
        # x = m(x)
        return x
