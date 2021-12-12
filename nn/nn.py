import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from BrainStatesDataset import BrainStatesDataset

BATCH_SIZE = 1
# TRAINSET = "fake_train_readings.csv"
TRAINSET_BRAIN_STATES = "./data/fake_readings.csv"
TESTSET_BRAIN_STATES = "data/fake_readings.csv"
VALIDSET_BRAIN_STATES = "data/fake_readings.csv"
TRAINSET_CHOICES = "data/fake_choices.csv"
TESTSET_CHOICES = "data/fake_choices.csv"
VALIDSET_CHOICES = "data/fake_choices.csv"
MODEL_PATH = "./choice_pred_net.pth"
NUM_EPOCHS = 1


# Brain states 20 seconds before each choice and the corresponding choice made
def load_data(brain_states_csv, choices_csv):
    dataset = BrainStatesDataset(brain_states_csv, choices_csv)
    # num_workers for parallelization ;)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=0)
    return dataloader


testset = load_data(TESTSET_BRAIN_STATES, TESTSET_CHOICES)
trainset = load_data(TRAINSET_BRAIN_STATES, TRAINSET_CHOICES)
# What happens when in real life you no longer have the labels and just have
# the the brain_states and no choices


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
        #XXX add softmax?? to convert logits to probabs?
        x = self.fc3(x)
        return x


net = Net()

# loss function to minimize classification models
# logarithmic loss function w smaller penalty for small diffs, 
# large for large diffs...
# if entropy of loss is high, that means random guessing, so higher penalty
criterion = nn.CrossEntropyLoss() 
# stochastic gradient descent calculates derivs of all 
# and minimizes error based on 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# epoch is number of times to train with same dataset, should be greater than or equal to 1
def train(training_data, criterion, optimizer, epoch=1):
    # print loss every 2000 samples through the net
    print_rate = 2000
    running_loss = 0.0
    print("Training data: ", str(training_data.dataset))
    print("Training data len: ", len(training_data.dataset))
    for i, data in enumerate(training_data, 0):
        input_layer, labels = data
        print(input_layer)
        optimizer.zero_grad()
        outputs = net(input_layer)
        loss = criterion(outputs, labels)
        # After getting how far off we were from the labels, calc
        # gradient descent derivs
        loss.backward()
        # update weights and biases
        optimizer.step()
        # summing up loss for all samples in the dataset, will zero after printing
        running_loss += loss.item()
        if i % print_rate == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / print_rate)
                  )
            running_loss = 0.0


for e in range(NUM_EPOCHS):
    train(trainset, criterion, optimizer, e)
# save updates to the model weights and biases after training
torch.save(net.state_dict(), MODEL_PATH)



