import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 20 seconds prior used to predict choice
STATE_LEN = 20 #XXX not using this currently
BATCH_SIZE = 1
# TRAINSET = "fake_train_readings.csv"
TRAINSET_BRAIN_STATES = "fake_readings.csv"
TESTSET_BRAIN_STATES = "fake_readings.csv"
VALIDSET_BRAIN_STATES = "fake_readings.csv"
TRAINSET_CHOICES = "fake_choices.csv"
TESTSET_CHOICES = "fake_choices.csv"
VALIDSET_CHOICES = "fake_choices.csv"
MODEL_PATH = "./choice_pred_net.pth"
NUM_EPOCHS = 1

class BrainStatesDataset(Dataset):

    def __init__(self, brain_states_csv, choices_csv):
        self.brain_states = self.get_pandas_data(
                brain_states_csv, choices_csv)
        # self.transform = transforms.Compose([transforms.ToTensor()])

    def get_pandas_data(self, brain_states_file, choices_file):
        brain_states = []
        choices = []
        with open(brain_states_file, "r") as f:
            twenty_second_state = []
            for line in f.readlines():
                # while line is not blank, add line as a array
                if line.strip() == "":
                    # skip empty lines
                    continue
                else:
                    line = line.strip()
                    state = line.split(",")
                    twenty_second_state.append(state)
                    # Add to brain_states and reset the vector
                    # associated with brain states before a choice
                    if len(twenty_second_state) == STATE_LEN:
                        brain_states.append(twenty_second_state)
                        twenty_second_state = []

            # Last part of data edge case
            # if len(twenty_second_state) ==

        with open(choices_file, "r") as f:
            choices = self.one_hot_encode_choices(f.readlines())

        assert(len(brain_states) == len(choices))
        # Trying to zip together brain_state w choice label / annotation
        # arr = pd.DataFrame(zip(brain_states, choices))
        arr = list(zip(brain_states, choices))
        return arr
        # arr = np.array(zip(brain_states, choices))
        # return arr


    def one_hot_encode_choices(self, choices_data):
        choices = []
        for choice in choices_data:
            choice = choice.strip()
            if choice == "a":
                choices.append([1, 0, 0, 0])
            elif choice == "b":
                choices.append([0, 1, 0, 0])
            elif choice == "b":
                choices.append([0, 0, 1, 0])
            elif choice == "d":
                choices.append([0, 0, 0, 1])
            else:
                raise ValueError("choices data is bad")
        return choices


    # Averages twenty seconds before [[[a, b, g]_1...[a,b,g]_20 seconds]...n choices]
    # [[a_avg, b_avg, g_avg]...n choices] (flattens a dim of brain_states)
    #INPUT: ONE x-second time frame [[a, b, g]_1 ... [a, b, g]_x]
    def average_brain_states(self, brain_states):
        avged_brain_states = []
        # for time_frame in brain_states:
        a_sum = 0
        b_sum = 0
        g_sum = 0
        #XXX replace brain_states / be precise with names
        for sample in brain_states:
            sample = [float(x) for x in sample]
            a_sum += sample[0]
            b_sum += sample[1]
            g_sum += sample[2]
        avged = [x / len(brain_states) for x in [a_sum, b_sum, g_sum]]
        avged_brain_states.append(avged)
        return avged_brain_states


    # Returns number of samples in the dataset
    def __len__(self):
        # return self.brain_states[0].size
        return len(self.brain_states[0])


    # Returns one fetched sample from the list of samples
    def __getitem__(self, idx):
        # Below is used for higher dimensions of indexing to get a single sample
        #if torch.is_tensor(idx):
            #idx = idx.tolist()
        #NOTE: currently use average of brain_states to account for too many dims
        for i, choice in enumerate(self.brain_states):
            avg = self.average_brain_states(self.brain_states[i][0])
            # Flatten useless dimension of array
            avg = avg[0]
            self.brain_states[i][0] = avg
        return self.brain_states[idx] # --> [a, b, t], [a|b|c|d]


# Brain states 20 seconds before each choice and the corresponding choice made
def load_data(brain_states_csv, choices_csv):
    dataset = BrainStatesDataset(brain_states_csv, choices_csv)
    # num_workers for parallelization ;)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=0)
    return dataloader


testset = load_data(TESTSET_BRAIN_STATES, TESTSET_CHOICES)
trainset = load_data(TRAINSET_BRAIN_STATES, TRAINSET_CHOICES)
# What happenns when in real life you no longer have the labels and just have
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
    for i, data in enumerate(training_data, 0):
        input_layer, labels = data
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



