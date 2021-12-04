import pandas as pd
import torch
import torch.optim as optim

# 20 seconds prior used to predict choice
STATE_LEN = 20
BATCH_SIZE = 1
TRAINSET = "fake_train_readings.csv"
TESTSET = "fake_readings.csv"
VALIDSET = "fake_readings.csv"

class BrainStatesDataset(Dataset):

    def __init__(self, brain_states_csv, choices_csv):
        self.brain_states = self.get_pandas_data(brain_states_csv)

    def get_pandas_data(self, brain_states_file):
        brain_states = []
        choices = []
        with open(brain_states, "r") as f:
            twenty_second_state = []
            for line in f.readlines():
               # while line is not blank, add line as a array  
                if line == "":
                    assert(len(twenty_second_state) == STATE_LEN)
                    brain_states.append(twenty_second_state)
                else:
                   state = line.split(",")
                   twenty_second_state.append(state)

        with open(choices, "r") as f:
            for line in f.readlines():
                choices.append(line)

        assert(len(brain_states) == len(choices))
        #XXX may need function that converts "a" --> [1, 0, 0, 0] etc
        # Trying to zip together brain_state w choice label / annotation
        return pd.df(zip(brain_states, choices))

    def one_hot_encode_choices(choices):
        choices = []
        for choice in choices:
            if choice == "a":
                choices.append[1, 0, 0, 0]
            else if choice == "b":
                choices.append[0, 1, 0, 0]
            else if choice == "b":
                choices.append[0, 0, 1, 0]
            else ifi choice == "d":
                choices.append[0, 0, 0, 1]
            else:
                raise ValueError("choices data is bad")
        return choices
                

    # Returns number of samples in the dataset
    def __len__(self):
        return len(self.brain_states)

    # Returns one fetched sample from the list of samples
    def __getitem__(self, idx):
        # Below is used for higher dimensions of indexing to get a single sample
        #if torch.is_tensor(idx):
            #idx = idx.tolist()

        return self.brain_states[idx] # --> [a, b, t], [a|b|c|d]

def load_data(brain_states_csv):
    dataset = BrainStatesDataset(brain_states_csv)
    # num_workers for parallelization ;)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=0)

testset = load_data(TESTSET)
trainset = load_data(TRAINSET)
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

def train(training_data, criterion, optimizer):
    loss = 0.0
    for i, data in enumerate(training_data, 0):
        input_layer, labels = data
        optimizer.zero_grad()
        output = net(input_layer)
        loss = criterion(outputs, labels)
        # After getting how far off we were from the labels, calc
        # gradient descent derivs
        loss.backward()
        # update weights and biases
        optimizer.step()
        # summing up loss for all samples in the dataset
        running_loss += loss.item()


