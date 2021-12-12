import torch
from torch.utils.data import Dataset


# 20 seconds prior used to predict choice
STATE_LEN = 20

class BrainStatesDataset(Dataset):

    def __init__(self, brain_states_csv, choices_csv):
        self.brain_states = self.get_pandas_data(
            brain_states_csv, choices_csv)
        self.averaged_brain_states = []
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

        with open(choices_file, "r") as f:
            choices = self.one_hot_encode_choices(f.readlines())

        assert(len(brain_states) == len(choices))
        # used map to get rid of tuples from zip
        arr = list(map(list, zip(brain_states, choices)))
        return arr


    # Encodes a, b, c, d choice into vector with 1 in place of choice, ie a --> [1, 0, 0, 0]
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
        for i, state in enumerate(self.brain_states):
            avg = self.average_brain_states(self.brain_states[i][0])
            # Flatten useless dimension of array of brain states since we turned 20 into 1
            avg = avg[0]
            choice = state[1]
            self.averaged_brain_states.append([torch.Tensor(avg), torch.Tensor(choice)])
        item = self.averaged_brain_states[idx]
        return item

    def __str__(self):
        return str(self.brain_states)
