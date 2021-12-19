import torch
from torch.utils.data import Dataset
from BrainStatesSample import BrainStatesSample
from Choice import Choice


# Abstraction Function
# sample_len is the number of readings we take for predicting each choice, for example if we measure
# the alpha, beta, gamma relaxations measurements twenty times (maybe constituting a few seconds of
# measurements) before a choice and use that data to predict their choice, then sample_len = 20
# num_options are for example a, b, c, d for each choice, so num_options would be 4
# num_choices is the number of total choices made throughout the game
# num_reading_metrics is the number of measurements per reading, for example alpha, beta, and gamma
# measurements is num_reading_metrics = 3

# 20 seconds prior used to predict choice
SAMPLE_LEN = 20
NUM_CHOICES = 4
NUM_OPTIONS = 4
NUM_READING_METRICS = 3


class BrainStatesDataset(Dataset):

    def __init__(self, brain_states_csv, choices_csv):
        self.brain_states = self.add_sample_choice_pairs(brain_states_csv, choices_csv)
        # self.transform = transforms.Compose([transforms.ToTensor()])


    def add_sample_choice_pairs(self, brain_states_file, choices_file):
        samples = []
        with open(brain_states_file, "r") as f:
            sample = BrainStatesSample()
            for line in f.readlines():
                # while line is not blank, add line as a array
                if line.strip() == "":
                    # skip empty lines
                    continue
                else:
                    # Build up the sample with readings
                    line = line.strip()
                    line = line.split(",")
                    sample.add_reading(line)
                    # Add to brain_states and reset the vector
                    # associated with brain states before a choice
                    if len(sample) == SAMPLE_LEN:
                        sample.avg_readings()
                        samples.append(sample)
                        # Reset the sample we're creating
                        sample = BrainStatesSample(num_reading_metrics=NUM_READING_METRICS, sample_len=SAMPLE_LEN)

        with open(choices_file, "r") as f:
            choices = self.one_hot_encode_choices(f.readlines())

        assert(len(choices) == NUM_CHOICES)
        assert(len(samples) == len(choices))
        # used map to get rid of tuples from zip
        arr = list(map(list, zip(samples, choices)))
        return arr


    def one_hot_encode_choices(self, choices_data):
        choices = [Choice(letter) for letter in choices_data]
        return choices


    # Returns number of samples in the dataset
    def __len__(self):
        return len(self.brain_states)


    # Returns one fetched sample from the list of samples
    def __getitem__(self, idx):
        # Below is used for higher dimensions of indexing to get a single sample
        #if torch.is_tensor(idx):
        #idx = idx.tolist()


        # brain_states[ind] = [Sample, Choice]
        assert(len(self.brain_states[idx]) == 2)
        # no type casting in python
        #sample_choice_pair = ([BrainStatesSample, Choice]) self.brain_states[idx]
        item = [torch.Tensor(self.brain_states[idx][0].averaged_readings),
                torch.Tensor(self.brain_states[idx][1].choice)]
        return item


    def __str__(self):
        return str(self.brain_states)
