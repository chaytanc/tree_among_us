import os
import unittest
import sys

sys.path.append('../')
sys.path.append('./net/')
from net.BrainStatesDataset import BrainStatesDataset
import net.train as train

# ASSUMES we're running from nn rootdir
# TRAINSET_BRAIN_STATES = "./data/fake/fake_readings.csv"
# TRAINSET_CHOICES = "./data/fake/fake_choices.csv"
TRAIN_DIR = "./data/fake/"
TRAINSET_BRAIN_STATES = TRAIN_DIR + "/readings.csv"
TRAINSET_CHOICES = TRAIN_DIR + "/choices.csv"

class TestPreprocess(unittest.TestCase):


    def setUp(self) -> None:
        try:
            os.remove(TRAINSET_BRAIN_STATES)
            os.remove(TRAINSET_CHOICES)
        except FileNotFoundError as e:
            print("Didn't remove choices or readings csv")
        self.setup = train.Setup(train_dir=TRAIN_DIR,
                            train_path_r=TRAINSET_BRAIN_STATES,
                            train_path_c=TRAINSET_CHOICES)

    def tearDown(self) -> None:
        try:
            os.remove(TRAINSET_BRAIN_STATES)
            os.remove(TRAINSET_CHOICES)
        except FileNotFoundError as e:
            print("Didn't remove choices or readings csv")

    def test_concat_data(self):
        readings, choices = self.setup.concat_data(TRAIN_DIR)
        self.data = BrainStatesDataset(readings, choices)
        pred_len = 7
        self.assertEqual(pred_len, len(self.data.brain_states))


class TestData(unittest.TestCase):

    def setUp(self) -> None:
        # Remove concatenated files and make fresh copies
        try:
            os.remove(TRAINSET_BRAIN_STATES)
            os.remove(TRAINSET_CHOICES)
        except FileNotFoundError as e:
            print("Didn't remove choices or readings csv")
        self.setup = train.Setup(train_dir=TRAIN_DIR,
                                 train_path_r=TRAINSET_BRAIN_STATES,
                                 train_path_c=TRAINSET_CHOICES)
        readings, choices = self.setup.concat_data(TRAIN_DIR)
        self.data = BrainStatesDataset(readings, choices)

    def tearDown(self) -> None:
        try:
            os.remove(TRAINSET_BRAIN_STATES)
            os.remove(TRAINSET_CHOICES)
        except FileNotFoundError as e:
            print("Didn't remove choices or readings csv")

    #XXX now that we concatenate, this breaks bc it adds the one other file choice
    # have 20 brain states per each of four choices, 3 readings per each of the brain states
    # (aka, fake_readings has around 80 lines and 2 measurements per line)
    def test_raw_data_size(self):
        # supposed data = [[[[5, 5, 5], [8, 3, 5], ... 20th brain state], [1, 0, 0, 0]], ... [brain states, 4th choice]]
        pred_len = 6
        self.assertEqual(pred_len, len(self.data.brain_states))

    # Tests averaging the, say, 20 brain states captured per choice into a single 1d array output
    def test_averaged_data(self):
        # first_sample_supposed_averages = torch.Tensor([2.0, 2.25, 2.25])
        first_sample_supposed_averages = [2.0, 2.25, 2.25]
        # second_sample_supposed_averages = torch.Tensor([4.5, 3.0, 2.5])
        second_sample_supposed_averages = [4.5, 3.0, 2.5]
        # Note that 3 is same as 1st sample, 2nd is same as 4th sample
        # self.assertTrue(torch.all(torch.eq(first_sample_supposed_averages, self.data.brain_states[0][0].averaged_readings)))
        self.assertTrue(first_sample_supposed_averages, self.data.brain_states[0][0].averaged_readings)
        self.assertTrue(second_sample_supposed_averages, self.data.brain_states[1][0].averaged_readings)
        self.assertTrue(first_sample_supposed_averages, self.data.brain_states[2][0].averaged_readings)
        self.assertTrue(second_sample_supposed_averages, self.data.brain_states[3][0].averaged_readings)


if __name__ == '__main__':
    unittest.main()
