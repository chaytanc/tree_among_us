import unittest
import sys
sys.path.append('../')
from BrainStatesDataset import BrainStatesDataset

TRAINSET_BRAIN_STATES = "../data/fake_readings.csv"
TRAINSET_CHOICES = "../data/fake_choices.csv"

class TestData(unittest.TestCase):

    def setUp(self) -> None:
        self.data = BrainStatesDataset(TRAINSET_BRAIN_STATES, TRAINSET_CHOICES)

    # have 20 brain states per each of four choices, 3 readings per each of the brain states
    # (aka, fake_readings has around 80 lines and 2 measurements per line)
    def test_raw_data_size(self):
        # supposed data = [[[[5, 5, 5], [8, 3, 5], ... 20th brain state], [1, 0, 0, 0]], ... [brain states, 4th choice]]
        pred_len = 4
        self.assertEqual(pred_len, len(self.data.brain_states))


    def test_averaged_data_size(self):
        pass

if __name__ == '__main__':
    unittest.main()
