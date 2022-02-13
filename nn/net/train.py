import os
import re
import sys

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from BrainStatesDataset import BrainStatesDataset
from Net import Net
import wandb
from Dict2Class import Dict2Class



# Settings
PRINT_RATE = 3
NUM_WORKERS = 0
WB = False

# Settings for BrainStatesDataset
dataset_settings = dict(
    sample_len=20,
    num_choices=6,
    num_options=2,
    num_reading_metrics=3,
    choices_row_length=4,
    train_dir="./data/real/train/",
    test_dir="./data/real/test/",
    train_path_r="/readings.csv",
    test_path_r="/readings.csv",
    train_path_c="/choices.csv",
    test_path_c="/choices.csv",
    model_path="/choice_pred_net.pth",
)

# Hyperparameters
config = dict(
    epochs=100,
    batch_size=1,
    lr=0.005,
    momentum=0.9,
)


def model_pipeline(hyperparameters, dataset_settings):

    setup = Setup(train_dir=dataset_settings['train_dir'],
                  test_dir=dataset_settings['test_dir'],
                  train_path_r=dataset_settings['train_path_r'],
                  test_path_r=dataset_settings['test_path_r'],
                  train_path_c=dataset_settings['train_path_c'],
                  test_path_c=dataset_settings['test_path_c'])
    if WB:
        with wandb.init(project="neurogame_choice_predictor", config=hyperparameters):
            # access all HPs through wandb.config, so logging matches execution!
            config = wandb.config

            # make the model, data, and optimization problem
            model, train_loader, test_loader, criterion, optimizer = setup.make(config)

            # and use them to train the model
            train(model, train_loader, criterion, optimizer, config)

            # and test its final performance
            # test(model, test_loader)
    else:
        config = Dict2Class(hyperparameters)

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = setup.make(config)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config)

        # and test its final performance
        # test(model, test_loader)

    return model

class Setup():

    def __init__(self, train_dir="./data/fake/", test_dir="./data/fake/",
                train_path_r="/fake_readings.csv",
                test_path_r="/fake_readings.csv" ,
                valid_path_r="/fake_readings.csv",
                train_path_c="/fake_choices",
                test_path_c="/fake_choices",
                valid_path_c="/fake_choices",
                model_path="/choice_pred_net.pth"):
        # Paths
        # print(os.getcwd())
        # working_dir = os.getcwd()
        self.TRAIN_DIR = train_dir
        self.TEST_DIR = test_dir
        self.TRAINSET_BRAIN_STATES = train_dir + train_path_r
        self.TESTSET_BRAIN_STATES = test_dir + test_path_r
        # VALIDSET_BRAIN_STATES = "./data/fake/fake_readings.csv"
        self.VALIDSET_BRAIN_STATES = test_dir + valid_path_r
        self.TRAINSET_CHOICES = train_dir + train_path_c
        self.TESTSET_CHOICES = test_dir + test_path_c
        self.VALIDSET_CHOICES = test_dir + valid_path_c
        self.MODEL_PATH = self.TRAIN_DIR + model_path

    # Config should be dot-referenceable object
    def make(self, config):

        # Brain states 20 seconds before each choice and the corresponding choice made
        def load_data(brain_states_csv, choices_csv, ds):
            dataset = BrainStatesDataset(brain_states_csv, choices_csv, dataset_settings=ds)
            # num_workers for parallelization ;)
            dataloader = DataLoader(dataset, batch_size=config.batch_size,
                                    shuffle=True, num_workers=NUM_WORKERS)
            return dataloader

        train_readings, train_choices = self.concat_data(self.TRAIN_DIR)
        test_readings, test_choices = self.concat_data(self.TEST_DIR)

        # testset = load_data(TESTSET_BRAIN_STATES, TESTSET_CHOICES, dataset_settings)
        # trainset = load_data(TRAINSET_BRAIN_STATES, TRAINSET_CHOICES, dataset_settings)

        testset = load_data(test_readings, test_choices, dataset_settings)
        trainset = load_data(train_readings, train_choices, dataset_settings)

        # What happens when in real life you no longer have the labels and just have
        # the the brain_states and no choices

        net = Net(dataset_settings['choices_row_length'], dataset_settings['num_options'])

        # loss function to minimize classification models
        # logarithmic loss function w smaller penalty for small diffs,
        # large for large diffs...
        # if entropy of loss is high, that means random guessing, so higher penalty
        # loss = -y_true * log(y_pred)
        criterion = nn.CrossEntropyLoss()
        # stochastic gradient descent calculates derivs of all
        # and minimizes error based on
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum)

        return net, trainset, testset, criterion, optimizer

    # given a dir, get all *_readings and all *_choices, combine into a single csv for each
    def concat_data(self, data_dir):
        readings = data_dir + "/readings.csv"
        choices = data_dir + "/choices.csv"
        files = os.listdir(data_dir)
        for f in files:

            # get the beginning readings pattern, match with a choices file, write to both
            if f.endswith("readings.csv"):
                date_pattern = f.split("readings.csv")[0] + r"[a-zA-Z0-9]*choices.csv"
                matches = [file for file in files if re.match(date_pattern, file)]
                # If we only have one choices csv matching the timestamped reading csv
                # and the match is not the choices.csv file itself, if that exists
                # then open the readings file and append to the concat readings file
                # and after that open the choices file and append to it
                if len(matches) == 1 and \
                        (not os.path.exists(choices) or not os.path.samefile(data_dir + matches[0], choices)):
                    with open(readings, "a+") as readings_file:
                        with open(data_dir + f, "r") as f:
                            for line in f.readlines():
                                readings_file.write(line)
                            f.close()
                        readings_file.close()

                    # Open matching choices file, readlines, write all lines to choices
                    with open(choices, "a+") as choices_file:
                        with open(data_dir + matches[0], "r") as c:
                            for line in c.readlines():
                                choices_file.write(line)
                            c.close()
                        choices_file.close()

        return readings, choices

# epoch is number of times to train with same dataset, should be greater than or equal to 1
def train_batch(net, data, criterion, optimizer, metrics):
    input_layer, labels = data
    optimizer.zero_grad()
    outputs = net(input_layer)
    softmax_outputs = nn.Softmax(outputs) # for printing probabilities
    loss = criterion(outputs, labels)
    # Get the most likely prediction and keep track of correct predictions
    pred = outputs.max(1, keepdim=True)[1][0].item()
    metrics.correct += 1 if pred == torch.argmax(labels) else 0

    # After getting how far off we were from the labels, calc
    # gradient descent derivs
    loss.backward()
    # update weights and biases
    optimizer.step()
    # summing up loss for all samples in the dataset, will zero after printing
    metrics.running_loss += loss.item()
    # Print
    if metrics.batch_i % PRINT_RATE == (PRINT_RATE - 1):    # print every 2000 mini-batches
        print('[prediction: {}, labels: {}, individual loss: {}]\n'.format(outputs, labels, loss))


def train(net, training_data, criterion, optimizer, config):
    if WB:
        wandb.watch(net, criterion, log="all", log_freq=10)

    metrics = dict(running_loss=0.0, correct=0, training_steps=0, batch_i=0, epochs=0)  # number of examples seen
    metrics = Dict2Class(metrics)
    for e in range(config.epochs):
        # metrics = dict(running_loss=0.0, correct=0, epoch=e)
        metrics.running_loss = 0.0
        metrics.correct = 0
        metrics.epoch = e
        for batch_i, data in enumerate(training_data, 0):
            metrics.batch_i = batch_i
            train_batch(net, data, criterion, optimizer, metrics)
            metrics.training_steps += 1

            if metrics.batch_i % PRINT_RATE == (PRINT_RATE - 1):
                print('[%d, %5d] loss: %.3f' %
                      (metrics.epoch + 1, metrics.batch_i + 1, metrics.running_loss / PRINT_RATE)
                      )
                if WB:
                    train_log(metrics.running_loss, metrics.training_steps, metrics.epoch)
                metrics.running_loss = 0.0  # reset running loss to recalculate for the next print_rate sized batch
        print('Correct in epoch: {}'.format(metrics.correct))

    # save updates to the model weights and biases after training
    torch.save(net.state_dict(), dataset_settings['train_dir'] + dataset_settings['model_path'])


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


# Build, train and analyze the model with the pipeline
model = model_pipeline(config, dataset_settings)


