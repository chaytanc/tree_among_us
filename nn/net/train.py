import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from BrainStatesDataset import BrainStatesDataset
from Net import Net
import wandb
from Dict2Class import Dict2Class

# Paths
TRAINSET_BRAIN_STATES = "../data/fake_readings.csv"
TESTSET_BRAIN_STATES = "../data/fake_readings.csv"
VALIDSET_BRAIN_STATES = "../data/fake_readings.csv"
TRAINSET_CHOICES = "../data/fake_choices.csv"
TESTSET_CHOICES = "../data/fake_choices.csv"
VALIDSET_CHOICES = "../data/fake_choices.csv"
MODEL_PATH = "choice_pred_net.pth"

# Settings
PRINT_RATE = 3
NUM_WORKERS = 0
WB = True

# Settings for BrainStatesDataset
dataset_settings = dict(
    sample_len=20,
    #XXX just changed
    num_choices=2,
    num_options=6,
    num_reading_metrics=3,
)

# Hyperparameters
config = dict(
    epochs=100,
    batch_size=1,
    lr=0.005,
    momentum=0.9,
)


def model_pipeline(hyperparameters):

    if WB:
        with wandb.init(project="neurogame_choice_predictor", config=hyperparameters):
            # access all HPs through wandb.config, so logging matches execution!
            config = wandb.config

            # make the model, data, and optimization problem
            model, train_loader, test_loader, criterion, optimizer = make(config)

            # and use them to train the model
            train(model, train_loader, criterion, optimizer, config)

            # and test its final performance
            # test(model, test_loader)
    else:
        config = Dict2Class(hyperparameters)

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config)

        # and test its final performance
        # test(model, test_loader)

    return model


# Config should be dot-referenceable object
def make(config):

    # Brain states 20 seconds before each choice and the corresponding choice made
    def load_data(brain_states_csv, choices_csv, ds):
        dataset = BrainStatesDataset(brain_states_csv, choices_csv, dataset_settings=ds)
        # num_workers for parallelization ;)
        dataloader = DataLoader(dataset, batch_size=config.batch_size,
                                shuffle=True, num_workers=NUM_WORKERS)
        return dataloader

    testset = load_data(TESTSET_BRAIN_STATES, TESTSET_CHOICES, dataset_settings)
    trainset = load_data(TRAINSET_BRAIN_STATES, TRAINSET_CHOICES, dataset_settings)
    # What happens when in real life you no longer have the labels and just have
    # the the brain_states and no choices

    net = Net()

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
    torch.save(net.state_dict(), MODEL_PATH)


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


# Build, train and analyze the model with the pipeline
model = model_pipeline(config)


