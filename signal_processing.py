from muselsl import stream, list_muses

# pylsl
import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop # Module to receive EEG data
import utils  # Our own utility functions

import subprocess

# Goals: 
    # Differentiate between stressed and unstressed based on thresholds
    # Differentiate between movement and not movement to denoise
    # Seems like red line detects movement most, Muse also has gyro to 
    # detect muse
    # StreamOutlet of high level metrics after processing of 
    # amount stressed, amount of movement, amount of acceleration
    # Figure out which brainwaves indicate what state ie delta, beta, alpha, theta..

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [1]

# Make sure muse can be found / is paired
def find_muse_stream(use_subprocess=False):
    # Async start stream
    if use_subprocess:
        done = False
        while not done:
            out = subprocess.check_output(["python", "connect.py"])
            print("out: ", str(out))
            if "Connected." in str(out):
                done = True
            elif "No Muses found" in str(out):
                raise RuntimeError("Couldn't connect to muse")

    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    return streams[0]

def start_stream(stream):
    
    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(stream, max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()


    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    # Create outlet for streaming brain waves
    outlet = create_outlet_stream(fs)

    eeg_buffer, band_buffer = configure_buffers(fs)

    while True:
        """ 3.1 ACQUIRE DATA """
        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))

        # Only keep the channel we're interested in
        ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

        filter_state = None  # for use with the notch filter
        # Update EEG buffer with the new data
        eeg_buffer, filter_state = utils.update_buffer(
            eeg_buffer, ch_data, notch=True,
            filter_state=filter_state)

        """ 3.2 COMPUTE BAND POWERS """
        # Get newest samples from the buffer
        data_epoch = utils.get_last_data(eeg_buffer,
                                         EPOCH_LENGTH * fs)

        # Compute band powers
        band_powers = utils.compute_band_powers(data_epoch, fs)
        band_buffer, _ = utils.update_buffer(band_buffer,
                                             np.asarray([band_powers]))
        # Compute the average band powers for all epochs in buffer
        # This helps to smooth out noise
        smooth_band_powers = np.mean(band_buffer, axis=0)
        a, b, t = compute_metrics(smooth_band_powers)
        outlet.push_sample([a, b, t])

        # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
        #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

def configure_buffers(fs):
        # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))
    return eeg_buffer, band_buffer

def compute_metrics(smooth_band_powers):

    # Alpha Protocol:
    # Simple redout of alpha power, 
    # divided by delta waves in order to rule out noise
    alpha_metric = smooth_band_powers[Band.Alpha] / \
        smooth_band_powers[Band.Delta]
    # print('Alpha Relaxation: ', alpha_metric)

     # Beta Protocol:
     # Beta waves have been used as a measure of mental activity and concentration
     # This beta over theta ratio is commonly used as neurofeedback for ADHD
    beta_metric = smooth_band_powers[Band.Beta] / \
         smooth_band_powers[Band.Theta]
    # print('Beta Concentration: ', beta_metric)

    # Alpha/Theta Protocol:
    # This is another popular neurofeedback metric for stress reduction
    # Higher theta over alpha is supposedly associated with reduced anxiety
    theta_metric = smooth_band_powers[Band.Theta] / \
        smooth_band_powers[Band.Alpha]
    # print('Theta Relaxation: ', theta_metric)
    #
    print("alpha: {:.4f}  beta: {:.4f}  delta: {:.3f}  theta: {:.4f}".format(
        smooth_band_powers[Band.Alpha],
        smooth_band_powers[Band.Beta],
        smooth_band_powers[Band.Delta],
        smooth_band_powers[Band.Theta]))

    print("alpha/delta: {:.4f}  beta/theta: {:.3f}  theta/alpha: {:.4f}".format(
        alpha_metric, beta_metric, theta_metric))
    print()
    return alpha_metric, beta_metric, theta_metric

def create_outlet_stream(fs):
    channels = ["Alpha Relaxation", "Beta Concentration", "Theta Relaxation"]
    info = StreamInfo(name='Stress Data', type='EEG',
                   channel_count=len(channels), nominal_srate=fs,
                   channel_format='float32', source_id='stressdata')
    outlet = StreamOutlet(info)
    return outlet

def main():
    stream = find_muse_stream()
    start_stream(stream)

if __name__ == "__main__":
    main()
