from pylsl import StreamInlet, resolve_stream

# first resolve an EEG stream on the lab network
def get_inlet():
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    print("Streams", streams)
    for stream in streams:
        print("id", str(stream.source_id()))
        if stream.source_id() == "stressdata":
            inlet = StreamInlet(stream)
    return inlet

# create a new inlet to read from the stream

inlet = get_inlet()
while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    print(timestamp, sample)
