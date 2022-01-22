from BrainStateReading import BrainStateReading


class BrainStatesSample():


    # Abstraction Function
    # num_reading_metrics is the number of measurements per reading, for example alpha, beta, and gamma
        # measurements is num_reading_metrics = 3
    # sample_len is the number of readings we take for predicting each choice, for example if we measure
    # the alpha, beta, gamma relaxations measurements twenty times (maybe constituting a few seconds of
    # measurements) before a choice and use that data to predict their choice, then sample_len = 20

    def __init__(self, num_reading_metrics=3, sample_len=20):
        # Should be full of SAMPLE_LEN number of BrainStateReadings
        self.raw_readings: [BrainStateReading] = []
        # The 20 (or whatever SAMPLE_LEN is) readings per sample are condensed to one averaged array
        # so that it looks like [avg_alpha, avg_beta, avg_gamma]
        self.averaged_readings = []
        # defaults to 20 of each kind of measurement for each choice (ie 20 alpha, 20 beta ... per choice)
        self.SAMPLE_LEN = sample_len
        self.NUM_READING_METRICS = num_reading_metrics


    def add_reading(self, *args):
        assert(len(*args) == self.NUM_READING_METRICS)
        reading = BrainStateReading(*args)  # pass a line of the data
        self.raw_readings.append(reading)


    # Precondition: Call only after SAMPLE_LEN readings have been added to self.readings
    def avg_readings(self):
        assert(len(self.raw_readings) == self.SAMPLE_LEN)
        sums = [0 for x in range(self.NUM_READING_METRICS)]
        for reading in self.raw_readings:
            reading = reading.reading
            for i in range(self.NUM_READING_METRICS):
                sums[i] += reading[i]
        avged = [x / self.SAMPLE_LEN for x in sums]
        self.averaged_readings = avged
        self.checkrep()


    def add_choice_number(self):
        self.averaged_readings.append(self.raw_readings.choice_number)


    def __len__(self):
        return len(self.raw_readings)


    def checkrep(self):
        assert(len(self.raw_readings) == self.SAMPLE_LEN)
        assert(len(self.averaged_readings) == self.NUM_READING_METRICS)


    def __str__(self):
        return str(self.averaged_readings)
