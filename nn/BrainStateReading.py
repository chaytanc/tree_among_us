
class BrainStateReading():
    # def __init__(self, alpha, beta, gamma):
    #     self.alpha = alpha
    #     self.beta = beta
    #     self.gamma = gamma
    #     self.reading = [self.alpha, self.beta, self.gamma]


    def __init__(self, *args):
        self.reading = []
        for arg in args:
            self.reading = [float(val) for val in arg]