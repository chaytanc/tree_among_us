
class Choice():

    def __init__(self, letter, num_options=4):
        self.NUM_OPTIONS = num_options
        self.letter = letter
        self.choice = self.encode()
        self.checkrep()


    def encode(self):
        self.letter = self.letter.strip()
        if self.letter == "a":
            choice = [1, 0, 0, 0]
        elif self.letter == "b":
            choice = [0, 1, 0, 0]
        elif self.letter == "b":
            choice = [0, 0, 1, 0]
        elif self.letter == "d":
            choice = [0, 0, 0, 1]
        else:
            raise ValueError("letter is not a supported Choice")
        return choice


    def checkrep(self):
        assert(len(self.choice) == self.NUM_OPTIONS)


    def __str__(self):
        return str(self.choice)
