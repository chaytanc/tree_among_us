
class Choice:
    # Choices are represented as 0, 1, ... n-1 number of choices
    # Last column of choices data is which choice they decided on

    def __init__(self, choice_row, num_options=4):
        self.NUM_OPTIONS = num_options
        # self.choice = int(choice_row[0])
        self.choice = self.encode(choice_row[0])
        self.choice_number = choice_row[-1]
        self.checkrep()


    # def encode(self):
    #     self.letter = self.letter.strip()
    #     if self.letter == "a":
    #         choice = [1, 0, 0, 0]
    #     elif self.letter == "b":
    #         choice = [0, 1, 0, 0]
    #     elif self.letter == "b":
    #         choice = [0, 0, 1, 0]
    #     elif self.letter == "d":
    #         choice = [0, 0, 0, 1]
    #     else:
    #         raise ValueError("letter is not a supported Choice")
    #     return choice

    def encode(self, which_choice):
        encoded_choice = [0 for x in range(self.NUM_OPTIONS)]
        encoded_choice[int(which_choice)] = 1
        return encoded_choice

    def checkrep(self):
        assert(len(self.choice) == self.NUM_OPTIONS)


    def __str__(self):
        return str(self.choice)
