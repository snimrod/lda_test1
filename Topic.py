from DataSet import DataSet

LEVELS = 5


def prob_to_level(probability):
    if probability < 0.2:
        return 0

    if probability < 0.4:
        return 1

    if probability < 0.6:
        return 2

    if probability < 0.8:
        return 3

    return 4


class Topic:

    def __init__(self, id):
        self.id = id
        self.dSets = []

        for i in range(LEVELS):
            self.dSets.append(DataSet(i))

    def add_item(self, probability, cand):
        self.dSets[prob_to_level(probability)].add_item(cand)

    def print_datasets(self):
        output = "Topic {}\n".format(self.id)
        for i in range(LEVELS):
            output = "{}{}\n".format(output, self.dSets[i].print_data())

        return output
