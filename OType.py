from Topic import Topic
from DataSet import DataSet
from CandData import CandData
#from TopicModeling_IDF import N
#from TopicModeling_IDF import O_TYPES_STR

# Officer types (KAKATZ)
NONE = 0
LAHAV = 1
MAOZ = 2
NACHSHON = 3

O_TYPES_STR = ["None", "Lahav", "Maoz", "Nachshon"]


def get_officer_type(i):
    # Note 2 lines diff between excel row and i

    if 0 <= i <= 2405:
        return LAHAV

    if 2406 <= i <= 3940:
        return MAOZ

    if 3941 <= i <= 8406:
        return NACHSHON

    return NONE


class OType:

    def __init__(self, topics_n, id):
        self.id = id
        self.dSet = DataSet(O_TYPES_STR[id])
        self.N = topics_n
        self.topics = []

        for i in range(self.N):
            self.topics.append(Topic(i))

    def add_item(self, probabilities, cand):
        #officer = db.officer[line]
        #if officer == 1:
        #    y = 2

        self.dSet.add_item(cand)
        for i in range(self.N):
            self.topics[i].add_item(probabilities[i], cand)

    def print_topics(self):
        output = "Officer Type {}\n".format(O_TYPES_STR[self.id])

        output = "{}\n{}\n".format(output, self.dSet.print_data())
        for i in range(self.N):
            output = "{}{}\n".format(output, self.topics[i].print_datasets())

        return output
