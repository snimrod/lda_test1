class DataSet:

    def __init__(self, name):
        self.count = 0
        self.officers = 0
        self.name = name

    def add_item(self, officer):
        self.count = self.count + 1
        if officer == 1:
            self.officers = self.officers + 1

    def officers_percent(self):
        if self.count > 0:
            return 100 * self.officers/self.count
        else:
            return 0

    def print_data(self):
        return "{} - Count: {}, Officers {:.2f}%".format(self.name, self.count, self.officers_percent())
