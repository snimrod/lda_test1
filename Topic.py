class Topic:

    def __init__(self):
        self.first = 0  # For how many cands this topic is first
        self.officers = 0

    def inc_first(self):
        self.first = self.first + 1

    def inc_officers(self):
        self.officers = self.officers + 1

    def officers_percent(self):
        if self.first > 0:
            return 100 * self.officers/self.first
        else:
            return 0
