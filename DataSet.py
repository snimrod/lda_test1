from CandData import CandData


class DataSet:

    def __init__(self, name):
        self.count = 0
        self.officers = 0
        self.kkz_g_total = 0
        self.honor = 0
        self.name = name

    def add_item(self, cand):
        self.count = self.count + 1
        if cand.officer == 1:
            self.officers = self.officers + 1
            self.kkz_g_total = self.kkz_g_total + cand.grade
            if cand.honor:
                self.honor = self.honor + 1

    def officers_pcnt(self):
        if self.count > 0:
            return 100 * self.officers/self.count
        else:
            return 0

    def kkz_g_avg(self):
        if self.officers > 0:
            return self.kkz_g_total/self.officers
        else:
            return 0

    def honor_pcnt(self):
        if self.officers > 0:
            return 100 * self.honor/self.officers
        else:
            return 0

    def print_data(self):
        return "{} - Count: {}, Officers {:.2f}% ({}), Exel {:.2f}% ({}), kkz avg {:.2f}".format(self.name, self.count, self.officers_pcnt(),
                                                                                                 self.officers, self.honor_pcnt(), self.honor,  self.kkz_g_avg())
