# from CandData import CandData


class DataSet:

    def __init__(self, name):
        self.count = 0
        self.officers = 0
        self.kkz_g_total = 0
        self.honor = 0
        self.nf = 0
        self.ne = 0
        self.name = name
        self.d_count = 0
        self.d_total = 0
        self.z_count = 0
        self.z_total = 0

    def add_item(self, cand):
        self.count = self.count + 1
        if cand.officer == 1:
            self.officers = self.officers + 1
            self.kkz_g_total = self.kkz_g_total + cand.grade
            if cand.excel:
                self.honor = self.honor + 1
        else:
            if cand.notFinished:
                self.nf = self.nf + 1
            if cand.notEntered:
                self.ne = self.ne + 1

        if cand.dapar > 0:
            self.d_count = self.d_count + 1
            self.d_total = self.d_total + cand.dapar

        if len(cand.tzadak) > 0:
            self.z_count = self.z_count + 1
            self.z_total = self.z_total + int(cand.tzadak)


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

    def nf_pcnt(self):
        if self.count > 0:
            return 100 * self.nf/self.count
        else:
            return 0

    def ne_pcnt(self):
        if self.count > 0:
            return 100 * self.ne/self.count
        else:
            return 0

    def dapar_avg(self):
        if self.d_count > 0:
            return self.d_total/self.d_count
        else:
            return 0

    def zadak_avg(self):
        if self.z_count > 0:
            return self.z_total/self.z_count
        else:
            return 0

    def print_data(self):
        return "{} - {}, Off {:.2f}% ({}), Exel {:.2f}% ({}), NF {:.2f}% ({}), NE {:.2f}% ({}), kkz avg {:.2f}, dap {:.2f}, tza {:.2f} ({})".format(self.name, self.count, self.officers_pcnt(),
                                                                                                                       self.officers, self.honor_pcnt(), self.honor,  self.nf_pcnt(),
                                                                                                                       self.nf, self.ne_pcnt(), self.ne, self.kkz_g_avg(), self.dapar_avg(),
                                                                                                                       self.zadak_avg(), self.z_count)
