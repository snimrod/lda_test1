from xls_loader import is_empty_text
from OType import get_officer_type
import gensim.corpora as corpora
#from character_analyzer import get_fchar_type
#from character_analyzer import get_mchar_type

EXCEL = "סיים בהצטיינות"
MOFET = "סיים למופת"


NF1 = "ותרנות"
NF2 = "אי התאמה"
NF3 = "הרחק אי-התקדמו"
NF4 = "לא עמד/ה בתנאי"
NE = "הרחק בחינת כ."


def get_cand_char_gender(chars_type):
    # gender: 0 = female, 1 = male, 2 = amorphous
    if chars_type[1] == chars_type[0] or (chars_type[1] > 0 and chars_type[0] > 0):
        charg = 2
        chart = chars_type[1]
    else:
        if chars_type[1] > 0:
            charg = 0
            chart = chars_type[1]
        else:
            charg = 1
            chart = chars_type[0]
    return charg, chart


class CandData:

    def __init__(self, db, characters_map, line):
        self.id = db.ID_coded[line]
        self.year = db.Test_Date[line].year
        self.officer = db.officer[line]
        self.otype = get_officer_type(line)
        self.hebText = db.text[line]
        self.probs = []

        t = self.hebText.split()
        self.words_n = len(t)
        id2word = corpora.Dictionary([t])
        self.unique_ratio = len(id2word) / self.words_n

        sium = db.OFEN_SIUM_KKZ[line]
        if self.officer == 1:
            if sium == EXCEL or sium == MOFET:
                self.excel = 1
            else:
                self.excel = 0
            self.notFinished = False
            self.notEntered = False
        else:
            self.excel = 0
            self.notFinished = (sium == NF1 or sium == NF2 or sium == NF3 or sium == NF4)
            self.notEntered = (sium == NE)

        if is_empty_text(db.TZIUN_KKZ[line]):
            self.grade_str = ""
            self.grade = 0
        else:
            self.grade_str = db.TZIUN_KKZ[line]
            self.grade = int(self.grade_str)

        if is_empty_text(db.DAPAR[line]):
            self.dapar = ""
        else:
            self.dapar = db.DAPAR[line]

        if is_empty_text(db.TZADAK[line]):
            self.tzadak = ""
            self.sex = 0
        else:
            self.tzadak = db.TZADAK[line]
            self.sex = 1

        if self.id in characters_map:
            chars_type = characters_map[self.id]
            #self.m_char_type = chars_type[0]
            #self.f_char_type = chars_type[1]
            self.charg, self.chart = get_cand_char_gender(characters_map[self.id])

        self.rejected = 0
        self.mavdak1 = ""
        self.mavdak2 = ""
        if not is_empty_text(db.TZIYUN_MAVDAK[line]):
            mavdak = db.TZIYUN_MAVDAK[line]
            if mavdak == 10 or mavdak == 11 or mavdak == 90 or mavdak == 91:
                self.rejected = 1
            else:
                if mavdak < 10:
                    self.mavdak1 = mavdak
                else:
                    self.mavdak2 = mavdak

        if is_empty_text(db.SOCIO_TIRONUT[line]):
            self.socio_t = ""
        else:
            self.socio_t = db.SOCIO_TIRONUT[line]

        if is_empty_text(db.SOTZIO_PIKUD[line]):
            self.socio_p = ""
        else:
            self.socio_p = db.SOTZIO_PIKUD[line]

        #self.fchar_type = get_fchar_type(line)
        #self.mchar_type = get_mchar_type(line)
