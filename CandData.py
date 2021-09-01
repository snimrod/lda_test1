from xls_loader import is_empty_text

EXCEL = "סיים בהצטיינות"
MOFET = "סיים למופת"


NF1 = "ותרנות"
NF2 = "אי התאמה"
NF3 = "הרחק אי-התקדמו"
NF4 = "לא עמד/ה בתנאי"
NE = "הרחק בחינת כ."


class CandData:

    def __init__(self, db, line):
        self.officer = db.officer[line]
        sium = db.OFEN_SIUM_KKZ[line]
        if self.officer == 1:
            self.honor = (sium == EXCEL or sium == MOFET)
            self.notFinished = False
            self.notEntered = False
        else:
            self.honor = False
            self.notFinished = (sium == NF1 or sium == NF2 or sium == NF3 or sium == NF4)
            self.notEntered = (sium == NE)

        if is_empty_text(db.TZIUN_KKZ[line]):
            self.grade = ""
        else:
            self.grade = db.TZIUN_KKZ[line]

        if is_empty_text(db.DAPAR[line]):
            self.dapar = ""
        else:
            self.dapar = db.DAPAR[line]

        if is_empty_text(db.TZADAK[line]):
            self.tzadak = ""
        else:
            self.tzadak = db.TZADAK[line]

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