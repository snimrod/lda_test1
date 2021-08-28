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
            self.grade = 0
        else:
            self.grade = db.TZIUN_KKZ[line]

        if is_empty_text(db.DAPAR[line]):
            self.dapar = 0
        else:
            self.dapar = db.DAPAR[line]

        if is_empty_text(db.TZADAK[line]):
            self.tzadak = 0
        else:
            self.tzadak = db.TZADAK[line]
