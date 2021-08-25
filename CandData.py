from xls_loader import is_empty_text

EXCEL = "סיים בהצטיינות"
MOFET = "סיים למופת"

class CandData:

    def __init__(self, db, line):
        self.officer = db.officer[line]
        if self.officer == 1:
            sium = db.OFEN_SIUM_KKZ[line]
            self.honor = (sium == EXCEL or sium == MOFET)
        else:
            self.honor = False

        if is_empty_text(db.TZIUN_KKZ[line]):
            self.grade = 0
        else:
            self.grade = db.TZIUN_KKZ[line]

