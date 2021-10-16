from xls_loader import is_valid_text
from CandData import CandData
from OType import NONE

ACCEPTED_RATIO = 1.3

# Validation errors
YEAR_MISMATCH = 0
INVALID_TEXT = 1
INVALID_TRANS_RATIO = 2
INVALID_OFFICER = 3
INVALID_SADIR = 4
INVALID_FAIL = 5


MEDIC = "הרחק-רפואיות"
BIRO = "סגירה מנהלתית"


def valid_translation(db, engLines, i):
    hebText = db.text[i]
    engText = engLines[i]
    ratio = len(engText)/len(hebText)
    return ratio > ACCEPTED_RATIO
    # return True


def get_cand(db, engLines, line, years, errors):

#    if db.ID_coded[i] == 13821367:
#        debugon = 1

    if not db.Test_Date[line].year in years:
        errors[YEAR_MISMATCH] = errors[YEAR_MISMATCH] + 1
        return None

    if not is_valid_text(db.text[line]):
        errors[INVALID_TEXT] = errors[INVALID_TEXT] + 1
        return None

    sium = db.OFEN_SIUM_KKZ[line]
    if sium == MEDIC or sium == BIRO:
        errors[INVALID_FAIL] = errors[INVALID_FAIL] + 1
        return None

    valid = True
    cand = CandData(db, line)

    if not valid_translation(db, engLines, line):
        errors[INVALID_TRANS_RATIO] = errors[INVALID_TRANS_RATIO] + 1
        valid = False

    if cand.officer == 1 and (cand.otype == NONE or cand.grade == 0 or cand.rejected == 1):
        errors[INVALID_OFFICER] = errors[INVALID_OFFICER] + 1
        valid = False

    if cand.officer == 0 and not cand.grade == 0:
        errors[INVALID_SADIR] = errors[INVALID_SADIR] + 1
        valid = False

    if valid:
        return cand
    else:
        return None

# Convert array of probabilities to array of locations (location in the list when ordered)
# gets a list of values and return a list where the value represent how many items in the list this item is bigger than
# the smallest value will be changed to 0 and biggest value to (size - 1)
def convert_locations(p_list):
    size = len(p_list)
    x = []
    ti = 0
    for n in p_list:
        x.append([n, ti])
        ti = ti + 1
    x.sort()
    loc = [0] * size

    for tn in range(1, size):
        if x[tn][0] == x[tn - 1][0]:
            loc[x[tn][1]] = loc[x[tn - 1][1]]
        else:
            loc[x[tn][1]] = tn

    return loc


# get list of tuples <id, val> and and return a list of list_len values.
# If index in list is id in tuples list add value in this index to the list, if not fill it with zero
def fill_zeros(tuples_list, list_len):
    val_list = [0] * list_len
    for t in tuples_list:
        val_list[t[0]] = t[1]
    return val_list
