from xls_loader import get_cands_data
from xls_loader import get_translated_text
from infra import get_cand
import pandas as pd

CHARS_FILE = 'char_thesis_db.xlsx'

# Types
PERSONAL = 1
HISTORIC = 2
FICTIONAL = 3
AMORPHOUS = 4

fiction_females = ['מולאן', 'מולן', 'הרמיוני', 'פוקהונ', 'רפונזל']
historic_females = ['רונה רמון', 'אשתו של אילן רמון', 'מרים פרץ', 'אנגלינה', 'מארי ק', 'אופרה ו', 'רוזה פ', 'מרגרט',
                    'אליס מילר', 'שרה אה', 'חנה סנש', 'פרידה ק', 'אלטשולר', 'הלן קלר', 'שרה גיבו', 'אילנה דיין']
real_females = ['אמא שלי', 'סבתא שלי', 'אחות שלי', 'חברה שלי', 'דודה שלי', 'אחותי']
historic_males = ['שמעון פרס', 'הרצל', 'רבין', 'אנשטיין', 'אינשטיין', 'איינשטין', 'איינשטיין', 'רועי קליין', 'רועי קלין', 'משה רב'
            , 'ינוש', 'יאנוש', 'לותר', 'מייקל גורדן', 'מייקל פלפס', 'מנדלה', 'גנדי ', 'קהלני', 'טרומפלדור',
            'קובי בר', 'אלי כהן', 'רונאלדו', 'דוד המלך', 'שלמה המלך', 'אריק שרון', 'נתניהו', 'צרציל', 'מרדכי',
            'בניה ריין', ' בנאי', 'בגין', 'אילן רמון', 'אברהם אבינו', 'אברהם לינק', 'הרב אברהם', 'אברם', 'שטרן',
            'עובדיה יוסף', 'אלכסנדר', 'נפוליאון', 'סטיב גובס', 'סטיבן ג', 'נל מסי', 'רבי עקיבא', 'בן נון', 'הרב קוק',
            'ביל ג', 'יגאל', 'אליעזר', 'מאסק', 'מייקל גק', 'גלילאו', 'ארמסטרונג', 'רמבם', 'הוקינג', 'נפתלי בנט',
            'צפלין', 'היטלר', 'מוחמד', 'מייק הררי', 'אובמה', 'מורנו', 'הוקינג', 'ריבלין', 'נועם גרשוני', 'וינברג',
            'זבוטינסקי', 'טסלה', 'אהרון הכהן','ארז גר', 'מאיר הר', 'אריאל שרון']
fiction_males = ['באטמן', 'בטמן', 'סופרמן', 'סופר מן', 'בובספוג', 'בוב ספוג', 'קפטן אמריקה', 'איירו', 'ספייד', 'פו הדוב',
                 'מיקי מאוס', 'פורסט', 'שרלוק']
real_males = ['אבא שלי', 'סבא שלי', 'אח שלי', 'חבר שלי', 'דוד שלי']
temp = ['מאיר הר']

# join_d = historic_males + historic_females + fiction_females + fiction_males + real_males + real_females


#def load_characters():
#char_db = pd.read_excel(CHARS_FILE, 'Original', index_col=None, usecols=None, header=0, nrows=8000)

def move_to_auto_found():
    not_found_db = pd.read_excel('not_found_chars.xlsx', 'Original', index_col=None, usecols=None, header=0, nrows=8000)
    found_db = pd.read_excel('found_chars.xlsx', 'Original', index_col=None, usecols=None, header=0, nrows=8000)


def search_characters(char_list, cand):
    found = False
    for char in char_list:
        if char in cand.hebText:
            found = True
            break
    return found


def analyze_characters():
    print("Analyzing characters\n")
    full_text = get_translated_text("Translated_text.txt")
    db = get_cands_data('thesis_db_copy.xls', 12110)
    reviewed_cands = []
    errors = [0, 0, 0, 0, 0, 0]
    to_drop = []
    found_char = []
    no_char = []
    tmp_cnt = 0

    cnt = 0
    char_cnt = 0
    found_cnt = 0

    for index in range(12110):
        this_cand_id = db.ID_coded[index]
        if this_cand_id in reviewed_cands:
            to_drop.append(index)
        else:
            reviewed_cands.append(this_cand_id)
            cand = get_cand(db, full_text, index, [2015], errors)
            if cand is None:
                to_drop.append(index)
            else:
                found = False
                ffound = False
                mfound = False
                cnt = cnt + 1
                db.f_char_type[index] = 0
                db.m_char_type[index] = 0
                for char in temp:
                    if char in cand.hebText:
                        tmp_cnt = tmp_cnt + 1
                        print(cand.hebText)

                if search_characters(fiction_males, cand):
                    found = True
                    char_cnt = char_cnt + 1
                    db.m_char_type[index] = 3
                    mfound = True
                if search_characters(historic_males, cand):
                    found = True
                    char_cnt = char_cnt + 1
                    db.m_char_type[index] = 2
                    mfound = True
                if search_characters(real_males, cand):
                    found = True
                    char_cnt = char_cnt + 1
                    db.m_char_type[index] = 1
                    mfound = True
                if search_characters(fiction_females, cand):
                    found = True
                    char_cnt = char_cnt + 1
                    db.f_char_type[index] = 3
                    ffound = True
                if search_characters(historic_females, cand):
                    found = True
                    char_cnt = char_cnt + 1
                    db.f_char_type[index] = 2
                    ffound = True
                if search_characters(real_females, cand):
                    found = True
                    char_cnt = char_cnt + 1
                    db.f_char_type[index] = 1
                    ffound = True


                    #print(cand.hebText)
                if found:
                    if mfound and ffound:
                        db.f_char_type[index] = ""
                        db.m_char_type[index] = ""
                        no_char.append(index)
                    else:
                        found_cnt = found_cnt + 1
                        found_char.append(index)
                else:
                    db.f_char_type[index] = ""
                    db.m_char_type[index] = ""
                    no_char.append(index)

    print(index)
    print(cnt)
    print("Different candidates that found a match: {}".format(found_cnt))
    print("Different characters found: {}".format(char_cnt))
    print("temps {}".format(tmp_cnt))
    #db2 = db.drop(to_drop, axis=0)
    #db2 = db2.drop(columns=["Test_Date", "bts_a", "bts_c", "bts_e", "bts_n", "bts_o", "T_LEIDA", "T_GIUS", "officer",
    #                       "DAPAR", "TZADAK", "TAARICH_MAVDAK", "TZIYUN_MAVDAK", "OFEN_SIUM_KKZ", "TZIUN_KKZ",
    #                       "SOCIO_TIRONUT", "SOTZIO_PIKUD"])
    #db2.to_excel(CHARS_FILE, sheet_name='Original')
    #found_db = db2.drop(no_char, axis=0)
    #not_found_db = db2.drop(found_char, axis=0)
    #found_db.to_excel('found_chars.xlsx', sheet_name='Original')
    #not_found_db.to_excel('not_found_chars.xlsx', sheet_name='Original')


#analyze_characters()
move_to_auto_found()
