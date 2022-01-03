from xls_loader import get_cands_data
from xls_loader import get_translated_text
from infra import get_cand

fiction_females = ['מולאן', 'מולן', 'הרמיוני', 'פוקהונ', 'רפונזל']
historic_females = ['רונה רמון', 'אשתו של אילן רמון', 'מרים פרץ', 'אנגלינה', 'מארי ק', 'אופרה ו', 'רוזה פ', 'מרגרט',
                    'אליס מילר', 'שרה אה', 'חנה סנש', 'פרידה ק', 'אלטשולר', 'הלן קלר', 'שרה גיבו']
real_females = ['אמא שלי', 'סבתא שלי', 'אחות שלי', 'חברה שלי', 'דודה שלי', 'אחותי']
historic_males = ['שמעון פרס', 'הרצל', 'רבין', 'אנשטיין', 'אינשטיין', 'איינשטין', 'איינשטיין', 'רועי קליין', 'רועי קלין', 'משה רב'
            , 'ינוש', 'יאנוש', 'לותר', 'מייקל גורדן', 'מייקל פלפס', 'מנדלה', 'גנדי ', 'קהלני', 'טרומפלדור',
            'קובי בר', 'אלי כהן', 'רונאלדו', 'דוד המלך', 'שלמה המלך', 'אריק שרון', 'נתניהו', 'צרציל', 'מרדכי',
            'בניה ריין', ' בנאי', 'בגין', 'אילן רמון', 'אברהם אבינו', 'אברהם לינק', 'הרב אברהם', 'אברם', 'שטרן',
            'עובדיה יוסף', 'אלכסנדר', 'נפוליאון', 'סטיב גובס', 'סטיבן ג', 'נל מסי', 'רבי עקיבא', 'בן נון', 'הרב קוק',
            'ביל ג', 'יגאל', 'אליעזר', 'מאסק', 'מייקל גק', 'גלילאו', 'ארמסטרונג', 'רמבם', 'הוקינג', 'נפתלי בנט', 'שרלוק',
            'צפלין', 'היטלר', 'מוחמד', 'מייק הררי', 'אובמה', 'מורנו', 'הוקינג', 'ריבלין', 'נועם גרשוני', 'וינברג',
            'זבוטינסקי', 'טסלה', 'אהרון הכהן']
fiction_males = ['באטמן', 'בטמן', 'סופרמן', 'בובספוג', 'בוב ספוג', 'קפטן אמריקה', 'איירו', 'ספייד', 'פו הדוב',
                 'מיקי מאוס', 'פורסט']
real_males = ['אבא שלי', 'סבא שלי', 'אח שלי', 'חבר שלי', 'דוד שלי']
guys = ['אחותי']

join_d = historic_males + historic_females + fiction_females + fiction_males + real_males + real_females


def analyze_characters():
    print("Analyzing characters\n")
    full_text = get_translated_text("Translated_text.txt")
    db = get_cands_data('thesis_db_copy.xls', 12110)


    reviewed_cands = []
    errors = [0, 0, 0, 0, 0, 0]
    #f = open("characters.csv", "w", encoding='utf-8')
    to_drop = []

    pc = 0
    cnt = 0

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
                cnt = cnt + 1
                for dude in join_d:
                    if dude in cand.hebText:
                        pc = pc + 1

    print(index)
    print(cnt)
    print(pc)
    #f.close()
    #db.to_excel('my_thesis_db.xlsx', sheet_name='Original')
    db2 = db.drop(to_drop, axis=0)
    db2.to_excel('my_thesis_db.xlsx', sheet_name='Original')


analyze_characters()
