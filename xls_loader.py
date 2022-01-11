import pandas as pd
import codecs
from googletrans import Translator


def get_cands_data(fname, lines):
    return pd.read_excel(fname, 'Original', index_col=None, usecols=None, header=0, nrows=lines)


def get_translated_text(fname):
    f = open(fname, errors='ignore')
    lines = f.readlines()
    f.close()
    return lines


def is_empty_text(text):
    return pd.isnull(text)

def is_valid_text(text):

    if pd.isnull(text):
        return False

    if isinstance(text, (int, float)):
        return False

    wlist = text.split()
    if len(wlist) < 2:
        return False

    return True


def load_characters(chars_map):
    char_db = pd.read_excel('not_found_chars.xlsx', 'Original', index_col=None, usecols=None, header=0, nrows=8000)
    no_char = 0
    for i in range(len(char_db)):
        if is_empty_text(char_db.m_char_type[i]):
            m = 0
        else:
            m = char_db.m_char_type[i]
        if is_empty_text(char_db.f_char_type[i]):
            f = 0
        else:
            f = char_db.f_char_type[i]
        chars_map[char_db.ID_coded[i]] = [int(m), int(f)]
        if chars_map[char_db.ID_coded[i]] == [0, 0]:
            no_char = no_char + 1

        if not m == f and m > 0 and f > 0:
            print("{}: {} {}".format(char_db.ID_coded[i], m, f))

    print("no char selected for {} candidates".format(no_char))
    print("manual chars loaded {}".format(len(char_db)))

    auto_char_db = pd.read_excel('found_chars.xlsx', 'Original', index_col=None, usecols=None, header=0, nrows=8000)
    for i in range(len(auto_char_db)):
        if is_empty_text(auto_char_db.m_char_type[i]):
            m = 0
        else:
            m = auto_char_db.m_char_type[i]
        if is_empty_text(auto_char_db.f_char_type[i]):
            f = 0
        else:
            f = auto_char_db.f_char_type[i]
        chars_map[auto_char_db.ID_coded[i]] = [m, f]

        if not m == f and m > 0 and f > 0:
            print("{}: {} {}".format(auto_char_db.ID_coded[i], m, f))

    print("auto chars loaded {}".format(len(auto_char_db)))
    print("total chars loaded {}".format(len(chars_map)))

# df = pd.read_excel('Book1.xlsx', sheetname=None, header=None)
# df = pd.read_excel('Book1.xlsx')

# Read Excel and select a single cell (and make it a header for a column)
# data = pd.read_excel('Book11.xls', 'Sheet1', index_col=None, usecols="C", header=0, nrows=0)
#data = pd.read_excel('thesis_db.xls', 'Original', index_col=None, usecols=None, header=0, nrows=12110)

#f = open("Translated_text.txt", errors='ignore')
#lines = f.readlines()

#for line in lines:
#    print(line)

#f.close()

#print(len(data.values[0,8]))
#print(len(lines[0]))
#print(len(lines[0])/len(data.values[0,8]))

#count1 = 0
#count2 = 0
#count3 = 0
#count4 = 0

#for i in range(0, 12109):
#    hebText = data.values[i, 8]
#    engText = lines[i]
#    if not(pd.isnull(hebText)) and not(isinstance(hebText, (int, float))):
#        ratio = len(engText)/len(hebText)
        # print("Checking line {} ({})".format(i, engText[0:20]))
#        if ratio < 1:
#            print("{})ratio is: {} ({})".format(i, ratio, engText[0:20]))
#            count1 = count1 + 1
#            print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))
#        if (ratio >= 1) and (ratio < 1.1):
#            print("{})ratio is: {} ({})".format(i, ratio, engText[0:20]))
#            count2 = count2 + 1
#            print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))
#        if (ratio >= 1.1) and (ratio < 1.2):
#            print("{})ratio is: {} ({})".format(i, ratio, engText[0:20]))
#            count3 = count3 + 1
#            print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))
#        if (ratio >= 1.2) and (ratio < 1.3):
#            print("{})ratio is: {} ({})".format(i, ratio, engText[0:20]))
#            count4 = count4 + 1
#            print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))
#f = codecs.open("txt_output.txt", "w", "utf-8")

# f = open("txt_output.txt", "w")
#print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))

#translator = Translator()



#print(data.values[1,8])
#result = translator.translate(data.values[1,8], src='he', dest='en')
#print(result.text)



#for c in data.דמות:
#    # print(i)
#    result = translator.translate(c)
#    f.write("{}\n".format(result.text))
#    i = i + 1

#f.close()
#print("Done")