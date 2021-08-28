from xls_loader import get_cands_data
from xls_loader import get_translated_text
from xls_loader import is_valid_text
from xls_loader import is_empty_text
from OType import OType
from OType import get_officer_type
from OType import NONE
from DataSet import DataSet
from CandData import CandData
import re
import numpy as np
import lda
import lda.datasets

DATA_LEN = 12110
N = 10
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


def valid_index(db, engLines, i, year, errors):
    if not db.Test_Date[i].year == year:
        errors[YEAR_MISMATCH] = errors[YEAR_MISMATCH] + 1
        return False

    if not is_valid_text(db.text[i]):
        errors[INVALID_TEXT] = errors[INVALID_TEXT] + 1
        return False

    valid = True

    if not valid_translation(db, engLines, i):
        errors[INVALID_TRANS_RATIO] = errors[INVALID_TRANS_RATIO] + 1
        valid = False

    if get_officer_type(i) == NONE and db.officer[i] == 1:
        errors[INVALID_OFFICER] = errors[INVALID_OFFICER] + 1
        valid = False

    officer = db.officer[i]

    if officer == 1 and is_empty_text(db.TZIUN_KKZ[i]):
        errors[INVALID_OFFICER] = errors[INVALID_OFFICER] + 1
        valid = False

    if officer == 0 and not is_empty_text(db.TZIUN_KKZ[i]):
        errors[INVALID_SADIR] = errors[INVALID_SADIR] + 1
        valid = False

#    sium = db.OFEN_SIUM_KKZ[i]
#    if sium == MEDIC or sium == BIRO:
#        errors[INVALID_FAIL] = errors[INVALID_FAIL] + 1
#        valid = False

    return valid

def is_valid(word):
    non_interesting = ['with', 'this', 'that', 'these', 'the', 'and', 'for', 'thing', 'did', 'all', 'who', 'are', 'was',
                       'not', 'them', 'there', 'but', 'who', 'whom', 'its', 'they', 'from', 'how', 'has', 'which',
                       'when', 'what', 'his', 'her', 'she', 'him', 'had', 'have', 'because']
    names = ['arik', 'ariel', 'sharon', 'david', 'bengurion', 'klein', 'meir', 'dror', 'moshe', 'rabbeinu', 'hood',
             'robin', 'luther', 'martin', 'mota', 'gur', 'rabin', 'gandhi', 'yair', 'khan', 'herzl', 'miriam',
             'harzion', 'peretz']

    if len(word) < 3:
        return False

    if word in non_interesting:
        return False

    if word in names:
        return False

    if isinstance(word, (int, float)):
        return False

    return True


def get_users_and_words(myDb, engLines, year, outFile):
    users = []
    words = []
    cands = {}
    duplicate = 0
    errors = [0, 0, 0, 0, 0, 0]

    for i in range(0, DATA_LEN):
        #if i == 8407:
        #    stop = True

        print(i)
        if valid_index(myDb, engLines, i, year, errors):
            user = myDb.ID_coded[i]
            body = engLines[i].lower()
            if user not in users:
                users.append(user)
                cands[user] = i
                wlist = body.split()
                for word in wlist:
                    word = re.sub('[^a-zA-Z]+', '', word)
                    if is_valid(word) and word not in words:
                        words.append(word)
            else:
                #print("{} already in users".format(user))
                duplicate = duplicate + 1

    outFile.write("Users: {} (Diff Year : {}, Inv text: {}, T.ratio: {}, Dup: {}, Inv Off: {}, Inv Sadir {}, Inv Fail {})\n".format(users.__len__(),
                                                                                            errors[YEAR_MISMATCH],
                                                                                            errors[INVALID_TEXT],
                                                                                            errors[INVALID_TRANS_RATIO],
                                                                                            duplicate,
                                                                                            errors[INVALID_OFFICER],
                                                                                            errors[INVALID_SADIR],
                                                                                            errors[INVALID_FAIL]))
    return users, words, cands


def get_np_array(myDb, engLines, users, words, year):
    reviewed = []
    nparr = np.array([0] * len(users) * len(words))
    nparr = np.reshape(nparr, [len(users), len(words)])
    errors = [0, 0, 0, 0, 0, 0]

    for i in range(0, DATA_LEN):
        if valid_index(myDb, engLines, i, year, errors):
            user = myDb.ID_coded[i]
            body = engLines[i].lower()

            if user not in reviewed:
                reviewed.append(user)
                wlist = body.split()
                for word in wlist:
                    word = re.sub('[^a-zA-Z]+', '', word)
                    if is_valid(word):
                        nparr[users.index(user), words.index(word)] = nparr[users.index(user), words.index(word)] + 1

    # Code to check which users has no word added to the nparr
    #for i in range(users.__len__()):
    #    if sum(nparr[i]) == 0:
    #        print("SOMETHING IS WRONG ({})!!!".format(users[i]))

    return nparr


def print_topic_modeling_stats(db, doc_topic, users, cands, file):
    Otypes = []
    ot = OType(N, 0)
    dSet = DataSet("Summary")

    for i in range(4):
        Otypes.append(OType(N, i))

    for i in range(0, len(users)):
        line = cands[users[i]]
        cand = CandData(db, line)
        dSet.add_item(cand)
        Otypes[get_officer_type(line)].add_item(doc_topic[i], cand)
        ot.add_item(doc_topic[i], cand)


    file.write("\n{}\n".format(dSet.print_data()))
    file.write("{}\n".format(ot.print_topics()))
    for t in Otypes:
        file.write("{}\n".format(t.print_topics()))


def run_topic_modeling(cand_year, outFile):
    outFile.write("Results for {}\n".format(cand_year))

    db = get_cands_data('thesis_db.xls', DATA_LEN)
    engLines = get_translated_text("Translated_text.txt")

    users, words, cands = get_users_and_words(db, engLines, cand_year, outFile)
    #return 0
    X = get_np_array(db, engLines, users, words, cand_year)

    model = lda.LDA(n_topics=N, n_iter=300, random_state=1)
    model.fit(X)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 8

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(words)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        str = 'Topic {}: {}'.format(i, ' '.join(topic_words))
        print(str)
        outFile.write(str + '\n')

    print_topic_modeling_stats(db, model.doc_topic_, users, cands, outFile)
    outFile.write('\n')


outFile = open("output.txt", "w")
run_topic_modeling(2015, outFile)
run_topic_modeling(2020, outFile)
run_topic_modeling(2021, outFile)
outFile.close()

print("Done")