from xls_loader import get_cands_data
from xls_loader import get_translated_text
from xls_loader import is_valid_text
from xls_loader import is_empty_text
from OType import OType
from OType import get_officer_type
from OType import NONE
from DataSet import DataSet
from CandData import CandData
from gensim_lda import get_data_lemmatized
from gensim_lda import text2corpus
import re
import numpy as np
import lda
import lda.datasets

DATA_LEN = 1110
N = 15
ITERATIONS = 1500
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

    #if cand.officer == 1 and cand.grade == 0:
    #    errors[INVALID_OFFICER] = errors[INVALID_OFFICER] + 1
    #    valid = False

    if cand.officer == 0 and not cand.grade == 0:
        errors[INVALID_SADIR] = errors[INVALID_SADIR] + 1
        valid = False

    if valid:
        return cand
    else:
        return None


def valid_index(db, engLines, i, year, errors):

#    if db.ID_coded[i] == 13821367:
#        debugon = 1

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

    officer = db.officer[i]

    if get_officer_type(i) == NONE and officer == 1:
        errors[INVALID_OFFICER] = errors[INVALID_OFFICER] + 1
        valid = False

    if officer == 1 and is_empty_text(db.TZIUN_KKZ[i]):
        errors[INVALID_OFFICER] = errors[INVALID_OFFICER] + 1
        valid = False

    if officer == 0 and not is_empty_text(db.TZIUN_KKZ[i]):
        errors[INVALID_SADIR] = errors[INVALID_SADIR] + 1
        valid = False

    sium = db.OFEN_SIUM_KKZ[i]
    if sium == MEDIC or sium == BIRO:
        errors[INVALID_FAIL] = errors[INVALID_FAIL] + 1
        valid = False

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

    #if word in names:
    #    return False

    if isinstance(word, (int, float)):
        return False

    return True


def get_users_and_words(myDb, engLines, year, outFile):
    users = []
    words = []
    cands = {}
    duplicate = 0
    errors = [0, 0, 0, 0, 0, 0]

    rawLines = get_translated_text("Translated_text.txt")

    for i in range(0, DATA_LEN):
        #if i == 8407:
        #    stop = True

        print(i)
        if valid_index(myDb, rawLines, i, year, errors):
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

    #outFile.write("Users: {} (Diff Year : {}, Inv text: {}, T.ratio: {}, Dup: {}, Inv Off: {}, Inv Sadir {}, Inv Fail {})\n".format(users.__len__(),
    #                                                                                        errors[YEAR_MISMATCH],
    #                                                                                        errors[INVALID_TEXT],
    #                                                                                        errors[INVALID_TRANS_RATIO],
    #                                                                                        duplicate,
    #                                                                                        errors[INVALID_OFFICER],
    #                                                                                        errors[INVALID_SADIR],
    #                                                                                        errors[INVALID_FAIL]))
    return users, words, cands


def get_np_array(myDb, engLines, users, words, year):
    reviewed = []
    nparr = np.array([0] * len(users) * len(words))
    nparr = np.reshape(nparr, [len(users), len(words)])
    errors = [0, 0, 0, 0, 0, 0]

    rawLines = get_translated_text("Translated_text.txt")

    for i in range(0, DATA_LEN):
        if valid_index(myDb, rawLines, i, year, errors):
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
    #engLines = get_translated_text("Translated_text.txt")
    engLines = get_translated_text("lemmatized_db.txt")

    users, words, cands = get_users_and_words(db, engLines, cand_year, outFile)
    X = get_np_array(db, engLines, users, words, cand_year)

    model = lda.LDA(n_topics=N, n_iter=ITERATIONS, random_state=1)
    model.fit(X)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 8

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(words)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        str = 'Topic {}: {}'.format(i, ' '.join(topic_words))
        print(str)
        outFile.write(str + '\n')

    #print_topic_modeling_stats(db, model.doc_topic_, users, cands, outFile)
    outFile.write('\n')


def dump_tm(db, doc_topic, users, cands, files):
    header = "id, KKZ, W_KKZ, "
    for i in range(N):
        header = "{}t{}, ".format(header, i)
    header = "{}A1, A10, A20, A30, A40, Single, Officer, DAPAR, TZADAK, MAVDAK1, MAVDAK2, REJECTED, EXCEL, GRADE, SOCIO_TIRONUT, SOCIO_PIKUD".format(header)

    for f in files:
        f.write("{}\n".format(header))

    for i in range(0, len(users)):
        line = cands[users[i]]
        oType = get_officer_type(line)
        cand = CandData(db, line)

        if oType == NONE:
            w_kkz = 0
        else:
            w_kkz = 1

        a1 = 0
        a10 = 0
        a20 = 0
        a30 = 0
        a40 = 0
        single = 0

        x = []
        ti = 0
        for n in doc_topic[i]:
            x.append([n, ti])
            ti = ti + 1
        x.sort()
        loc = [0] * N

        for tn in range(1, N):
            if x[tn][0] == x[tn-1][0]:
                loc[x[tn][1]] = loc[x[tn-1][1]]
            else:
                loc[x[tn][1]] = tn

        str = "{}, {}, {}, ".format(db.ID_coded[line], oType, w_kkz, )
        for topic in range(N):
            #str = "{}{:.5f}, ".format(str, doc_topic[i][topic])
            str = "{}{}, ".format(str, loc[topic])

            p = doc_topic[i][topic]
            if p >= 0.01:
                a1 = a1 + 1
            if p >= 0.1:
                a10 = a10 + 1
            if p >= 0.2:
                a20 = a20 + 1
            if p >= 0.3:
                a30 = a30 + 1
            if p >= 0.4:
                a40 = a40 + 1
            if p >= 0.7:
                single = 1

        str = "{}{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(str, a1, a10, a20, a30, a40, single, cand.officer, cand.dapar, cand.tzadak,
                                                                  cand.mavdak1, cand.mavdak2, cand.rejected, cand.excel,
                                                                  cand.grade, cand.socio_t, cand.socio_p)
        files[0].write("{}".format(str))
        files[oType + 1].write("{}".format(str))


def corpus2nparray(corpus, id2word):
    nparr = np.array([0] * id2word.num_docs * len(id2word))
    nparr = np.reshape(nparr, [id2word.num_docs, len(id2word)])
    doc_i = 0
    for doc in corpus:
        for t in doc:
            nparr[doc_i, t[0]] = t[1]
        doc_i = doc_i + 1
    return nparr


def run_tm_and_dump(cand_year, files):
    db = get_cands_data('thesis_db.xls', DATA_LEN)
    engLines = get_translated_text("Translated_text.txt")

    engLines = engLines[:DATA_LEN]


    index = 0
    reviewed_cands = []
    cand_ids = []
    index2cand = {}
    run_text = []
    errors = [0, 0, 0, 0, 0, 0]
    for line in engLines:
        this_cand_id = db.ID_coded[index]
        if this_cand_id not in reviewed_cands:
            reviewed_cands.append(this_cand_id)
            cand = get_cand(db, engLines, index, [cand_year], errors)
            if cand is not None:
                run_text.append(line)
                cand_ids.append(cand.id)
                index2cand[index] = cand
        index = index + 1

    lem_text = get_data_lemmatized(run_text)
    id2word, corpus = text2corpus(lem_text)
    X2 = corpus2nparray(corpus, id2word)

    users, words, cands = get_users_and_words(db, engLines, cand_year, files[0])
    X = get_np_array(db, engLines, users, words, cand_year)

    model = lda.LDA(n_topics=N, n_iter=ITERATIONS, random_state=1)
    model.fit(X2)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 8

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(words)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        str = 'Topic {}: {}'.format(i, ' '.join(topic_words))
        print(str)

    #dump_tm(db, model.doc_topic_, users, cands, files)
    rr= 6
    #outFile.write('\n')


outFile = open("topic_modeling_idf.csv", "w")
#f1 = open("topic_modeling_idf_1.csv", "w")
#f2 = open("topic_modeling_idf_2.csv", "w")
#f3 = open("topic_modeling_idf_3.csv", "w")
#f4 = open("topic_modeling_idf_4.csv", "w")
#files = [outFile, f1, f2, f3, f4]
files = [outFile]

#run_tm_and_dump(2015, files)
#run_topic_modeling(2015, outFile)
#run_topic_modeling(2020, outFile)
#run_topic_modeling(2021, outFile)
outFile.close()
#f1.close()
#f2.close()
#f3.close()
#f4.close()
#print("Done")
