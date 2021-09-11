from xls_loader import get_cands_data
from xls_loader import get_translated_text
from TopicModeling_IDF import valid_index
from CandData import CandData
from gensim_lda import gensim_lda_run
from gensim_lda import gensim_analyze_corpus
from infra import convert_locations

MAX_LINE = 12110
N = 5
YEAR = 2015
NAMES = ["kkz0", "kkz1", "kkz2", "kkz3"]
DUMP_LOCATIONS = False
index2cand = {}


def dump_results(model, corpus, otype):
    header = "id,"
    for i in range(N):
        header = "{}t{},".format(header, i)
    header = "{}A10,A30,MAX,Officer,DAPAR,TZADAK,MAVDAK1,MAVDAK2,REJECTED,EXCEL,GRADE,SOCIO_TIRONUT,SOCIO_PIKUD".format(header)

    index = 0
    fname = "_gensim_{}_{}_topics.csv".format(NAMES[otype], N)
    f = open(fname, "w")
    f.write("{}\n".format(header))

    for key in index2cand:
        cand = index2cand[key]
        if cand.otype == otype:
            a10 = 0
            a30 = 0
            max = 0
            f.write("{},".format(cand.id))
            probabilities = model.get_document_topics(corpus[index], minimum_probability=0.00001)
            prob_list = [prob[1] for prob in probabilities]
            locations = convert_locations(prob_list)
            for p in probabilities:
                if DUMP_LOCATIONS:
                    f.write("{},".format(locations[p[0]]))
                else:
                    f.write("{:.5f},".format(p[1]))
                if p[1] >= 0.1:
                    a10 = a10 + 1
                if p[1] >= 0.3:
                    a30 = a30 + 1
                if p[1] > max:
                    max = p[1]

            index = index + 1
            f.write("{},{},{:.5f},{},{},{},{},{},{},{},{},{},{}\n".format(a10, a30, max, cand.officer, cand.dapar,
                                                                          cand.tzadak, cand.mavdak1, cand.mavdak2,
                                                                          cand.rejected, cand.excel, cand.grade,
                                                                          cand.socio_t, cand.socio_p))
    f.close()


def run_lda(texts, otype):
    size = len(texts)
    print("Valid Candidates ({}): {}".format(NAMES[otype], size))
    if size > 0:
        model, corp = gensim_lda_run(texts, N)
        topicsf = open("_gensim_topics_{}.txt".format(NAMES[otype]), "w")
        s = model.print_topics()
        for topic in s:
            topicsf.write("{}\n".format(topic))
        topicsf.close()
        dump_results(model, corp, otype)


full_text = get_translated_text("Translated_text.txt")
text = full_text[:MAX_LINE]
db = get_cands_data('thesis_db.xls', MAX_LINE)
reviewed_cands = []
errors = [0, 0, 0, 0, 0, 0]

lda_text = [[], [], [], []]

high = []
low = []

index = 0
for line in text:
    this_cand_id = db.ID_coded[index]
    if this_cand_id not in reviewed_cands:
        reviewed_cands.append(this_cand_id)
        if valid_index(db, full_text, index, YEAR, errors):
            cand = CandData(db, index)
            # lda_text[cand.otype].append(line)
            index2cand[index] = cand
            if cand.otype == 3 and not cand.grade == "" and int(cand.grade) > 83:
                high.append(line)

            if cand.otype == 3 and not cand.grade == "" and int(cand.grade) < 80:
                low.append(line)
    index = index + 1

high_d, high_l, id2word_h = gensim_analyze_corpus(high, "_high_grades.txt")
low_d, low_l, id2word_l = gensim_analyze_corpus(low, "_low_grades.txt")

for key in high_d:
    if high_d[key] > 30:
        word = id2word_h[key]
        low_key = id2word_l.token2id[word]
        if low_key in low_d:
            ph = high_d[key] / len(high)
            pl = low_d[low_key] / len(low)
            if ph >= 1.8 * pl:
                print("{}) {} ({:.5f}) VS  {} ({:.5f}) ".format(id2word_h[key], high_d[key], ph, low_d[low_key], pl))

#for i in range(4):
#    if len(lda_text[i]) > 0:
#        name = "_{}_high_grade.txt".format(NAMES[i])
#        print("{} size: {}".format(name, len(lda_text[i])))
#        gensim_analyze_corpus(lda_text[i], name)


#for i in range(4):
#    run_lda(lda_text[i], i)

print("Done")
