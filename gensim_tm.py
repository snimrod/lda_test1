from xls_loader import get_cands_data
from xls_loader import get_translated_text
from TopicModeling_IDF import valid_index
from CandData import CandData
from gensim_lda import gensim_lda_run
from pprint import pprint

MAX_LINE = 12110
N = 15
YEAR = 2015
NAMES = ["kkz0", "kkz1", "kkz2", "kkz3"]
index2cand = {}


def dump_results(model, corpus, otype):
    header = "id,"
    for i in range(N):
        header = "{}t{},".format(header, i)
    #header = "{}A1, A10, A20, A30, A40, Single, Officer, DAPAR, TZADAK, MAVDAK1, MAVDAK2, REJECTED, EXCEL, GRADE, SOCIO_TIRONUT, SOCIO_PIKUD".format(header)
    header = "{}Officer, DAPAR, TZADAK, MAVDAK1, MAVDAK2, REJECTED, EXCEL, GRADE, SOCIO_TIRONUT, SOCIO_PIKUD".format(header)

    index = 0
    fname = "_gensim_{}_{}_topics.csv".format(NAMES[otype], N)
    f = open(fname, "w")
    f.write("{}\n".format(header))

    for key in index2cand:
        cand = index2cand[key]
        if cand.otype == otype:
            f.write("{},".format(cand.id))
            probabilities = model.get_document_topics(corpus[index], minimum_probability=0.00001)
            for p in probabilities:
                f.write("{:.5f},".format(p[1]))
            index = index + 1
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(cand.officer, cand.dapar, cand.tzadak, cand.mavdak1,
                                                             cand.mavdak2, cand.rejected, cand.excel, cand.grade,
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

#kkz0_text = []
#kkz1_text = []
#kkz2_text = []
#kkz3_text = []
#lda_text = [kkz0_text, kkz1_text, kkz2_text, kkz3_text]
lda_text = [[], [], [], []]

index = 0
for line in text:
    this_cand_id = db.ID_coded[index]
    if this_cand_id not in reviewed_cands:
        reviewed_cands.append(this_cand_id)
        if valid_index(db, full_text, index, YEAR, errors):
            cand = CandData(db, index)
#            otype = get_officer_type(index)
            lda_text[cand.otype].append(line)
            index2cand[index] = cand
    index = index + 1

for i in range(4):
    run_lda(lda_text[i], i)

#run_lda(lda_text, "suf")
#if len(lda_text) > 0:
#    model, corp = gensim_lda_run(lda_text, N)
#    pprint(model.print_topics())
#    print("\n")
#    dump_results(model, corp, index_to_cand)

print("Done")
# print(model.get_document_topics(corp[6], minimum_probability=0.01))