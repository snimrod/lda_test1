from xls_loader import get_cands_data
from xls_loader import get_translated_text
from TopicModeling_IDF import valid_index
from CandData import CandData
from gensim_lda import gensim_lda_run
from gensim_lda import gensim_load_model
from gensim_lda import gensim_apply_text_on_model
from gensim_lda import gensim_analyze_corpus
from infra import convert_locations
from xls_loader import is_empty_text

MAX_LINE = 12110
N = 10
YEAR = 2015
NAMES = ["kkz0", "kkz1", "kkz2", "kkz3", "all"]
DUMP_LOCATIONS = False
index2cand = {}


def dump_results(model, corpus, otype):
    header = "id,"
    for i in range(N):
        header = "{}t{},".format(header, i)
    header = "{}A10,A30,MAX,KKZ,KKZT,Officer,DAPAR,TZADAK,MAVDAK1,MAVDAK2,REJECTED,EXCEL,GRADE,SOCIO_TIRONUT,SOCIO_PIKUD".format(header)

    index = 0
    fname = "_gensim_{}_{}_topics.csv".format(NAMES[otype], N)
    f = open(fname, "w")
    f.write("{}\n".format(header))

    for key in index2cand:
        cand = index2cand[key]
        if otype == 4 or cand.otype == otype:
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

            if cand.otype == 0:
                kkz = 0
            else:
                kkz = 1

            f.write("{},{},{:.5f},{},{},{},{},{},{},{},{},{},{}\n".format(a10, a30, max, kkz, cand.otype, cand.officer,
                                                                          cand.dapar, cand.tzadak, cand.mavdak1,
                                                                          cand.mavdak2, cand.rejected, cand.excel,
                                                                          cand.grade, cand.socio_t, cand.socio_p))
            index = index + 1
    f.close()


def print_probabilities(model, corpus, otype):

    for i in corpus:
        print(model.get_document_topics(i, minimum_probability=0.00001))


def dump_big5():
    f = open("big5.csv", "w")
    h = "B1,B2,B3,B4,B5,dapar,tzadak,mavdak1,mavdak2,rejected,nf,ne,socioT,socioP\n"
    f.write("{}".format(h))
    db = get_cands_data('thesis_db.xls', MAX_LINE)
    reviewed_cands = []
    c = 0

    for line in range(MAX_LINE):
        this_cand_id = db.ID_coded[line]
        if this_cand_id not in reviewed_cands:
            reviewed_cands.append(this_cand_id)
            if not is_empty_text(db.bts_a[line]):
                c = c + 1
                cand = CandData(db, line)

                if cand.notEntered:
                    ne = 1
                else:
                    ne = 0

                if cand.notFinished:
                    nf = 1
                else:
                    nf = 0
                s = "{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(db.bts_a[line], db.bts_c[line],
                                                  db.bts_e[line], db.bts_n[line],db.bts_o[line], cand.dapar, cand.tzadak,
                                                  cand.mavdak1, cand.mavdak2, cand.rejected, nf, ne,
                                                                 cand.socio_t, cand.socio_p)
                f.write("{}\n".format(s))
                print(s)

    print("sum {}".format(c))
    f.close()


def load_lda(fname):
    model = gensim_load_model(fname)
    topicsf = open("_gensim_loaded_{}.txt".format(fname), "w")
    s = model.print_topics()
    for topic in s:
        topicsf.write("{}\n".format(topic))
    topicsf.close()


def run_lda(texts, otype, backup_name=''):
    size = len(texts)
    print("Valid Candidates ({}): {}".format(NAMES[otype], size))
    if size > 0:
        model, corp = gensim_lda_run(texts, N, backup_name)
        topicsf = open("_gensim_topics_{}.txt".format(NAMES[otype]), "w")
        s = model.print_topics()
        for topic in s:
            topicsf.write("{}\n".format(topic))
        topicsf.close()
        #dump_results(model, corp, otype)
        #for i in corp:
        #    print(model.get_document_topics(i, minimum_probability=0.1))


full_text = get_translated_text("Translated_text.txt")
text = full_text[:MAX_LINE]
db = get_cands_data('thesis_db.xls', MAX_LINE)
reviewed_cands = []
errors = [0, 0, 0, 0, 0, 0]

lda_text = [[], [], [], []]
accum_kkz_text = [""] * 4
accum_grade_text = [""] * 41
entire_text = []
sample_text = []

high = []
low = []

index = 0
for line in text:
    this_cand_id = db.ID_coded[index]
    if this_cand_id not in reviewed_cands:
        reviewed_cands.append(this_cand_id)
        if valid_index(db, full_text, index, YEAR, errors):
            cand = CandData(db, index)
            lda_text[cand.otype].append(line)
            #accum_kkz_text[cand.otype] = accum_kkz_text[cand.otype] + " " + line
            if cand.grade == "":
                accum_grade_text[0] = accum_grade_text[0] + line
            else:
                ii = int(cand.grade) - 59
                accum_grade_text[ii] = accum_grade_text[ii] + line
            entire_text.append(line)
            if len(sample_text) < 100 and not cand.otype == 0:
                sample_text.append(line)
            index2cand[index] = cand
            #if cand.otype == 3 and not cand.grade == "" and int(cand.grade) > 83:
            #entire_text.append(line)

            #if cand.otype == 3 and not cand.grade == "" and int(cand.grade) < 80:
            #   low.append(line)
    index = index + 1

#dump_big5()

#high_d, high_l, id2word_h = gensim_analyze_corpus(high, "_high_grades.txt")
#low_d, low_l, id2word_l = gensim_analyze_corpus(low, "_low_grades.txt")

#for key in high_d:
#    if high_d[key] > 30:
#        word = id2word_h[key]
#        low_key = id2word_l.token2id[word]
#        if low_key in low_d:
#            ph = high_d[key] / len(high)
#            pl = low_d[low_key] / len(low)
#            if ph >= 1.8 * pl:
#                print("{}) {} ({:.5f}) VS  {} ({:.5f}) ".format(id2word_h[key], high_d[key], ph, low_d[low_key], pl))

#for i in range(4):
#    if len(lda_text[i]) > 0:
#        name = "_{}_high_grade.txt".format(NAMES[i])
#        print("{} size: {}".format(name, len(lda_text[i])))
#        gensim_analyze_corpus(lda_text[i], name)


#for i in range(4):
#    run_lda(lda_text[i], i)

#run_lda(entire_text, 0)

accum_text = []
for t in accum_grade_text:
    if not len(t) == 0:
        accum_text.append(t)

#run_lda(accum_text, 1, 'my_lda_backup')
#load_lda('my_lda_backup')

run_lda(lda_text[0], 0, 'lda0_backup')
gensim_apply_text_on_model('lda0_backup', sample_text)


print("Done")
