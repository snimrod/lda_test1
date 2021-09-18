import datetime
from xls_loader import get_cands_data
from xls_loader import get_translated_text
#from TopicModeling_IDF import valid_index
from TopicModeling_IDF import get_cand
from CandData import CandData
from gensim_lda import gensim_lda_run
from gensim_lda import gensim_load_model
from gensim_lda import gensim_apply_text_on_model
from gensim_lda import gensim_analyze_corpus
from infra import convert_locations
from infra import fill_zeros
from xls_loader import is_empty_text

MAX_LINE = 120
N = 10
YEAR = 2015
NAMES = ["kkz0", "kkz1", "kkz2", "kkz3", "all"]
DUMP_LOCATIONS = False
index2cand = {}
sample1_index2cand = {}
sample2_index2cand = {}



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


def dump_single_run_results(model, corpus, otype):
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
            f.write("{},{},{:.5f},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(a10, a30, max, kkz, cand.otype, cand.officer,
                                                                          cand.dapar, cand.tzadak, cand.mavdak1,
                                                                          cand.mavdak2, cand.rejected, cand.excel,
                                                                          cand.grade, cand.socio_t, cand.socio_p))
            index = index + 1
    f.close()


def run_lda(texts, otype, to_dump, backup_name=''):
    size = len(texts)
    print("Valid Candidates ({}): {}".format(NAMES[otype], size))
    if size > 0:
        model, corp = gensim_lda_run(texts, N, backup_name)
        topicsf = open("_gensim_topics_{}.txt".format(NAMES[otype]), "w")
        s = model.print_topics()
        for topic in s:
            topicsf.write("{}\n".format(topic))
        topicsf.close()
        if to_dump:
            dump_single_run_results(model, corp, otype)
        return model
    else:
        return None


def dump_applied_results(d_lists, i2c, name_suf):
    print(d_lists)
    header = "id,"
    for i in range(N):
        header = "{}t{},".format(header, i)
    header = "{}A10,A30,MAX,KKZ,KKZT,Officer,DAPAR,TZADAK,MAVDAK1,MAVDAK2,REJECTED,EXCEL,GRADE,SOCIO_TIRONUT,SOCIO_PIKUD".format(header)

    index = 0
    fname = "_gensim_applied_{}_{}_topics.csv".format(name_suf, N)
    f = open(fname, "w")
    f.write("{}\n".format(header))

    for key in i2c:
        cand = i2c[key]
        a10 = 0
        a30 = 0
        max = 0
        f.write("{},".format(cand.id))
        #probabilities = model.get_document_topics(corpus[index], minimum_probability=0.00001)
        #prob_list = [prob[1] for prob in probabilities]
        prob_list = d_lists[index]
        locations = convert_locations(prob_list)
        for p in prob_list:
            if DUMP_LOCATIONS:
                f.write("{},".format(locations[index]))
            else:
                f.write("{:.5f},".format(p))
            if p >= 0.1:
                a10 = a10 + 1
            if p >= 0.3:
                a30 = a30 + 1
            if p > max:
                max = p

        if cand.otype == 0:
            kkz = 0
        else:
            kkz = 1
        f.write("{},{},{:.5f},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(a10, a30, max, kkz, cand.otype, cand.officer,
                                                                      cand.dapar, cand.tzadak, cand.mavdak1,
                                                                      cand.mavdak2, cand.rejected, cand.excel,
                                                                      cand.grade, cand.socio_t, cand.socio_p))
        index = index + 1
    f.close()


def apply_new_text_on_train_model(train_model, new_text, i2c, name_suf):
    vectors = gensim_apply_text_on_model(train_model, new_text)
    dist_lists = []
    for v in vectors:
        dist_lists.append(fill_zeros(v, N))
    dump_applied_results(dist_lists, i2c, name_suf)


print(datetime.datetime.now())
#full_text = get_translated_text("Translated_text.txt")
full_text = get_translated_text("fake_translated.txt")
text = full_text[:MAX_LINE]
db = get_cands_data('fake_db.xls', MAX_LINE)
reviewed_cands = []
errors = [0, 0, 0, 0, 0, 0]
lda_text = [[], [], [], []]
accum_kkz_text = [""] * 4
accum_grade_text = [""] * 41
entire_text = []
sample1_text = []
sample2_text = []
# Accumulated train text is gathered by certain criteria (e.g grade) and every index in it is a value that accumulates
# text matching this value, There are no values higher than 100 for any criteria
accum_train_text = [""] * 100
accum_cands = 0

high = []
low = []

index = 0
for line in text:
    this_cand_id = db.ID_coded[index]
    if this_cand_id not in reviewed_cands:
        reviewed_cands.append(this_cand_id)
        cand = get_cand(db, full_text, index, YEAR, errors)
        if cand is not None:
            entire_text.append(line)
            #lda_text[cand.otype].append(line)
            index2cand[index] = cand

            sample1_index = len(sample1_text)
            sample2_index = len(sample2_text)
            #if (sample1_index < 100 and cand.otype == 0) or (sample2_index < 100 and cand.otype == 3):
            if index > 89:
                # Cand goes to applied sample
            #    if sample1_index < 100 and cand.otype == 0:
                sample1_text.append(line)
                sample1_index2cand[sample1_index] = cand
            #    else:
            #        sample2_text.append(line)
            #        sample2_index2cand[sample2_index] = cand

            else:
                # Cand goes to accumulate train text
                if cand.grade == 0:
                    ii = 0
                else:
                    ii = cand.grade - 59
                accum_train_text[ii] = accum_train_text[ii] + line
                accum_cands = accum_cands + 1


            #if cand.otype == 3 and not cand.grade == "" and int(cand.grade) > 83:
            #entire_text.append(line)

            #if cand.otype == 3 and not cand.grade == "" and int(cand.grade) < 80:
            #   low.append(line)
    index = index + 1


print(errors)
#for i in range(4):
#    run_lda(lda_text[i], i)
#run_lda(entire_text, 4, True)

print("Total {}, Train {}, Sample1 {}, Sample2 {}".format(len(entire_text), accum_cands, len(sample1_text), len(sample2_text)))

# Ignore values which did not accumulated any text
accum_text = []
for t in accum_train_text:
    if not len(t) == 0:
        accum_text.append(t)

t_model = run_lda(accum_text, 4, False)
if len(sample1_text) > 0:
    apply_new_text_on_train_model(t_model, sample1_text, sample1_index2cand, "last30")
#if len(sample2_text) > 0:
#    apply_new_text_on_train_model(t_model, sample2_text, sample2_index2cand, "2")

#load_lda('my_lda_backup')

#run_lda(lda_text[0], 0, 'lda0_backup')
#gensim_apply_text_on_model('lda0_backup', sample_text)

print(datetime.datetime.now())
print("Done")

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
