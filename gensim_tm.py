import datetime
#import gensim.corpora as corpora
from xls_loader import get_cands_data
from xls_loader import get_translated_text
from infra import get_cand
from CandData import CandData
from gensim_lda import gensim_lda_run
from gensim_lda import gensim_load_model
from gensim_lda import gensim_apply_text_on_model
from gensim_lda import gensim_apply_text_on_model2
from gensim_lda import gensim_analyze_corpus
from gensim_lda import dump_data_lemmatized
from gensim_lda import get_data_lemmatized
from gensim_lda import text2corpus
from infra import convert_locations
from infra import fill_zeros
from xls_loader import is_empty_text
from keras_lda import keras_lda_run
import numpy as np
from sklearn import linear_model
from xls_loader import load_characters
import random
from operator import itemgetter

MAX_LINE = 12110
N = 15
YEARS = [2015]
NAMES = ["kkz0", "kkz1", "kkz2", "kkz3", "all"]
DUMP_LOCATIONS = False
DUMP_MY_MATCHES = False
TOPIC_WORDS = 20
id2matches = {}
index2cand = {}
sample1_index2cand = {}
sample2_index2cand = {}
G = "gensim"
K = "keras"
clf = linear_model.LogisticRegression(C=1e5)
LR_train_size = 1500
LR_train_samples = []
LR_train_categories = []
LR_predict_samples = []
LR_predict_categories = []

distributions = np.array([0] * 90).reshape(2, 3, 15)

# Genders
FEMALE = 0
MALE = 1
NO_GENDER = 2

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
        f.write("{},{},".format(cand.id, cand.year))
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
    vectors = gensim_apply_text_on_model2(train_model, new_text[0])
    #dist_lists = []
    #for v in vectors:
    #    dist_lists.append(fill_zeros(v, N))
    #dump_applied_results(dist_lists, i2c, name_suf)


def load_lda(fname):
    model = gensim_load_model(fname)
    topicsf = open("_gensim_loaded_{}.txt".format(fname), "w")
    s = model.print_topics()
    for topic in s:
        topicsf.write("{}\n".format(topic))
    topicsf.close()


def create_word2prob_for_topic(topic_str):
    words = {}
    words_and_probs = topic_str[1].split(' + ')
    for item in words_and_probs:
        t = item.split('*')
        words[t[1].strip('"')] = float(t[0])

    return words, topic_str[0]


def create_word2prob_dicts(model):
    word2prob_dicts_list = [[]] * N
    s = model.print_topics(num_topics=N, num_words=TOPIC_WORDS)

    for topic in s:
        w2p, topic_i = create_word2prob_for_topic(topic)
        word2prob_dicts_list[topic_i] = w2p

    return word2prob_dicts_list


def lr_by_text():
    train_cnt = [0, 0, 0, 0]
    predict_cnt = [0, 0]
    len_ctrs = {}
    ratio_ctrs = {}
    cnt = 0
    for c in index2cand.values():

        #t = c.hebText.split()
        #id2word = corpora.Dictionary([t])
        #unique_ratio = len(id2word)/len(t)
        #print("{} {} {}".format(len(t), len(id2word), unique_ratio))
        #sample = [len(t), unique_ratio]
        len_key = int(c.words_n/10)
        ratio_key = int(c.unique_ratio * 100)

        #if c.dapar >= 80:
        #    category = 1
        #else:
        #    category = 0
        category = c.sex

        if len_key not in len_ctrs:
            len_ctrs[len_key] = [0, 0]

        len_ctrs[len_key][category] = len_ctrs[len_key][category] + 1

        if ratio_key not in ratio_ctrs:
            ratio_ctrs[ratio_key] = [0, 0]

        ratio_ctrs[ratio_key][category] = ratio_ctrs[ratio_key][category] + 1

        sample = [c.words_n]
        train_quota = [2200, 2200, 220, 220]
        predict_quota = [100, 100]
        if c.words_n < 25:
            p_group = 0
            if c.sex == 0:
                group = 0
            else:
                group = 1
        else:
            p_group = 1
            if c.sex == 0:
                group = 2
            else:
                group = 3

        group = category
        p_group = category
        cnt = cnt + 1
        if train_cnt[group] < train_quota[group]:
            LR_train_samples.append(sample)
            LR_train_categories.append(category)
            train_cnt[group] = train_cnt[group] + 1
        elif predict_cnt[p_group] < predict_quota[p_group]:
            LR_predict_samples.append(sample)
            LR_predict_categories.append(category)
            predict_cnt[p_group] = predict_cnt[p_group] + 1

    print(cnt)
    print("ratio:\n")
    for k in sorted(ratio_ctrs.keys()):
        i = ratio_ctrs[k]
        if sum(i) == 0:
            pcnt = 0
        else:
            pcnt = 100 * i[0] / sum(i)
        print("{}) {:.1f}% ({}/{})".format(k, pcnt, i[0], sum(i)))
    print("\n")

    print("len:\n")
    for k in sorted(len_ctrs.keys()):
        i = len_ctrs[k]
        if sum(i) == 0:
            pcnt = 0
        else:
            pcnt = 100 * i[0] / sum(i)
        print("{}) {:.1f}% ({}/{})".format(k, pcnt, i[0], sum(i)))

    run_linear_regression()


def calc_total_match(words, dict):
    match = 0
    match_str = ""

    for word in words:
        if word in dict:
            match = match + dict[word]
            match_str = "{} {}*{} ".format(match_str, word, dict[word])

    return match, match_str


def dump_cand_matches(cand, cand_words, word2prob, f, calc_f, topics_n, c_index, gensim_probs):
    cand_matches = []
    f.write("{}) Candidate {}:\n".format(c_index, cand.id))
    calc_f.write("{}) Candidate {}:\n".format(c_index, cand.id))

    prob_list = [prob[1] for prob in gensim_probs]
    locations = convert_locations(prob_list)

    i = 0
    for w2p in word2prob:
        p = gensim_probs[i]
        match, match_str = calc_total_match(cand_words, w2p)
        calc_f.write("{}) [{:.3f}] - {}\n".format(i, match, match_str))
        if i == 0:
            zero_match = match
        cand_matches.append(match)
        f.write("Topic {} : words matches = {:.3f}, engine_prob = {:.3f}, engine location = {}\n".format(i, match, p[1], locations[p[0]]))
        i = i + 1

    f.write("\n")
    zero_first = (zero_match == max(cand_matches))
    #if zero_first and cand.sex == 1:
    #    print("{} - {}".format(cand.id, cand_words))
    return cand_matches, zero_first


def dump_matches(engine, model, corpus, texts, word2prob_list, otype, topics_n):
    lemmatized_text = get_data_lemmatized(texts)
    calc_f = open("_{}_match_calc_details.txt".format(engine), "w")
    fname = "_{}_cands_matches_{}_{}_topics".format(engine, NAMES[otype], topics_n)
    f = open(fname, "w")
    fzl = []
    mzl = []
    males = [0, 0, 0, 0]
    females = [0, 0, 0, 0]

    i = 0
    for key in index2cand:
        cand = index2cand[key]
        if otype == 4 or cand.otype == otype:
            if cand.id == 11847784:
                stop = True
            probabilities = get_probabilities(engine, model, i, corpus)
            cand_matches, zero_first = dump_cand_matches(cand, lemmatized_text[i], word2prob_list, f, calc_f, topics_n, i, probabilities)
            id2matches[cand.id] = cand_matches
            if zero_first:
                if cand.sex == 0:
                    fzl.append(cand.id)
                    females[cand.otype] = females[cand.otype] + 1
                else:
                    mzl.append(cand.id)
                    males[cand.otype] = males[cand.otype] + 1
            i = i + 1

    f_z = len(fzl)
    total_z = f_z + len(mzl)
    #print("f%={} -  {}/{}".format(f_z/total_z, f_z, total_z))
    #print(males)
    #print(females)
    #print(fzl)
    #print(mzl)
    calc_f.close()
    f.close()


def get_probabilities(engine, model, doc_index, corpus=None):
    if engine == G:
        return model.get_document_topics(corpus[doc_index], minimum_probability=0.00001)
    else:
        return list((i, prob) for i, prob in enumerate(model.doc_topic_[doc_index]))


def print_list_distribution(l):
    dist = [0] * N
    for probs in l:
        #cand = t[0]
        #probs = t[1]
        m = 0
        first_topic = 0
        for p in probs:
            if p[1] > m:
                m = p[1]
                first_topic = p[0]
        dist[first_topic] = dist[first_topic] + 1
    s = sum(dist)
    print(s)
    if s > 0:
        dp = [round(x/s, 3) for x in dist]
        print(dp)


def dump_distributions():
    f = open("characters_detailed_distribution.csv", "w")
    for g in range(2):
        for cg in range(3):
            output = ""
            for topic in range(N):
                output = "{}{}, ".format(output, distributions[g, cg, topic])
            f.write("{}-{},{}\n".format(g, cg, output))
    f.close()


def update_distributions(cand, topic_id):
    distributions[cand.sex, cand.charg, topic_id] = distributions[cand.sex, cand.charg, topic_id] + 1


def dump_single_run_results(engine, model, corpus, otype, topics_n):
    train_cnt = [0, 0, 0, 0]
    predict_cnt = [0, 0]
    r = 0
    e = 0
    lists_by_sex_excel = np.empty((2, 2), object)
    lists_by_sex_excel[0, 0] = []
    lists_by_sex_excel[0, 1] = []
    lists_by_sex_excel[1, 0] = []
    lists_by_sex_excel[1, 1] = []

    lists_by_sex_off = np.empty((2, 2), object)
    lists_by_sex_off[0, 0] = []
    lists_by_sex_off[0, 1] = []
    lists_by_sex_off[1, 0] = []
    lists_by_sex_off[1, 1] = []

    lists_by_sex = [[], []]
    lists_by_excel = [[], []]

    lists_by_grades = [[], [], [], [], [], []]
    samples = [0, 0]
    gc = [0,0]
    boysd = [0] * N
    girlsd = [0] * N
    boysoffd = [0] * N
    girlsoffd = [0] * N
    maxcnt = 0


    header = "id,gender,char_g, char_t,year,"
    for i in range(topics_n):
        header = "{}t{}F,".format(header, i)
    for i in range(topics_n):
        header = "{}t{}V,".format(header, i)
    for i in range(topics_n):
        header = "{}t{}L,".format(header, i)
    header = "{}KKZ,KKZT,Officer,DAPAR,TZADAK,MAVDAK1,MAVDAK2,REJECTED,EXCEL,GRADE,SOCIO_TIRONUT,SOCIO_PIKUD,WORDS_N,UNIQUE_R".format(header)

    fname = "_{}_{}_{}_topics_dist.csv".format(engine, NAMES[otype], topics_n)
    f = open(fname, "w")
    f.write("{}\n".format(header))

    grps = np.array([0.0] * 30).reshape(2, 15)
    grps_cnt = [0, 0]

    for i, key in enumerate(index2cand):
        cand = index2cand[key]
        #if not cand.charg == NO_GENDER:
        #    continue

        grp_id = cand.sex
        grps_cnt[grp_id] = grps_cnt[grp_id] + 1

        if otype == 4 or cand.otype == otype:
        #if cand.excel or cand.rejected:
            # NOTICE: Assumes dump_matches was already called before
            my_matches = id2matches[cand.id]

            if cand.id == 11846665:
                stopp = 1

            f.write("{},{},{},{},{},".format(cand.id, cand.sex, cand.charg, cand.chart, cand.year))
            probabilities = get_probabilities(engine, model, i, corpus)
            cand.probs = probabilities

            if DUMP_MY_MATCHES:
                prob_list = my_matches
            else:
                prob_list = [prob[1] for prob in probabilities]

            #if DUMP_LOCATIONS:
            locations = convert_locations(prob_list)
            r_locations = convert_locations(prob_list, True)

            ## Creating lists for groups diff
            #lists_by_sex_excel[cand.sex, cand.excel].append(cand_probs)
            #lists_by_sex_off[cand.sex, cand.officer].append(cand_probs)
            #lists_by_sex[cand.sex].append(cand_probs)
            #lists_by_excel[cand.excel].append(cand_probs)

            #if cand.officer == 0:
            #    lists_by_grades[1].append(cand_probs)
            #    if cand.rejected:
            #        lists_by_grades[0].append(cand_probs)
            #else:
            #    lists_by_grades[3].append(cand_probs)
            #    if cand.grade < 70:
            #        lists_by_grades[2].append(cand_probs)
            #    if cand.grade > 80:
            #        lists_by_grades[4].append(cand_probs)
            #    if cand.excel:
            #        lists_by_grades[5].append(cand_probs)

            # Print 'is first' indication + update distributions tracker
            for p in probabilities:
                if r_locations[p[0]] == 0:
                    f.write("1,")
                    update_distributions(cand, p[0])
                else:
                    f.write("0,")

            # Print probabilities
            for ti, p in enumerate(probabilities):
                f.write("{:.5f},".format(p[1]))
                grps[grp_id][ti] = grps[grp_id][ti] + p[1]


            # Print locations
            for p in probabilities:
                f.write("{},".format(locations[p[0]]))

            gc[cand.sex] = gc[cand.sex] + 1


            if cand.otype == 0:
                kkz = 0
            else:
                kkz = 1
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4f}\n".format(kkz, cand.otype, cand.officer,
                                                                          cand.dapar, cand.tzadak, cand.mavdak1,
                                                                          cand.mavdak2, cand.rejected, cand.excel,
                                                                          cand.grade, cand.socio_t, cand.socio_p,
                                                                          cand.words_n, cand.unique_ratio))

    print(boysd)
    print(boysoffd)
    print(sum(boysd))
    print(girlsd)
    print(girlsoffd)
    print(sum(girlsd))
    print(maxcnt)
    print(gc)
    print(grps_cnt)
    print(grps)
    #meansf = open("males_and_females_on_females_means.csv", "w")
    mstr = ""
    for i in range(2):
        for j in range(15):
            grps[i][j] = grps[i][j]/grps_cnt[i]
            mstr = "{}{},".format(mstr, grps[i][j])
        #meansf.write(mstr + "\n")
        mstr = ""
    #meansf.close()
    print(grps)
    #print("r={} e={}".format(r, e))
    ## PRINTING distributions to test diff between groups
    #print_list_distribution(lists_by_sex_excel[0, 0])
    #print_list_distribution(lists_by_sex_excel[0, 1])
    #print_list_distribution(lists_by_sex_excel[1, 0])
    #print_list_distribution(lists_by_sex_excel[1, 1])

    #print_list_distribution(lists_by_excel[0])
    #print_list_distribution(lists_by_excel[1])

    #print_list_distribution(lists_by_sex[0])
    #print_list_distribution(lists_by_sex[1])

    #print("women non officer")
    #print_list_distribution(lists_by_sex_excel[0, 0])
    #print("men non officer")
    #print_list_distribution(lists_by_sex_excel[1, 0])
    #print("women officer")
    #print_list_distribution(lists_by_sex_excel[0, 1])
    #print("men officer")
    #print_list_distribution(lists_by_sex_excel[1, 1])

    #for disl in lists_by_grades:
        #print_list_distribution(disl)

    f.close()


def run_gensim_engine(text, fname, topics_n):
    model, corp = gensim_lda_run(text, topics_n)
    topicsf = open(fname, "w")
    s = model.print_topics(num_topics=N, num_words=TOPIC_WORDS)
    for topic in s:
        topicsf.write("{}\n".format(topic))
    topicsf.close()
    word2prob_list = create_word2prob_dicts(model)
    return model, corp, word2prob_list


def run_keras_engine(text, fname, topics_n):
    word2prob_list = []

    id2word, corpus = text2corpus(text)
    model = keras_lda_run(corpus, id2word, topics_n)

    topicsf = open(fname, "w")
    topic_word = model.topic_word_  # model.components_ also works

    for i, topic_dist in enumerate(topic_word):
        d = {}
        topic_words = np.array(list(id2word.token2id.keys()))[np.argsort(topic_dist)][:-(TOPIC_WORDS + 1):-1]
        topic_str = '{}: '.format(i)
        for word in topic_words:
            prob = topic_dist[id2word.token2id[word]]
            d[word] = round(prob, 3)
            topic_str = "{}{:.3f}*{}, ".format(topic_str, prob, word)
        word2prob_list.append(d)
        topicsf.write(topic_str + '\n')
    topicsf.close()
    return model, corpus, word2prob_list


def run_lda(engine, text, otype, to_dump, topics_n=N):
    size = len(text)
    print("Valid Candidates ({}): {}".format(NAMES[otype], size))
    if size > 0:
        fname = "_{}_{}_{}_topics.txt".format(engine, NAMES[otype], topics_n)
        if engine == G:
            model, corpus, word2prob_list = run_gensim_engine(text, fname, topics_n)
        else:
            model, corpus, word2prob_list = run_keras_engine(text, fname, topics_n)
        dump_matches(engine, model, corpus, text, word2prob_list, otype, topics_n)
        if to_dump:
            dump_single_run_results(engine, model, corpus, otype, topics_n)
            #run_linear_regression()
        return model
    else:
        return None


def print_characters_count():
    #f = open("characters_count_by_type.csv", "w")
    counters = np.array([0] * 30).reshape(2, 3, 5)
    cnt = 0
    fcnt = 0
    #family = ["אמא", "אמי", "אימי", "אחות", "דודה"]
    family = ["אמי"]
    for c in index2cand.values():
        #cg, ct = get_cand_char_gender(c)
        counters[c.sex, c.charg, c.chart] = counters[c.sex, c.charg, c.chart] + 1
        if c.sex == 1 and c.charg == 0 and c.chart == 1:
            cnt = cnt + 1
            found = False
            for word in family:
                if word in c.hebText:
                    if not found:
                        fcnt = fcnt + 1
                        found = True
                        print(c.hebText)
    print(cnt)
    print(fcnt)

    for i in range(2):
        for j in range(3):
            s = ""
            for z in range(5):
                s = "{}{},".format(s, counters[i, j, z])
            #print(s)
            #f.write(s + '\n')
    #f.close()


def prepare_lr_sample(cand):
    return [cand.sex, cand.chart, cand.probs[0][1], cand.probs[1][1], cand.probs[2][1], cand.probs[3][1],
            cand.probs[4][1], cand.probs[5][1], cand.probs[6][1], cand.probs[7][1], cand.probs[8][1], cand.probs[9][1],
            cand.probs[10][1], cand.probs[11][1], cand.probs[12][1], cand.probs[13][1], cand.probs[14][1]]


def prepare_linear_regression_db(train_quota1, train_quota2, train_quota3, train_quota4, predict_quota1, predict_quota2,
                                 predict_quota3, predict_quota4):
    train_quota = [train_quota1, train_quota2, train_quota3, train_quota4]
    predict_quota = [predict_quota1, predict_quota2, predict_quota3, predict_quota4]
    #train_quota = [3, 3]
    #predict_quota = [1, 1]
    train_cnt = [0, 0, 0, 0]
    predict_cnt = [0, 0, 0, 0]
    train_samples = []
    train_categories = []
    predict_samples = []
    predict_categories = []
    predict_cands = []

    for key in index2cand:
        cand = index2cand[key]

        if cand.charg == NO_GENDER:
            continue

        category = cand.charg
        group = (2 * cand.sex) + cand.charg
        p_group = cand.charg


        if train_cnt[group] < train_quota[group]:
            #print("gender {}, char gender {}, group {}, p_group {}".format(cand.sex, cand.charg, group, p_group))
            #print("adding to train")
            sample = prepare_lr_sample(cand)
            train_samples.append(sample)
            train_categories.append(category)
            train_cnt[group] = train_cnt[group] + 1
        elif predict_cnt[p_group] < predict_quota[p_group]:
            #print("gender {}, char gender {}, group {}, p_group {}".format(cand.sex, cand.charg, group, p_group))
            #print("adding to predict")
            sample = prepare_lr_sample(cand)
            predict_samples.append(sample)
            predict_cands.append(cand)
            predict_categories.append(category)
            predict_cnt[p_group] = predict_cnt[p_group] + 1

    #for i, item in enumerate(predict_cands):
        #print("{},{}".format(predict_categories[i], predict_cands[i].hebText))

    newl = list(zip(predict_samples, predict_categories, predict_cands))
    random.shuffle(newl)

    new_predict_samples, new_predict_categories, new_predict_cands = zip(*newl)

    print(train_cnt)
    print(predict_cnt)
    return train_samples, train_categories, new_predict_samples, new_predict_categories, new_predict_cands


def run_linear_regression(train_quota1, train_quota2, train_quota3, train_quota4, predict_quota1, predict_quota2,
                          predict_quota3, predict_quota4):
    results = [0, 0]
    right = [0, 0]
    train_samples, train_results, predict_samples, predict_results, predict_cands = prepare_linear_regression_db(train_quota1,
                                                                                                  train_quota2,
                                                                                                  train_quota3,
                                                                                                  train_quota4,
                                                                                                  predict_quota1,
                                                                                                  predict_quota2,
                                                                                                  predict_quota3,
                                                                                                  predict_quota4)

    clf.fit(train_samples, train_results)
    for i, sample in enumerate(predict_samples):
        prediction = clf.predict([sample])
        results[prediction[0]] = results[prediction[0]] + 1
        print(predict_cands[i].hebText)
        if prediction[0] == predict_results[i]:
            #print("prediction: {}, result: {} - TRUE".format(prediction[0], predict_results[i]))
            right[prediction[0]] = right[prediction[0]] + 1
        else:
            print("prediction: {}, result: {} - FALSE".format(prediction[0], predict_results[i]))

    print("Train data: {}/{}".format(sum(train_results), len(train_results)))
    print("Predict data: {}/{}".format(sum(predict_results), len(predict_results)))
    print("predictions: [0]={}/{}, [1]={}/{}, rate = {}".format(right[0], results[0], right[1], results[1], sum(right)/len(predict_samples)))


print(datetime.datetime.now())
full_text = get_translated_text("Translated_text.txt")
#full_text = get_translated_text("fake_translated.txt")
text = full_text[:MAX_LINE]
db = get_cands_data('thesis_db.xls', MAX_LINE)
#db = get_cands_data('fake_db.xls', MAX_LINE)
reviewed_cands = []
characters_map = {}
errors = [0, 0, 0, 0, 0, 0]
lda_text = [[], [], [], []]
accum_kkz_text = [""] * 4
accum_grade_text = [""] * 41
entire_text = []
sample1_text = []
sample2_text = []
boys = []
girls = []
# Accumulated train text is gathered by certain criteria (e.g grade) and every index in it is a value that accumulates
# text matching this value, There are no values higher than 100 for any criteria
accum_train_text = [""] * 100
accum_cands = 0
cand_ids = []

high = []
low = []

index = 0

load_characters(characters_map)

for line in text:
    this_cand_id = db.ID_coded[index]
    if this_cand_id not in reviewed_cands:
        reviewed_cands.append(this_cand_id)
        cand = get_cand(db, full_text, characters_map, index, YEARS, errors)
        if cand is not None:
            entire_text.append(line)
            cand_ids.append(cand.id)
            lda_text[cand.otype].append(line)
            index2cand[index] = cand


            #if cand.sex == 0:
            #    girls.append(line)
            #else:
            #    boys.append(line)

            #sample1_index = len(sample1_text)
            #sample2_index = len(sample2_text)
            #if (sample1_index < 100 and cand.otype == 0) or (sample2_index < 100 and cand.otype == 3):
            #if index > 89:
                # Cand goes to applied sample
                #if sample1_index < 100 and cand.otype == 0:
                #sample1_text.append(line)
                #sample1_index2cand[sample1_index] = cand
                #else:
                #    sample2_text.append(line)
                #    sample2_index2cand[sample2_index] = cand

            #else:
            #    # Cand goes to accumulate train text
            #    if cand.grade == 0:
            #        ii = 0
            #    else:
            #        ii = cand.grade - 59
            #    accum_train_text[ii] = accum_train_text[ii] + line
            #    accum_cands = accum_cands + 1


            #if cand.otype == 3 and not cand.grade == "" and int(cand.grade) > 83:
            #entire_text.append(line)

            #if cand.otype == 3 and not cand.grade == "" and int(cand.grade) < 80:
            #   low.append(line)
    index = index + 1

#print_characters_count()
# These two are the actual 'main' for latest run.
lem_text = get_data_lemmatized(entire_text)
run_lda(K, lem_text, 4, True, N)
### THIS ONE GAVE 76.5% run_linear_regression(1000, 1000, 200, 1000, 100, 100, 0, 0)
### THIS ONE GAVE 95% - 900 200 900 900 200 200 0 0
### THIS ONE GAVE 97.5% 960 240 240 960 100 100 0 0

val = input("Enter values for linear regression (train quota 1-4, predict quota 1-4): ")
while not val == "":
    parts = val.split()
    if len(parts) > 7:
        run_linear_regression(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]),
                              int(parts[6]), int(parts[7]))
    else:
        print("Invalid input")
    val = input("Enter values for linear regression (train quota 1-4, predict quota 1-4): ")


#dump_distributions()

#if characters_map[11783427][0] > 0:
#    print("yes 1")
#if characters_map[11783427][1] > 0:
#    print("yes 2")

#print(len(index2cand))
#print(pc)

#for i in range(4):
#    run_lda(lda_text[i], i, True, N)
#for topics in range(16, 26):
#run_gensim(lem_text, 4, True, N)
#run_lda(K, lem_text, 4, True, N)
#lr_by_text()
#run_keras(lem_text, 4, N)
#dump_data_lemmatized(entire_text, cand_ids, "_All_2015_lemmatized.txt")

#gensim_analyze_corpus(entire_text, "_2020_adv_verb_word_count.txt")
#gensim_analyze_corpus(boys, "_boys_word_count.txt")
#gensim_analyze_corpus(girls, "_girls_word_count.txt")


#for i in range(4):
#    fn = "_kkz{}_2020_words_count.txt".format(i)
#    gensim_analyze_corpus(lda_text[i], fn)

#print("Total {}, Train {}, Sample1 {}, Sample2 {}".format(len(entire_text), accum_cands, len(sample1_text), len(sample2_text)))

# Ignore values which did not accumulated any text
#accum_text = []
#for t in accum_train_text:
#    if not len(t) == 0:
#        accum_text.append(t)

#t_model = run_lda(accum_text, 4, False)
#if len(sample1_text) > 0:
#    apply_new_text_on_train_model(t_model, sample1_text, sample1_index2cand, "last30")
#if len(sample2_text) > 0:
#    apply_new_text_on_train_model(t_model, sample2_text, sample2_index2cand, "2")

#run_lda(lda_text[0], 0, 'lda0_backup')
#gensim_apply_text_on_model('lda0_backup', sample_text)

print(datetime.datetime.now())
print("Done")

#load_lda('my_lda_backup')

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
