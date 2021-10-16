from xls_loader import get_translated_text

# Run in python console
import nltk;

# nltk.download('stopwords')
from nltk.corpus import stopwords

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy


#my_stop_words = ['character', 'person', 'people', 'always', 'know', 'choose', 'good', 'positive', 'think', 'man',
#                 'also', 'give', 'life', 'way', 'believe', 'quality', 'thing', 'even', 'opinion', 'trait', 'help',
#                 'see', 'other', 'love', 'work', 'care', 'make', 'go', 'lot', 'take', 'lead', 'important']
my_stop_words = ['character', 'person', 'people']

def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    # NLTK Stop words
    stop_words = stopwords.words('english')
    stop_words.extend(my_stop_words)
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=3, threshold=20)  # higher threshold fewer phrases.

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=3, threshold=20)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=20)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def rawtext2corpus(text):
    data_words = list(sent_to_words(text))

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['ADJ', 'VERB'])

    for i in range(len(data_lemmatized)):
        for word in my_stop_words:
            while word in data_lemmatized[i]:
                data_lemmatized[i].remove(word)

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return id2word, corpus


def text2corpus(texts):
    # Create Dictionary
    id2word = corpora.Dictionary(texts)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return id2word, corpus


def gensim_lda_run(text, topics_n, backup_name=''):
    id2word, corpus = text2corpus(text)
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=topics_n,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                # alpha='auto',
                                                # alpha='symmetric',
                                                # alpha='asymmetric',
                                                per_word_topics=True)

    if len(backup_name) > 0:
        lda_model.save(backup_name)

    outFile = open("temp_results2.txt", "w")
    for i in range(len(text)):
        str = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[i:i + 1]]
        for s in str[0]:
            outFile.write("({}, {}),".format(s[0], s[1]))
        outFile.write("\n")
        str = lda_model.get_document_topics(corpus[i], minimum_probability=0.01)
        for s in str:
            outFile.write("({}, {:.5f}),".format(s[0], s[1]))
        outFile.write("\n")
    outFile.close()

    return lda_model, corpus


def gensim_load_model(fname):
    return gensim.models.ldamodel.LdaModel.load(fname)


def gensim_apply_text_on_model(model, text):
    id2word, corpus = text2corpus(text)
    #model = gensim.models.ldamodel.LdaModel.load(model_name)

    vectors = []

    for doc in corpus:
        vector = model[doc]
        vectors.append(vector[0])
    return vectors


def gensim_apply_text_on_model2(model, text):
    my_text = [text]
    id2word, corpus = text2corpus(my_text)
    #model = gensim.models.ldamodel.LdaModel.load(model_name)

    x = model.get_document_topics(corpus)
    f = 5



def gensim_analyze_corpus(text, fname):
    data_words = list(sent_to_words(text))

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['ADJ', 'VERB'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    all = {}
    for item in corpus:
        for t in item:
            if t[0] in all:
                all[t[0]] = all[t[0]] + t[1]
            else:
                all[t[0]] = t[1]

    all_list = [(k, v) for k, v in all.items()]
    all_list.sort(key=lambda x: x[1])
    all_list.reverse()

    f = open(fname, "w")
    size = len(text)
    f.write("Size: {}\n".format(size))
    for item in all_list:
        f.write("{}) {} ({:.3f})\n".format(id2word[item[0]], item[1], item[1] / size))
    f.close()

    return all, all_list, id2word


def dump_data_lemmatized(text, ids, fname):
    data_words = list(sent_to_words(text))

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    f = open(fname, "w")
    i = 0
    for d in data_lemmatized:
        f.write("{} - {}\n".format(ids[i], d))
        i = i + 1
    f.close()


def get_data_lemmatized(text):
    data_words = list(sent_to_words(text))

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    for i in range(len(data_lemmatized)):
        for word in my_stop_words:
            while word in data_lemmatized[i]:
                data_lemmatized[i].remove(word)

    return data_lemmatized
