import numpy as np
import lda
import lda.datasets

ITERATIONS = 100


def corpus2nparray(corpus, id2word):
    nparr = np.array([0] * id2word.num_docs * len(id2word))
    nparr = np.reshape(nparr, [id2word.num_docs, len(id2word)])
    doc_i = 0
    for doc in corpus:
        for t in doc:
            nparr[doc_i, t[0]] = t[1]
        doc_i = doc_i + 1
    return nparr


def keras_lda_run(corpus, id2word, topics_n):

    arr = corpus2nparray(corpus, id2word)

    model = lda.LDA(n_topics=topics_n, n_iter=ITERATIONS, random_state=1)
    model.fit(arr)  # model.fit_transform(X) is also available

    return model

