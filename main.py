import numpy as np
import lda
import lda.datasets

N = 10

from helloWorld import get_users_and_words, get_np_array, pp_file


def print_sorted(name, values):
    topics = range(10)
    tarr = list(zip(topics, values))
    s_topics = sorted(tarr, key=lambda x: x[-1], reverse=True)
    print("{} - {}".format(name, s_topics))



#fname = pp_file("Mellanox_nvmx.csv")
fname = "input.txt"
titles, vocab = get_users_and_words(fname)
X = get_np_array(fname, titles, vocab)

#X = lda.datasets.load_reuters()
#vocab = lda.datasets.load_reuters_vocab()
#titles = lda.datasets.load_reuters_titles()



model = lda.LDA(n_topics=N, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_

maxTopics = [0] * N

for i in range(len(titles)):
    t = doc_topic[i].argmax()
    maxTopics[t] = maxTopics[t] + 1
    if t == 6:
        print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
    #print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
        print_sorted(titles[i], doc_topic[i])

for i in range(N-1):
    print("Topic {}: {}".format(i, maxTopics[i]))




