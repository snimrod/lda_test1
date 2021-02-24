import numpy as np
import lda
import lda.datasets

from helloWorld import get_users_and_words, get_np_array


def print_sorted(name, values):
    print("{} : {%.5f}".format(name, values))

#usersList, wordsList = get_users_and_words("Mellanox_nvmx.csv")
titles, vocab = get_users_and_words("Mellanox_nvmx.csv")
X = get_np_array("Mellanox_nvmx.csv", titles, vocab)

#X = lda.datasets.load_reuters()
#vocab = lda.datasets.load_reuters_vocab()
#titles = lda.datasets.load_reuters_titles()

model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_

for i in range(len(titles)):
    print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
    print_sorted(titles[i], doc_topic[i])

c = 1

