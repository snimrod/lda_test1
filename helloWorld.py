import csv
import numpy as np


def digits(str):

    for i in str:
        if i.isdigit():
            return True

    return False


def is_valid(word):
    non_interesting = ['with', 'coverity']

    if len(word) < 4:
        return False

    if word[0] == '@':
        return False

    if word in non_interesting:
        return False

    if digits(word):
        return False

    return True

def get_users_and_words(f):
    users = []
    words = []
    with open(f, newline='') as csvfile:

        cnt = 0
        fieldnames = ['line', 'user', 'body']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)

        for row in reader:
            user = row['user']
            body = row['body'].lower()

            if user not in users:
                users.append(user)

            wlist = body.split()
            for word in wlist:
                if is_valid(word):
                    cnt = cnt + 1
                if is_valid(word) and word not in words:
                    words.append(word)

    print(cnt)
    return users, words


def get_np_array(f, users, words):
    nparr = np.array([0] * len(users) * len(words))
    nparr = np.reshape(nparr, [len(users), len(words)])

    with open(f, newline='') as csvfile:

        fieldnames = ['line', 'user', 'body']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)

        for row in reader:
            user = row['user']
            body = row['body'].lower()

            wlist = body.split()
            for word in wlist:
                if is_valid(word):
                    nparr[users.index(user), words.index(word)] = nparr[users.index(user), words.index(word)] + 1

    return nparr



#a = [0] * 250

#b = [1] * 250

#y = np.array(a)

#b[50] = 8

#y = np.append(y, b)
#y = np.reshape(y, [2, 250])

# print(y.size)


#usersList, wordsList = get_users_and_words("Mellanox_nvmx.csv")
#x = get_np_array("Mellanox_nvmx.csv", usersList, wordsList)
#b=2
#analyze_csv("text1.txt")
