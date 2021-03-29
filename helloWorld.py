import csv
import numpy as np
import re

bots = ["swx-jenkins3", "swx-jenkins2", "MrBr-github"]

def digits(str):

    for i in str:
        if i.isdigit():
            return True

    return False


def is_valid(word):
    non_interesting = ['with', 'coverity', 'bot:retest']

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

            if user in bots:
                continue

            if user not in users:
                users.append(user)

            wlist = body.split()
            for word in wlist:
                if "from:" in word:
                    break

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

            if user in bots:
                continue

            #if "From" in :
            #    yy = 9

            wlist = body.split()
            for word in wlist:
                if "from:" in word:
                    break

                if is_valid(word):
                    nparr[users.index(user), words.index(word)] = nparr[users.index(user), words.index(word)] + 1

    return nparr


def pp_file(f):
    newName = "PP_" + f
    newF = open(newName, "w")

    with open(f, newline='') as csvfile:

        fieldnames = ['line', 'user', 'body']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)

        for row in reader:
            line = row['line']
            user = row['user']
            body = row['body'].lower()

            newF.write("{},{},".format(line, user))
            wlist = body.split()
            for word in wlist:
                newWord = re.sub('[^a-zA-Z]+', '', word)
                if len(newWord) > 0:
                    space = " "
                else:
                    space = ""

                if "?" in word:
                    q = "???? "
                else:
                    q = ""

                if "!" in word:
                    ex = "!!!! "
                else:
                    ex = ""

                newF.write("{}{}{}{}".format(newWord, space, q, ex))
                #if q:
                #    newF.write("{} ? ".format(newWord))
                #else:
                #    newF.write("{} ".format(newWord))

            newF.write("\n")

    newF.close()
    return newName


#usersList, wordsList = get_users_and_words("Mellanox_nvmx.csv")
#x = get_np_array("Mellanox_nvmx.csv", usersList, wordsList)

#print(pp_file("example.csv"))
