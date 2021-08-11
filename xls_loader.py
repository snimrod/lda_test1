import pandas as pd
import codecs
from googletrans import Translator

def export_txt():
    f = open("txt_output.txt", "w")


# df = pd.read_excel('Book1.xlsx', sheetname=None, header=None)
# df = pd.read_excel('Book1.xlsx')

# Read Excel and select a single cell (and make it a header for a column)
# data = pd.read_excel('Book11.xls', 'Sheet1', index_col=None, usecols="C", header=0, nrows=0)
data = pd.read_excel('thesis_db.xls', 'Original', index_col=None, usecols=None, header=0, nrows=12110)

f = open("Translated_text.txt", errors='ignore')
lines = f.readlines()

#for line in lines:
#    print(line)

f.close()

#print(len(data.values[0,8]))
#print(len(lines[0]))
#print(len(lines[0])/len(data.values[0,8]))

count1 = 0
count2 = 0
count3 = 0
count4 = 0

for i in range(0, 12109):
    hebText = data.values[i, 8]
    engText = lines[i]
    if not(pd.isnull(hebText)) and not(isinstance(hebText, (int, float))):
        ratio = len(engText)/len(hebText)
        # print("Checking line {} ({})".format(i, engText[0:20]))
        if ratio < 1:
            print("{})ratio is: {} ({})".format(i, ratio, engText[0:20]))
            count1 = count1 + 1
            print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))
        if (ratio >= 1) and (ratio < 1.1):
            print("{})ratio is: {} ({})".format(i, ratio, engText[0:20]))
            count2 = count2 + 1
            print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))
        if (ratio >= 1.1) and (ratio < 1.2):
            print("{})ratio is: {} ({})".format(i, ratio, engText[0:20]))
            count3 = count3 + 1
            print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))
        if (ratio >= 1.2) and (ratio < 1.3):
            print("{})ratio is: {} ({})".format(i, ratio, engText[0:20]))
            count4 = count4 + 1
            print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))
#f = codecs.open("txt_output.txt", "w", "utf-8")

# f = open("txt_output.txt", "w")
print("Counters: {}, {}, {}, {}".format(count1, count2, count3, count4))

#translator = Translator()



#print(data.values[1,8])
#result = translator.translate(data.values[1,8], src='he', dest='en')
#print(result.text)



#for c in data.דמות:
#    # print(i)
#    result = translator.translate(c)
#    f.write("{}\n".format(result.text))
#    i = i + 1

#f.close()
print("Done")