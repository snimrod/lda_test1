from xls_loader import get_cands_data
from xls_loader import get_translated_text
from TopicModeling_IDF import valid_index
#from xls_loader import is_valid_text
#from xls_loader import is_empty_text
#from OType import OType
#from OType import get_officer_type
#from OType import NONE
#from DataSet import DataSet
from CandData import CandData
from gensim_lda import gensim_lda_run
from pprint import pprint

MAX_LINE = 750
N = 15
YEAR = 2015


def dump_results(model, corpus, index_to_cand):
    #l = [model.get_document_topics(item, minimum_probability=0.01) for item in corpus]
    print(model.get_document_topics(corp[0], minimum_probability=0.01))


full_text = get_translated_text("Translated_text.txt")
text = full_text[:MAX_LINE]
db = get_cands_data('thesis_db.xls', MAX_LINE)
lda_text = []
reviewed_cands = []
index_to_cand = {}
errors = [0, 0, 0, 0, 0, 0]

index = 0
for line in text:
    this_cand_id = db.ID_coded[index]
    if this_cand_id not in reviewed_cands:
        reviewed_cands.append(this_cand_id)
        if valid_index(db, full_text, index, YEAR, errors):
            lda_text.append(line)
            cand = CandData(db, index)
            index_to_cand[index] = cand
    index = index + 1

#print("Rejected:\nDifferent Year: {}\n Invalid Text: {}\n Invalid Ratio: {}\nInvalid Officer: {}\nInvalid Sadir: {}\n\n".format(errors[Y]))
print("Valid Candidates: {}\n\n".format(len(lda_text)))

if len(lda_text) > 0:
    model, corp = gensim_lda_run(text, N)
    pprint(model.print_topics())
    dump_results(model, corp, index_to_cand)

print("Done")
# print(model.get_document_topics(corp[6], minimum_probability=0.01))