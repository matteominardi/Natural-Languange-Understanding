from functions import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from nltk.metrics.scores import accuracy
import numpy as np
import nltk
nltk.download('senseval')
from nltk.corpus import senseval
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
nltk.download('wordnet_ic')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

data = [" ".join([t[0] for t in inst.context]) for inst in senseval.instances('interest.pos')]
lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]

vectorizer = CountVectorizer()
classifier = MultinomialNB()
lblencoder = LabelEncoder()

stratified_split = StratifiedKFold(n_splits=5, shuffle=True)

vectors = vectorizer.fit_transform(data)

lblencoder.fit(lbls)
labels = lblencoder.transform(lbls)

scores = cross_validate(classifier, vectors, labels, cv=stratified_split, scoring=['f1_micro'])

print('Using BOW:', sum(scores['test_f1_micro'])/len(scores['test_f1_micro']))

data_col = [collocational_features(inst) for inst in senseval.instances('interest.pos')]

dvectorizer = DictVectorizer(sparse=False)
dvectors = dvectorizer.fit_transform(data_col)

scores = cross_validate(classifier, dvectors, labels, cv=stratified_split, scoring=['f1_micro'])

print('Using collocational features:', sum(scores['test_f1_micro'])/len(scores['test_f1_micro']))

uvectors = np.concatenate((vectors.toarray(), dvectors), axis=1)

scores = cross_validate(classifier, uvectors, labels, cv=stratified_split, scoring=['f1_micro'])
print('Concatenating feature vectors:', sum(scores['test_f1_micro'])/len(scores['test_f1_micro']))

mapping = {
    'interest_1': 'interest.n.01',
    'interest_2': 'interest.n.03',
    'interest_3': 'pastime.n.01',
    'interest_4': 'sake.n.01',
    'interest_5': 'interest.n.05',
    'interest_6': 'interest.n.04',
}

refs = {k: set() for k in mapping.values()}
hyps1 = {k: set() for k in mapping.values()}
hyps2 = {k: set() for k in mapping.values()}
refs_list = []
hyps1_list, hyps2_list = [], []

synsets = []
for ss in wordnet.synsets('interest', pos='n'):
    if ss.name() in mapping.values():
        defn = ss.definition() 
        tags = preprocess(defn) # Preprocess the definition
        toks = [l for w, l, p in tags] # Get the tokens
        synsets.append((ss,toks))

for i, inst in enumerate(senseval.instances('interest.pos')):
    txt = [t[0] for t in inst.context]
    raw_ref = inst.senses[0] # let's get first sense
    hyp1 = original_lesk(txt, txt[inst.position], synsets=synsets).name()
    hyp2 = lesk_similarity(txt, txt[inst.position], synsets=synsets).name()
    
    ref = mapping.get(raw_ref)
           
    refs[ref].add(i)
    hyps1[hyp1].add(i)
    hyps2[hyp2].add(i)
    
    refs_list.append(ref)
    hyps1_list.append(hyp1)
    hyps2_list.append(hyp2)

print("----------------------------------------------------------------")
print("----------------------------------------------------------------")
print("Acc original_lesk:", round(accuracy(refs_list, hyps1_list), 3))
print("Macro Precision: %.3f" % precision_score(refs_list, hyps1_list, average="macro"))
print("Macro Recall: %.3f" % recall_score(refs_list, hyps1_list, average="macro"))
print("Macro F1: %.3f" % f1_score(refs_list, hyps1_list, average="macro"))
print("Weighted Precision: %.3f" % precision_score(refs_list, hyps1_list, average="weighted"))
print("Weighted Recall: %.3f" % recall_score(refs_list, hyps1_list, average="weighted"))
print("Weighted F1: %.3f" % f1_score(refs_list, hyps1_list, average="weighted"))
print("----------------------------------------------------------------")
print("----------------------------------------------------------------")
print("Acc lesk_similarity:", round(accuracy(refs_list, hyps2_list), 3))
print("Macro Precision: %.3f" % precision_score(refs_list, hyps2_list, average="macro"))
print("Macro Recall: %.3f" % recall_score(refs_list, hyps2_list, average="macro"))
print("Macro F1: %.3f" % f1_score(refs_list, hyps2_list, average="macro"))
print("Weighted Precision: %.3f" % precision_score(refs_list, hyps2_list, average="weighted"))
print("Weighted Recall: %.3f" % recall_score(refs_list, hyps2_list, average="weighted"))
print("Weighted F1: %.3f" % f1_score(refs_list, hyps2_list, average="weighted"))