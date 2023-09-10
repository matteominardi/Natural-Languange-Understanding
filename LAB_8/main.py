from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from nltk.metrics.scores import accuracy
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('senseval')
from nltk.corpus import senseval

nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
nltk.download('wordnet_ic')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

def collocational_features(inst):
    p = inst.position
    return {
        "w-3_word": 'NULL' if p < 3 else inst.context[p-3][0],
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0],
        "w+3_word": 'NULL' if len(inst.context) - 1 < p+3 else inst.context[p+3][0],
        "w_word": inst.context[p][0],
        "w_pos": inst.context[p][1],
        "w-3_pos": 'NULL' if p < 3 else inst.context[p-3][1],
        "w-2_pos": 'NULL' if p < 2 else inst.context[p-2][1],
        "w-1_pos": 'NULL' if p < 1 else inst.context[p-1][1],
        "w+1_pos": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][1],
        "w+2_pos": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][1],
        "w+3_pos": 'NULL' if len(inst.context) - 1 < p+3 else inst.context[p+3][1],
        "ngram-3": 'NULL' if p < 2 else inst.context[p-2][0] + inst.context[p-1][0] + inst.context[p][0],
        "ngram-2": 'NULL' if p < 1 else inst.context[p-1][0] + inst.context[p][0],
        "ngram+2": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p][0] + inst.context[p+1][0],
        "ngram+3": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p][0] + inst.context[p+1][0] + inst.context[p+2][0]
    }

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


def preprocess(text):
    mapping = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}
    sw_list = stopwords.words('english')
    
    lem = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text) if type(text) is str else text
    tagged = nltk.pos_tag(tokens, tagset="universal")
    tagged = [(w.lower(), p) for w, p in tagged]
    tagged = [(w, p) for w, p in tagged if p in mapping]
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    tagged = list(set(tagged))
    
    return tagged

def get_sense_definitions(context):
    lemma_tags = preprocess(context)

    senses = [(w, wordnet.synsets(l, p)) for w, l, p in lemma_tags]

    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            def_list = []
            for s in sense_list:
                defn = s.definition()
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    return definitions

def get_top_sense(words, sense_list):
    val, sense = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    return val, sense

def original_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, majority=False):
    context_senses = get_sense_definitions(set(context_sentence)-set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    scores = []

    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense(sense[1], synsets))

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    return best_sense

def get_top_sense_sim(context_sense, sense_list, similarity):
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append(context_sense.path_similarity(ss), ss)
            except:
                scores.append((0, ss))    
        elif similarity == "lch":
            try:
                scores.append(context_sense.lch_similarity(ss), ss)
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append(context_sense.wup_similarity(ss), ss)
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append(context_sense.res_similarity(ss, semcor_ic), ss)
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append(context_sense.lin_similarity(ss, semcor_ic), ss)
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append(context_sense.jcn_similarity(ss, semcor_ic), ss)
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    val, sense = max(scores)
    return val, sense

def lesk_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None, synsets=None, majority=False):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    scores = []
    
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))
                    
    if len(scores) == 0:
        return synsets[0][0]
                    
    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    
    return best_sense

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

from sklearn.metrics import precision_score, recall_score, f1_score

print("Acc original_lesk:", round(accuracy(refs_list, hyps1_list), 3))
print('Precision: %.3f' % precision_score(refs_list, hyps1_list, average='macro'))
print('Recall: %.3f' % recall_score(refs_list, hyps1_list, average='macro'))
print('F1: %.3f' % f1_score(refs_list, hyps1_list, average='macro'))
print('Precision: %.3f' % precision_score(refs_list, hyps1_list, average='weighted'))
print('Recall: %.3f' % recall_score(refs_list, hyps1_list, average='weighted'))
print('F1: %.3f' % f1_score(refs_list, hyps1_list, average='weighted'))
print("----------------------------------------------------------------")
print("Acc lesk_similarity:", round(accuracy(refs_list, hyps2_list), 3))
print('Precision: %.3f' % precision_score(refs_list, hyps2_list, average='macro'))
print('Recall: %.3f' % recall_score(refs_list, hyps2_list, average='macro'))
print('F1: %.3f' % f1_score(refs_list, hyps2_list, average='macro'))
print('Precision: %.3f' % precision_score(refs_list, hyps1_list, average='weighted'))
print('Recall: %.3f' % recall_score(refs_list, hyps1_list, average='weighted'))
print('F1: %.3f' % f1_score(refs_list, hyps1_list, average='weighted'))