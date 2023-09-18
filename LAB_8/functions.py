import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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