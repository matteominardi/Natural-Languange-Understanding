import spacy
from spacy.tokenizer import Tokenizer
from sklearn_crfsuite import CRF
import pandas as pd
import es_core_news_sm
nlp = es_core_news_sm.load()
nlp.tokenizer = Tokenizer(nlp.vocab) 

def word2features(sent, i):
    word = sent[i][0]
    return {'bias': 1.0, 'word.lower()': word.lower()}

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2tokens(sent):
    return [token for token, label in sent]

def sent2labels(sent):
    return [label for token, label in sent]

def baseline_sent2spacy_features(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)
    
    return feats

def suffix_sent2spacy_features(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'suffix': token.suffix_,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)
    
    return feats

def conll_sent2spacy_features(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.text[-3:],
            'word[-2:]': token.text[-2:],
            'word.isupper()': token.is_upper,
            'word.istitle()': token.is_title,
            'word.isdigit()': token.is_digit,
            'pos': token.pos_,
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
            'suffix': token.suffix_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)
    
    return feats

def window1_conll_sent2spacy_features(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for j in range(len(spacy_sent)):
        token = spacy_sent[j]
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.text[-3:],
            'word[-2:]': token.text[-2:],
            'word.isupper()': token.is_upper,
            'word.istitle()': token.is_title,
            'word.isdigit()': token.is_digit,
            'pos': token.pos_,
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
            'suffix': token.suffix_,
            'lemma': token.lemma_ 
        }
        if j > 0:
            word1 = spacy_sent[j-1]
            postag1 = word1.pos_
            token_feats.update({
                '-1:word.lower()': word1.lower_,
                '-1:word.istitle()': word1.is_title,
                '-1:word.isupper()': word1.is_upper,
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            token_feats['BOS'] = True
            
        if j < len(spacy_sent)-1:
            word1 = spacy_sent[j+1]
            postag1 = word1.pos_
            token_feats.update({
                '+1:word.lower()': word1.lower_,
                '+1:word.istitle()': word1.is_title,
                '+1:word.isupper()': word1.is_upper,
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            token_feats['EOS'] = True
        feats.append(token_feats)
    
    return feats

def window2_conll_sent2spacy_features(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for j in range(len(spacy_sent)):
        token = spacy_sent[j]
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.text[-3:],
            'word[-2:]': token.text[-2:],
            'word.isupper()': token.is_upper,
            'word.istitle()': token.is_title,
            'word.isdigit()': token.is_digit,
            'pos': token.pos_,
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
            'suffix': token.suffix_,
            'lemma': token.lemma_ , 
        }
        if j > 1:
            word1 = spacy_sent[j-1]
            postag1 = word1.pos_
            word2 = spacy_sent[j-2]
            postag2 = word2.pos_
            token_feats.update({
                '-1:word.lower()': word1.lower_,
                '-1:word.istitle()': word1.is_title,
                '-1:word.isupper()': word1.is_upper,
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
                '-2:word.lower()': word2.lower_,
                '-2:word.istitle()': word2.is_title,
                '-2:word.isupper()': word2.is_upper,
                '-2:postag': postag2,
                '-2:postag[:2]': postag2[:2],
            })
        else:
            token_feats['BOS'] = True
            
        if j < len(spacy_sent)-2:
            word1 = spacy_sent[j+1]
            postag1 = word1.pos_
            word2 = spacy_sent[j-2]
            postag2 = word2.pos_
            token_feats.update({
                '+1:word.lower()': word1.lower_,
                '+1:word.istitle()': word1.is_title,
                '+1:word.isupper()': word1.is_upper,
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
                '+2:word.lower()': word2.lower_,
                '+2:word.istitle()': word2.is_title,
                '+2:word.isupper()': word2.is_upper,
                '+2:postag': postag2,
                '+2:postag[:2]': postag2[:2],
            })
        else:
            token_feats['EOS'] = True
        feats.append(token_feats)
    
    return feats

def train_test(trn_feats, trn_label, tst_feats, tst_sents):
    crf = CRF(
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=True
    )

    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    pred = crf.predict(tst_feats)

    hyp = [[(tst_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]
    from conll import evaluate
    results = evaluate(tst_sents, hyp)

    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    print(pd_tbl.round(decimals=3))