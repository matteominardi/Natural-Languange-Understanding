from functions import *
import nltk
nltk.download('conll2002')
from nltk.corpus import conll2002

if __name__ == "__main__":
    trn_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
    tst_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testa')]

    print("\n\nBaseline performance:")
    trn_feats = [baseline_sent2spacy_features(s) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]
    tst_feats = [baseline_sent2spacy_features(s) for s in tst_sents]
    train_test(trn_feats, trn_label, tst_feats, tst_sents)
    print("----------------------------------------")

    print("\n\nAdding suffix performance:")
    trn_feats = [suffix_sent2spacy_features(s) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]
    tst_feats = [suffix_sent2spacy_features(s) for s in tst_sents]
    train_test(trn_feats, trn_label, tst_feats, tst_sents)
    print("----------------------------------------")

    print("\n\nAdding CoNLL features performance:")
    trn_feats = [conll_sent2spacy_features(s) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]
    tst_feats = [conll_sent2spacy_features(s) for s in tst_sents]
    train_test(trn_feats, trn_label, tst_feats, tst_sents)
    print("----------------------------------------")

    print("\n\n[-1,+1] Windowed CoNLL performance:")
    trn_feats = [window1_conll_sent2spacy_features(s) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]
    tst_feats = [window1_conll_sent2spacy_features(s) for s in tst_sents]
    train_test(trn_feats, trn_label, tst_feats, tst_sents)
    print("----------------------------------------")

    print("\n\n[-2,+2] Windowed CoNLL performance:")
    trn_feats = [window2_conll_sent2spacy_features(s) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]
    tst_feats = [window2_conll_sent2spacy_features(s) for s in tst_sents]
    train_test(trn_feats, trn_label, tst_feats, tst_sents)