import nltk
import math
from nltk.tag import NgramTagger
from nltk.corpus import treebank
nltk.download("treebank")

def split_data():
    print("Dataset total size:", len(treebank.tagged_sents()))
    trn_data = treebank.tagged_sents(tagset="universal")[:math.ceil(len(treebank.tagged_sents()) * 0.8)]
    tst_data = treebank.tagged_sents(tagset="universal")[math.ceil(len(treebank.tagged_sents()) * 0.8):]
    print("Train length and test length", len(trn_data), len(tst_data))
    val_data = trn_data[-len(trn_data)//10:]
    trn_data = trn_data[:-len(trn_data)//10]
    print("Train length, val length, test length", len(trn_data), len(val_data), len(tst_data))

    return trn_data, val_data, tst_data


def experiment(experiment_id, train_data, val_data, tst_data, cutoff, backoff = False, default_tagger = None):
    if not backoff:
        unigram_tagger = NgramTagger(1, train = train_data, cutoff = cutoff)
        bigram_tagger = NgramTagger(2, train = train_data, cutoff = cutoff)
        trigram_tagger = NgramTagger(3, train = train_data, cutoff = cutoff)
    else:
        unigram_tagger = NgramTagger(1, train = train_data, cutoff = cutoff, backoff = default_tagger)
        bigram_tagger = NgramTagger(2, train = train_data, cutoff = cutoff, backoff = unigram_tagger)
        trigram_tagger = NgramTagger(3, train = train_data, cutoff = cutoff, backoff = bigram_tagger)

    print(experiment_id)
    print("Accuracy: ")

    if default_tagger is None:
        print("Unigram tagger:", unigram_tagger.accuracy(val_data))
        print("Bigram tagger:", bigram_tagger.accuracy(val_data))
    
    print("Trigram tagger:", trigram_tagger.accuracy(val_data))

    if default_tagger is not None:
        print("NLTK Final Test Accuracy:", trigram_tagger.accuracy(tst_data))


def extract_words_only(tst_data):
    res = []
    for l in tst_data:
        for pair in l:
            res.append(pair[0])
    return res


def extract_token_text_pos_pair(doc):
    res = []
    for token in doc:
        res.append((token.text, token.pos_))
    return res


def map_spacy_to_NLTK(spacy_res, dict):
    mapped_spacy_res = []
    for tup in spacy_res:
        mapped_spacy_res.append((tup[0], dict[tup[1]]))
    print("MAPPED spacy output to NLTK format")
    return mapped_spacy_res