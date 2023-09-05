from functions import *
import math
import nltk
from nltk.corpus import treebank
from nltk.tag import NgramTagger
nltk.download('treebank')
import spacy
from spacy.tokens import Doc
from nltk.metrics import accuracy

print("Splitting data for train (+val) and test")
print('Dataset total size: ', len(treebank.tagged_sents()))
trn_data = treebank.tagged_sents(tagset='universal')[:math.ceil(len(treebank.tagged_sents()) * 0.8)]
tst_data = treebank.tagged_sents(tagset='universal')[math.ceil(len(treebank.tagged_sents()) * 0.8):]
print("Train length and test length", len(trn_data), len(tst_data))
val_data = trn_data[-len(trn_data)//10:]
trn_data = trn_data[:-len(trn_data)//10]
print("Train length, val length, test length", len(trn_data), len(val_data), len(tst_data))

print("\n\nExperimenting with different tagger parameters")
unigram_tagger = NgramTagger(1, train = trn_data, cutoff = 1)
bigram_tagger = NgramTagger(2, train = trn_data, cutoff = 1)
trigram_tagger = NgramTagger(3, train = trn_data, cutoff = 1)
print('BASELINE')
print('Accuracy: ', unigram_tagger.accuracy(val_data), bigram_tagger.accuracy(val_data), trigram_tagger.accuracy(val_data))
print('------------------------------------------------------------------')

unigram_tagger = NgramTagger(1, train = trn_data, cutoff = 2)
bigram_tagger = NgramTagger(2, train = trn_data, cutoff = 2)
trigram_tagger = NgramTagger(3, train = trn_data, cutoff = 2)
print('HIGHER CUTOFF')
print('Accuracy: ', unigram_tagger.accuracy(val_data), bigram_tagger.accuracy(val_data), trigram_tagger.accuracy(val_data))
print('------------------------------------------------------------------')

unigram_tagger = NgramTagger(1, train = trn_data, cutoff = 1)
bigram_tagger = NgramTagger(2, train = trn_data, cutoff = 1, backoff = unigram_tagger)
trigram_tagger = NgramTagger(3, train = trn_data, cutoff = 1, backoff = bigram_tagger)
print('BACKOFF WITH NO DEFAULT TAGGER')
print('Accuracy: ', unigram_tagger.accuracy(val_data), bigram_tagger.accuracy(val_data), trigram_tagger.accuracy(val_data))
print('------------------------------------------------------------------')

default_tagger = nltk.DefaultTagger('NOUN')
unigram_tagger = NgramTagger(1, train = trn_data, cutoff = 1, backoff = default_tagger)
bigram_tagger = NgramTagger(2, train = trn_data, cutoff = 1, backoff = unigram_tagger)
trigram_tagger = NgramTagger(3, train = trn_data, cutoff = 1, backoff = bigram_tagger)
print('FINAL VALIDATION Accuracy: ', trigram_tagger.accuracy(val_data))
print('------------------------------------------------------------------')

print('TEST Accuracy: ', trigram_tagger.accuracy(tst_data))
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')
nlp = spacy.load('en_core_web_sm')
tst_data_words = []

for l in tst_data:
    for pair in l:
        tst_data_words.append(pair[0])

print(len(tst_data_words), tst_data_words)

tmp = Doc(nlp.vocab, words=tst_data_words)
doc = nlp(tmp)

res = []
for token in doc:
    res.append((token.text, token.pos_))
print(len(res), res)

mapping_spacy_to_NLTK = {
    "ADJ": "ADJ",
    "ADP": "ADP",
    "ADV": "ADV",
    "AUX": "VERB",
    "CCONJ": "CONJ",
    "DET": "DET",
    "INTJ": "X",
    "NOUN": "NOUN",
    "NUM": "NUM",
    "PART": "PRT",
    "PRON": "PRON",
    "PROPN": "NOUN",
    "PUNCT": ".",
    "SCONJ": "CONJ",
    "SYM": "X",
    "VERB": "VERB",
    "X": "X"
}

mapped_res = []
for tup in res:
    mapped_res.append((tup[0], mapping_spacy_to_NLTK[tup[1]]))
print('MAPPED spacy output to NLTK format')
print(mapped_res)
print('------------------------------------------------------------------')

ref_tst_data = [tup for l in tst_data for tup in l]
print('Reference test data')
print(ref_tst_data)
print('------------------------------------------------------------------')

print('Lengths: ', len(ref_tst_data), len(mapped_res))
print('Accuracy: ', accuracy(reference = ref_tst_data, test = mapped_res))