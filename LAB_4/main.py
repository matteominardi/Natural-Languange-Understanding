from functions import *
import nltk
import spacy
from spacy.tokens import Doc
from nltk.metrics import accuracy

print("Splitting data for train (+ validation) and test")
trn_data, val_data, tst_data = split_data()

print("\n\n(NLTK) Experimenting with different tagger parameters")
experiment("BASELINE", trn_data, val_data, tst_data, 1)
experiment("HIGHER CUTOFF", trn_data, val_data, tst_data, 2)
experiment("BACKOFF WITH NO DEFAULT TAGGER", trn_data, val_data, tst_data, 1, True)
experiment("BACKOFF WITH DEFAULT TAGGER", trn_data, val_data, tst_data, 1, True, nltk.DefaultTagger('NOUN'))

print("------------------------------------------------------------------")
print("------------------------------------------------------------------")

print("\n\n(Spacy) Experimenting with Spacy tagger")
tst_data_words = extract_words_only(tst_data)

nlp = spacy.load("en_core_web_sm")
doc = Doc(nlp.vocab, words=tst_data_words)
doc = nlp(doc)

spacy_res = extract_token_text_pos_pair(doc)

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

mapped_spacy_res = map_spacy_to_NLTK(spacy_res, mapping_spacy_to_NLTK)

ref_tst_data = [tup for l in tst_data for tup in l]
print("------------------------------------------------------------------")
print("Spacy Final Test Accuracy:", accuracy(reference = ref_tst_data, test = mapped_spacy_res))