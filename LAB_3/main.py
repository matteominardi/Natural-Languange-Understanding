from functions import *
from nltk.corpus import gutenberg

macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-macbeth.txt')]
macbeth_words = [word for sent in macbeth_sents for word in sent]

test_sents = [
    "the tragedy of",
    "he was born",
    "the king is dead",
    "when shall we three meet"
]

backoff_order = 2

stupid_backoff, myBackoff = inst_backoffs(macbeth_sents, macbeth_words, backoff_order)

compare_test_sents_scores(stupid_backoff, myBackoff, test_sents)
compare_pp(stupid_backoff, myBackoff, macbeth_sents, backoff_order)