import math
import numpy as np
from nltk.lm import MLE
from itertools import chain
from nltk.lm import StupidBackoff
from nltk.lm import NgramCounter
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline


def inst_backoffs(macbeth_sents, macbeth_words, backoff_order = 2):
    vocab = Vocabulary(macbeth_words, unk_cutoff=1)

    macbeth_ngrams_oov, flat_text_oov = padded_everygram_pipeline(2, macbeth_sents)
    counter = NgramCounter(macbeth_ngrams_oov)

    stupid_backoff = StupidBackoff(alpha = 0.4, vocabulary = vocab, counter = counter, order = backoff_order)

    macbeth_ngrams_oov, flat_text_oov = padded_everygram_pipeline(2, macbeth_sents)
    stupid_backoff.fit(macbeth_ngrams_oov, flat_text_oov)

    myBackoff = MyBackoff(alpha = 0.4, order = backoff_order, vocabulary = vocab, counter = counter)

    return stupid_backoff, myBackoff


class MyBackoff(MLE):
    def __init__(self, alpha = 0.4, order = 2, vocabulary = None, counter = None):
        super().__init__(order, vocabulary) 
        self.alpha = alpha
        self.counter = counter

    def unmasked_score(self, word, context = None):
        if context is None or len(context) == 0:
            return self.counter[word] / self.counter.N()
        if self.counter[context][word] > 0:
            return self.counter[context][word] / self.counter[context].N()
        else:
            return self.alpha * self.unmasked_score(word, context[1:])
            
    def my_perplexity(self, ngrams):
        scores = [] 
        for ngram in ngrams:
            word = ngram[-1]
            context = ngram[:-1]
            scores.extend([-1 * math.log2(self.unmasked_score(word = word, context = context))])
        return math.pow(2.0, np.asarray(scores).mean())


def compare_test_sents_scores(nltk_stupid_backoff, my_stupid_backoff, sents):
    print("\n\nSents scores:")

    for sent in sents:
        print("\n\n",sent)
        padded_ngrams, _ = padded_everygram_pipeline(2, [sent.split()])
        padded_ngrams = [list(x) for x in padded_ngrams][0]
        ngrams = [x for x in padded_ngrams if len(x) == 2]
        
        for gram in ngrams:
            word = gram[-1]
            context = gram[:-1]
            print("NLTK StupidBackoff:\t\t", nltk_stupid_backoff.score(word = word, context = context))
            print("My StupidBackoff:\t\t", my_stupid_backoff.unmasked_score(word = word, context = context))
            print("------------------------------------------------------------------")


def compare_pp(nltk_stupid_backoff, my_stupid_backoff, sents, backoff_order = 2):
    ngrams_oov, _ = padded_everygram_pipeline(backoff_order, sents)
    ngrams = chain.from_iterable(ngrams_oov)
    ngrams = [x for x in ngrams if len(x) == backoff_order]

    print("\n\nPerplexities:")
    print("NLTK StupidBackoff:\t\t", nltk_stupid_backoff.perplexity(ngrams))
    print("My StupidBackoff:\t\t", my_stupid_backoff.my_perplexity(ngrams))
