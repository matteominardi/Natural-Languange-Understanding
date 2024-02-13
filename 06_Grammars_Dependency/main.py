from functions import *
import nltk
nltk.download('dependency_treebank')
import spacy 
import spacy_stanza
from spacy.tokenizer import Tokenizer
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    treebank = nltk.corpus.dependency_treebank
    sents = treebank.sents()[-100:]

    spacy_nlp = nlp = spacy.load("en_core_web_sm")
    spacy_nlp.tokenizer = Tokenizer(spacy_nlp.vocab, token_match=None)
    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"DEPREL": {"nsubj": "subj"}}}
    spacy_nlp.add_pipe("conll_formatter", config=config, last=True)

    evaluate("Evaluating Spacy", spacy_nlp, sents)

    print("-------------------------------------------------------------------")

    stanza_nlp = spacy_stanza.load_pipeline('en', verbose=False, tokenize_pretokenized = True)
    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"deprel": {"nsubj": "subj", "root":"ROOT"}}}
    stanza_nlp.add_pipe("conll_formatter", config=config, last=True)

    evaluate("Evaluating Stanza", stanza_nlp, sents)