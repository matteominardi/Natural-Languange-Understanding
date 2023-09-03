import functions
from collections import Counter
import spacy
import nltk
nltk.download('punkt')
nltk.download('gutenberg')

source = "shakespeare-macbeth.txt"

print("REFERENCE STATISTICS")
words = nltk.corpus.gutenberg.words(source)
sents = nltk.corpus.gutenberg.sents(source)

functions.print_desc_statistics(words, sents)
print("-------------------------------------------------------")

chars = nltk.corpus.gutenberg.raw(source)

print("\n\nSPACY STATISTICS")
nlp = spacy.load("en_core_web_sm")
doc = nlp(chars)
words_spacy = [token.text for token in doc]
sents_spacy = list(doc.sents)

functions.print_desc_statistics(words_spacy, sents_spacy)
print("-------------------------------------------------------")


print("\n\nNLTK STATISTICS")
words_nltk = nltk.word_tokenize(chars)
sents_nltk = nltk.sent_tokenize(chars)

functions.print_desc_statistics(words_nltk, sents_nltk)
print("-------------------------------------------------------")


print("\n\nCOMPARING LOWERCASE LEXICON SIZES")
words_lower = [word.lower() for word in words]
words_spacy_lower = [word.lower() for word in words_spacy]
words_nltk_lower = [word.lower() for word in words_nltk]

lexicon = set(words_lower)
lexicon_spacy = set(words_spacy_lower)
lexicon_nltk = set(words_nltk_lower)

print("Reference lexicon size:", len(lexicon))
print("Spacy lexicon size:", len(lexicon_spacy))
print("NLTK lexicon size:", len(lexicon_nltk))
print("-------------------------------------------------------")


print("\n\nCOMPARING TOP N FREQUENCIES")
n = 5
lex_freq_list = Counter(words_lower) 
lex_freq_list_spacy = Counter(words_spacy_lower) 
lex_freq_list_nltk = Counter(words_nltk_lower) 

print("Reference n best:", functions.nbest(lex_freq_list, n))
print("Spacy n best:", functions.nbest(lex_freq_list_spacy, n))
print("NLTK n best:", functions.nbest(lex_freq_list_nltk, n))