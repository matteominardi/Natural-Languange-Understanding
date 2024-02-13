from functions import *
import nltk
from pcfg import PCFG
from nltk.parse.generate import generate

if __name__ == "__main__":
    sent1 = "John saw a dog below the table"
    sent2 = "the dog took the plane"
    test_sents = [sent1, sent2]

    print_sents("Test sentences:", test_sents)

    print("--------------------------------------------------------------")

    print("\n\nDefining grammar rules and parsing test sentences:")

    rules = [
        'S -> NP VP [1.0]',
        'NP -> N [0.5] | Det N [0.5]',
        'VP -> V NP [0.5] | V NP PP [0.5]',
        'PP -> P NP [1.0]',
        'P -> "below" [1.0]',
        'Det -> "the" [0.5] | "a" [0.5]',
        'N -> "John" [0.25] | "dog" [0.25] | "table" [0.25] | "plane" [0.25]',
        'V -> "saw" [0.5] | "took" [0.5]'
    ]
    grammar = nltk.PCFG.fromstring(rules)
    parser = nltk.ChartParser(grammar)

    for sent in test_sents:
        print("\nParsing sentence:", sent)
        for tree in parser.parse(sent.split()):
            print(tree.pretty_print())

    print("--------------------------------------------------------------")

    print("\n\nGenerating 10 sentences from the grammar using nltk.parse.generate:")
    for sent in generate(grammar, n=10):
            print(' '.join(sent))

    print("--------------------------------------------------------------")

    print("\n\nGenerating 10 sentences from the grammar using PCFG.generate():")
    grammar = PCFG.fromstring(rules)
    for sent in grammar.generate(10):
        print(sent)