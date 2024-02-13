from nltk.parse.dependencygraph import DependencyGraph
from nltk.corpus import dependency_treebank
from nltk.parse import DependencyEvaluator

def evaluate(title, pipe, sents):
    print(title)

    dps = []

    for sent in sents:
        sent = ' '.join(sent)
        doc = pipe(sent)
        df = doc._.pandas
        tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        dp = DependencyGraph(tmp)
        dps.append(dp)
        
    de = DependencyEvaluator(dps, dependency_treebank.parsed_sents()[-100:])
    las, uas = de.eval()
    print('LAS', las, 'UAS', round(uas,3))