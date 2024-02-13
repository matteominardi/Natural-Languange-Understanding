from functions import *
from sklearn.datasets import fetch_20newsgroups 
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = fetch_20newsgroups(subset='all')
    model = LinearSVC(C=0.1, max_iter=1000)

    print("\n\n(CountVect) Binary of Count Vectorization")
    vectorizer = CountVectorizer(binary = True, lowercase = True)
    experiment(df, model, vectorizer)
    print("----------------------------------------------------------------")

    print("\n\n(TF-IDF) TF-IDF using lowercase and stop words")
    tfidfvectorizer = TfidfVectorizer(lowercase = True, stop_words = 'english')
    experiment(df, model, tfidfvectorizer)
    print("----------------------------------------------------------------")

    print("\n\n(CutOff) TF-IDF using min and max cut-offs")
    tfidfvectorizer = TfidfVectorizer(lowercase = True, stop_words = 'english', min_df = 0.00001, max_df = 0.04)
    experiment(df, model, tfidfvectorizer)
    print("----------------------------------------------------------------")

    print("\n\n(WithoutStopWords) TF-IDF without using stop words")
    tfidfvectorizer = TfidfVectorizer(lowercase = True)
    experiment(df, model, tfidfvectorizer)
    print("----------------------------------------------------------------")

    print("\n\n(NoLowercase) TF-IDF without using lowercase")
    tfidfvectorizer = TfidfVectorizer(lowercase = False)
    experiment(df, model, tfidfvectorizer)