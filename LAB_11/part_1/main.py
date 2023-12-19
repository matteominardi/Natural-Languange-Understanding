from utils import *
from functions import *
from sklearn.model_selection import StratifiedKFold


from nltk.corpus import subjectivity
# print("Subjectivity task")
# sents, labels = load_dataset(base = True)
# skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
# min_loss = float("inf")

# for i, (train_index, test_index) in enumerate(skf.split(sents, labels)):
#     x_train, x_test = [sents[indx] for indx in train_index], [sents[indx] for indx in test_index]
#     y_train, y_test = [labels[indx] for indx in train_index], [labels[indx] for indx in test_index]

#     train_raw = list(zip(x_train, y_train))
#     test_raw = list(zip(x_test, y_test))

#     train_dataset, test_dataset = get_datasets_task_1(train_raw, test_raw, subjectivity)
#     train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

#     print("Model 1 for Subjectivity - Iteration:", i)
#     min_loss = train_and_eval(train_loader, test_loader, min_loss, "model_1")

# print("---------------------------------------------------------------------------------")
from nltk.corpus import movie_reviews
# print("Polarity task - standard")
# sents, labels = load_dataset(base = False)
# skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
# min_loss = float("inf")
# stop_words = set(stopwords.words("english"))

# for i, (train_index, test_index) in enumerate(skf.split(sents, labels)):
#     x_train, x_test = [sents[indx] for indx in train_index], [sents[indx] for indx in test_index]
#     y_train, y_test = [labels[indx] for indx in train_index], [labels[indx] for indx in test_index]
    
#     x_train = [remove_stopwords_and_punctuation(sent, stop_words) for sent in x_train]
#     x_test = [remove_stopwords_and_punctuation(sent, stop_words) for sent in x_test]

#     train_raw = list(zip(x_train, y_train))
#     test_raw = list(zip(x_test, y_test))

#     train_raw = [(words, label) for words, label in train_raw if words]
#     test_raw = [(words, label) for words, label in test_raw if words]

#     train_dataset, test_dataset = get_datasets_task_2(train_raw, test_raw, movie_reviews)
#     train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

#     print("Model 2 for Polarity - Iteration:", i)
#     min_loss = train_and_eval(train_loader, test_loader, min_loss, "model_2")

# print("---------------------------------------------------------------------------------")
# print("Polarity task - removing objective sentences")
# sents, labels = load_dataset(base = False)
# sents, labels = remove_objective(list(sents), list(labels))
# skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
# min_loss = float("inf")
# stop_words = set(stopwords.words("english"))

# for i, (train_index, test_index) in enumerate(skf.split(sents, labels)):
#     x_train, x_test = [sents[indx] for indx in train_index], [sents[indx] for indx in test_index]
#     y_train, y_test = [labels[indx] for indx in train_index], [labels[indx] for indx in test_index]

#     x_train = [remove_stopwords_and_punctuation(sent, stop_words) for sent in x_train]
#     x_test = [remove_stopwords_and_punctuation(sent, stop_words) for sent in x_test]

#     print("Removed stopwords and punctuation")

#     print(len(x_train), len(y_train))

#     train_raw = list(zip(x_train, y_train))
#     test_raw = list(zip(x_test, y_test))

#     train_raw = [(words, label) for words, label in train_raw if words]
#     test_raw = [(words, label) for words, label in test_raw if words]

#     train_dataset, test_dataset = get_datasets_task_2(train_raw, test_raw, movie_reviews)
#     train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

#     print("Model 3 for Polarity (removing objective sents) - Iteration:", i)
#     min_loss = train_and_eval(train_loader, test_loader, min_loss, "model_3")

print("---------------------------------------------------------------------------------")
print("Polarity task - removing objective sentences")
sents, labels = load_dataset(base = False)
sents, labels = remove_objective(list(sents), list(labels))
skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
min_loss = float("inf")
stop_words = set(stopwords.words("english"))

for i, (train_index, test_index) in enumerate(skf.split(sents, labels)):
    x_train, x_test = [sents[indx] for indx in train_index], [sents[indx] for indx in test_index]
    y_train, y_test = [labels[indx] for indx in train_index], [labels[indx] for indx in test_index]

    x_train = [remove_stopwords(sent, stop_words) for sent in x_train]
    x_test = [remove_stopwords(sent, stop_words) for sent in x_test]

    print("Removed stopwords")

    print(len(x_train), len(y_train))

    train_raw = list(zip(x_train, y_train))
    test_raw = list(zip(x_test, y_test))

    train_raw = [(words, label) for words, label in train_raw if words]
    test_raw = [(words, label) for words, label in test_raw if words]

    train_dataset, test_dataset = get_datasets_task_2(train_raw, test_raw, movie_reviews)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

    print("Model 4 for Polarity (removing objective sents) - Iteration:", i)
    min_loss = train_and_eval(train_loader, test_loader, min_loss, "model_4")