from utils import *
from functions import *
from sklearn.model_selection import StratifiedKFold


sents, labels = load_dataset(base = True)
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
min_loss = float("inf")

for i, (train_index, test_index) in enumerate(skf.split(sents, labels)):
    x_train, x_test = [sents[indx] for indx in train_index], [sents[indx] for indx in test_index]
    y_train, y_test = [labels[indx] for indx in train_index], [labels[indx] for indx in test_index]

    train_raw = list(zip(x_train, y_train))
    test_raw = list(zip(x_test, y_test))

    train_dataset, test_dataset = get_datasets(train_raw, test_raw)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

    print("Iteration:", i)
    min_loss = train_and_eval(train_loader, test_loader, min_loss)