from utils import *
from model import *
from functions import *


if __name__ == "__main__":
    tmp_train_raw, test_raw = load_dataset()
    train_raw, dev_raw, test_raw = split_data(tmp_train_raw, test_raw)

    lang, train_dataset, dev_dataset, test_dataset = get_lang_datasets(train_raw, dev_raw, test_raw)

    train_loader, dev_loader, test_loader = get_dataloaders(train_dataset, dev_dataset, test_dataset)

    print("ModelIAS baseline:")
    train_and_eval(lang, train_loader, dev_loader, test_loader, bidirectional=False, dropout=False, PAD_TOKEN = 0)
    print("--------------------------------------------------------")

    print("ModelIAS bidirectional:")
    train_and_eval(lang, train_loader, dev_loader, test_loader, bidirectional=True, dropout=False, PAD_TOKEN = 0)
    print("--------------------------------------------------------")

    print("ModelIAS bidirectional and dropout:")
    train_and_eval(lang, train_loader, dev_loader, test_loader, bidirectional=True, dropout=True, PAD_TOKEN = 0)
    print("--------------------------------------------------------")