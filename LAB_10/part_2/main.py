from utils import *
from model import *
from functions import *


tmp_train_raw, test_raw = load_dataset()
train_raw, dev_raw, test_raw = split_data(tmp_train_raw, test_raw)

lang, train_dataset, dev_dataset, test_dataset = get_lang_datasets(train_raw, dev_raw, test_raw)

train_loader, dev_loader, test_loader = get_dataloaders(train_dataset, dev_dataset, test_dataset)

print("MyBERT:")
train_and_eval(lang, train_loader, dev_loader, test_loader)