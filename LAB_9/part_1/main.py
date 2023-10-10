from functions import *
from model import *
from utils import *


just_test = False
models_names = ["rnn_base", "lstm_base", "lstm_dropout", "lstm_dropout_adamw"]
lang, train_loader, dev_loader, test_loader = get_lang_loaders()

if just_test:
    models = load_models(models_names)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    print("PPL with RNN")
    test_PPL(models[0], test_loader, criterion_eval)
    print("PPL with LSTM")
    test_PPL(models[1], test_loader, criterion_eval)
    print("PPL with LSTM and dropout layers")
    test_PPL(models[2], test_loader, criterion_eval)
    print("PPL with LSTM and dropout layers and AdamW")
    test_PPL(models[3], test_loader, criterion_eval)
else:
    print("PPL with RNN")
    rnn_base = train_and_eval(0.1, lang, train_loader, dev_loader, test_loader)
    save_models([rnn_base], ["rnn_base"])
    print("PPL with LSTM")
    lstm_base = train_and_eval(1.0, lang, train_loader, dev_loader, test_loader, lstm=True)
    save_models([lstm_base], ["lstm_base"])
    print("PPL with LSTM and dropout layers")
    lstm_dropout = train_and_eval(1.0, lang, train_loader, dev_loader, test_loader, lstm=True, dropout=True)
    save_models([lstm_dropout], ["lstm_dropout"])
    print("PPL with LSTM and dropout layers and AdamW")
    lstm_dropout_adamw = train_and_eval(0.01, lang, train_loader, dev_loader, test_loader, lstm=True, dropout=True, adamw=True)
    save_models([lstm_dropout_adamw], ["lstm_dropout_adamw"])