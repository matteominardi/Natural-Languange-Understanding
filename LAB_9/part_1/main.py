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
    rnn_base = train_and_eval(lang, train_loader, dev_loader, test_loader)
    print("PPL with LSTM")
    lstm_base = train_and_eval(lang, train_loader, dev_loader, test_loader, lstm=True)
    print("PPL with LSTM and dropout layers")
    lstm_dropout = train_and_eval(lang, train_loader, dev_loader, test_loader, lstm=True, dropout=True)
    print("PPL with LSTM and dropout layers and AdamW")
    lstm_dropout_adamw = train_and_eval(lang, train_loader, dev_loader, test_loader, lstm=True, dropout=True, adamw=True)

    models = [rnn_base, lstm_base, lstm_dropout, lstm_dropout_adamw]

    save_models(models, models_names)