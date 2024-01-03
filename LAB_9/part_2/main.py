from functions import *
from model import *
from utils import *


if __name__ == "__main__":
    lang, train_loader, dev_loader, test_loader = get_lang_loaders()
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    old_model = load_old_model()
    print("\nPPL with LSTM and dropout layers and AdamW regularization")
    test_PPL(old_model, test_loader, criterion_eval)

    print("\nPPL with LSTM + dropout layers + weight tying and AdamW regularization")
    lstm_weight_tying = train_and_eval(0.01, lang, train_loader, dev_loader, test_loader, weight_tying=True, adamw=True)
    save_models([lstm_weight_tying], ["lstm_weight_tying"])

    print("\nPPL with LSTM + dropout layers + weight tying + variational dropout and AdamW regularization")
    lstm_variational_dropout = train_and_eval(0.08, lang, train_loader, dev_loader, test_loader, weight_tying=True, variational_dropout=True, adamw=True)
    save_models([lstm_variational_dropout], ["lstm_variational_dropout"])

    lr = 1.8
    print("\nPPL with LSTM + dropout layers + weight tying + variational dropout and NTAvSGD optimizer with lr=" + str(lr))
    lstm_ntavsgd = train_and_eval(lr, lang, train_loader, dev_loader, test_loader, weight_tying=True, variational_dropout=True, adamw=False, ntavsgd=True, n=2)
    save_models([lstm_ntavsgd], ["lstm_ntavsgd"])