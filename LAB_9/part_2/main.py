from functions import *
from model import *
from utils import *

lang, train_loader, dev_loader, test_loader = get_lang_loaders()
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

# old_model = load_old_model()
# print("PPL with LSTM and dropout layers and AdamW regularization")
# test_PPL(old_model, test_loader, criterion_eval)

# print("PPL with LSTM + dropout layers + weight tying and AdamW regularization")
# lstm_weight_tying = train_and_eval(0.01, lang, train_loader, dev_loader, test_loader, weight_tying=True, adamw=True)
# save_models([lstm_weight_tying], ["lstm_weight_tying"])

print("PPL with LSTM + dropout layers + weight tying + variational dropout and AdamW regularization with lr " + str(lr))
lstm_variational_dropout = train_and_eval(0.08, lang, train_loader, dev_loader, test_loader, weight_tying=True, variational_dropout=True, adamw=True)
save_models([lstm_variational_dropout], ["lstm_variational_dropout"])