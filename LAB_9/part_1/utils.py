from functions import *
from model import *
import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line + eos_token)
    return output


def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0 
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {"source": src, "target": trg}
        return sample
    
    def mapping_seq(self, data, lang): 
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print("OOV found!")
                    print("You have to deal with that") 
                    break
            res.append(tmp_seq)
        return res


def get_lang_loaders():
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    return lang, train_loader, dev_loader, test_loader


def train_and_eval(lr, lang, train_loader, dev_loader, test_loader, lstm=False, dropout=False, adamw=False):
    hid_size = 200
    emb_size = 300
    vocab_len = len(lang.word2id)
    device = 'cuda:0'
    model = LM(emb_size, hid_size, vocab_len, lstm = lstm, dropout = dropout, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
 
    clip = 5 

    if adamw:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    test_PPL(best_model, test_loader, criterion_eval)

    return best_model


def save_models(models, models_names):
    for i in range(len(models)):
        path = "bin/" + models_names[i] + ".pt"
        torch.save(models[i].state_dict(), path)


def load_models(models_names):
    lang, _, _, _ = get_lang_loaders()
    hid_size = 200
    emb_size = 300
    vocab_len = len(lang.word2id)
    device = 'cuda:0'

    rnn_base = LM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    lstm_base = LM(emb_size, hid_size, vocab_len, lstm=True, pad_index=lang.word2id["<pad>"]).to(device)
    lstm_dropout = LM(emb_size, hid_size, vocab_len, lstm=True, dropout=True, pad_index=lang.word2id["<pad>"]).to(device)
    lstm_dropout_adamw = LM(emb_size, hid_size, vocab_len, lstm=True, dropout=True, adamw=True, pad_index=lang.word2id["<pad>"]).to(device)

    models = [rnn_base, lstm_base, lstm_dropout, lstm_dropout_adamw]

    for i in range(len(models_names)):
        path = "/bin/" + models_names[i] + ".pt"
        models[i].load_state_dict(torch.load(path))

    return models