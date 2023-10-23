from model import *
from utils import *
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from conll import evaluate
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        # print("\n\nSample Ids")
        # print(sample["ids"].size())
        # print("\n\nSample masks")
        # print(sample["mask"].size())
        slots, intent = model(sample["ids"], sample["mask"])
        loss_intent = criterion_intents(intent, sample["intents"])
        # print("\n\n")
        # print("intent")
        # print(len(intent))
        # print(intent.size())
        # print(intent)
        # print("\n\n")
        # print("slots")
        # print(len(slots))
        # print(slots.size())
        # print(slots)
        # print("\n")
        # print("y_slots")
        # print(len(sample["y_slots"]))
        # print(sample["y_slots"].size())
        # print(sample["y_slots"])
        # print("\n\n")
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample["ids"], sample["mask"])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        pass
        # Sometimes the model predics a class that is not in REF
        # print(ex)
        # ref_s = set([x[1] for x in ref_slots])
        # hyp_s = set([x[1] for x in hyp_slots])
        # print(hyp_s.difference(ref_s))
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def collate_fn(data, PAD_TOKEN = 0, device = "cuda:0" if torch.cuda.is_available() else "cpu"):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        # max_len = 1 if max(lengths)==0 else max(lengths)
        max_len = 52
        # print("\n\nmax_len")
        # print(max_len)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 

    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our seleceted device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    ids = pad_sequence(new_item["ids"], batch_first=True, padding_value=0).to(device)
    mask = pad_sequence(new_item["mask"], batch_first=True, padding_value=0).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["ids"] = ids
    new_item["mask"] = mask

    return new_item


def get_dataloaders(train_dataset, dev_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


def get_lang_datasets(train_raw, dev_raw, test_raw):
    words = sum(
        [x["utterance"].split() for x in train_raw], []
    )  # No set() since we want to compute
    # the cutoff
    corpus = train_raw + dev_raw + test_raw  # We do not wat unk labels,
    # however this depends on the research purpose
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = max([len(tokenizer.encode(x['utterance'])) for x in corpus])

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang, max_len, tokenizer)
    val_dataset = IntentsAndSlots(dev_raw, lang, max_len, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, max_len, tokenizer)

    return lang, train_dataset, val_dataset, test_dataset


def train_and_eval(lang, train_loader, dev_loader, test_loader, bidirectional=False, dropout=False, PAD_TOKEN = 0):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    lr = 0.0001 # learning rate

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    # print("\n\nOut_slot")
    # print(out_slot)
    # print("\n\nOut_int")
    # print(out_int)

    runs = 5
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = MyBERT(out_slot=out_slot, out_int=out_int).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        n_epochs = 15
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        for x in range(1,n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0: # Early stoping with patient
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(), 3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    save_model(model)