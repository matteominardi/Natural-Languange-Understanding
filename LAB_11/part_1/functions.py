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
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def get_dataloaders(train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    return train_loader, test_loader


def remove_objective(dataset, labels):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    res_dataset = dataset.copy()
    res_labels = labels.copy()
    res_dataset = dataset[:]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = MyBERT().to(device)
    path = "bin/model_1.pt"
    model.load_state_dict(torch.load(path))

    import time
    print(len(dataset))
    i = 0
    for idx, sent in enumerate(dataset):
        i = i + 1
        if i % 500 == 0: 
            print(i)
        inputs = tokenizer.encode_plus(
            sent,
            max_length=512,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        
        ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)
        label = model(ids, mask)
        out_label = torch.argmax(label, dim=1)
        
        if out_label == 1:          
            res_dataset.remove(sent)
            res_labels[idx] = ""

    res_labels = [s for s in res_labels if s.strip() != ""]
    
    print(len(dataset), len(res_dataset), len(res_labels))
    return res_dataset, res_labels


def train_and_eval(train_loader, test_loader, min_loss, name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    lr = 0.0001 # learning rate
    runs = 5
    losses_train = []
    acc = []

    for x in tqdm(range(0, runs)):
        model = MyBERT().to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_labels = nn.CrossEntropyLoss()
        
        n_epochs = 15
        patience = 3
         
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_labels, model)
            curr_loss = np.asarray(loss).mean()
            losses_train.append(curr_loss)
            
            if curr_loss < min_loss:
                min_loss = curr_loss
                save_model(model, name)
            else:
                patience -= 1
            if patience <= 0: 
                break 

        report, _ = eval_loop(test_loader, criterion_labels, model)
        print("Run:", x, " - Acc:", report["accuracy"])
        acc.append(report["accuracy"])
        
    acc = np.asarray(acc)
    print("Accuracy", round(acc.mean(), 3))
    return min_loss


def train_loop(data, optimizer, criterion_labels, model):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        label = model(sample["ids"], sample["mask"])
        loss = criterion_labels(label, sample["label"])
        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()
    return loss_array


def eval_loop(data, criterion_labels, model):
    model.eval()
    loss_array = []
    
    ref_labels = []
    hyp_labels = []
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            label = model(sample["ids"], sample["mask"])

            loss = criterion_labels(label, sample["label"])
            loss_array.append(loss.item())
            
            out_labels = torch.argmax(label, dim=1)
            gt_labels = sample["label"]

            ref_labels.extend(gt_labels.cpu())
            hyp_labels.extend(out_labels.cpu())
        
    report = classification_report(ref_labels, hyp_labels, zero_division=False, output_dict=True)
    return report, loss_array


def collate_fn(data, PAD_TOKEN = 0, device = "cuda:0" if torch.cuda.is_available() else "cpu"):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = SubjAndObj.Max_len
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, sent in enumerate(sequences):
            end = lengths[i]
            for j in range(0, end):
                padded_seqs[i, j] = sent[j] 
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs
    
    data.sort(key=lambda x: len(x['sent']), reverse=True) 

    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    src_sent = merge(new_item['sent'])
    label = torch.LongTensor(new_item["label"])
    
    src_sent = src_sent.to(device) # We load the Tensor on our seleceted device
    label = label.to(device)
    ids = pad_sequence(new_item["ids"], batch_first=True, padding_value=0).to(device)
    mask = pad_sequence(new_item["mask"], batch_first=True, padding_value=0).to(device)
    
    new_item["sent"] = src_sent
    new_item["label"] = label
    new_item["ids"] = ids
    new_item["mask"] = mask
    return new_item