import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import os 
import torch
import torch.utils.data as data

def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
        
    return dataset


def load_dataset():
    tmp_train_raw = load_data(os.path.join('dataset','train.json'))
    test_raw = load_data(os.path.join('dataset','test.json'))

    return tmp_train_raw, test_raw


def split_data(tmp_train_raw, test_raw):
    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10)/(len(tmp_train_raw)),2)

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    Y = []
    X = []
    mini_Train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occure once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=Y)
    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev
    
    return train_raw, dev_raw, test_raw


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True, PAD_TOKEN = 0):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True, PAD_TOKEN = 0):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    

class IntentsAndSlots (data.Dataset):
    Max_len = 1
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, max_len, tokenizer, unk='unk'):
        IntentsAndSlots.Max_len = max_len
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = self.utterances[idx]
        encodings = self.tokenizer.encode_plus(
            utt,
            max_length=IntentsAndSlots.Max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        
        ids = encodings['input_ids'][0]
        mask = encodings['attention_mask'][0]
        
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent, 'ids': ids, 'mask': mask}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res