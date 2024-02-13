import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import os 
import torch
import string
import torch.utils.data as data
import nltk
nltk.download("subjectivity")
from nltk.corpus import subjectivity
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
import nltk
nltk.download('punkt')
from transformers import BertTokenizer


def load_dataset(base = True):
    if base:
        sents_subj = subjectivity.sents(categories='subj')
        sents_obj = subjectivity.sents(categories='obj')
        X = sents_subj + sents_obj
        y = ["subj"]*len(sents_subj) + ["obj"]*len(sents_obj)
    else:
        reviews_pos = movie_reviews.sents(categories='pos')
        reviews_neg = movie_reviews.sents(categories='neg')
        X = reviews_pos + reviews_neg
        y = ["pos"]*len(reviews_pos) + ["neg"]*len(reviews_neg)

    dataset = list(zip(X, y))
    random.shuffle(dataset)
    X, y = zip(*dataset)
    return X, y


def remove_stopwords(sentence, stop_words):
    return [word for word in sentence   if  (word.lower() not in stop_words)]


def remove_stopwords_and_punctuation(sentence, stop_words):
    return [word for word in sentence   if  (word.lower() not in stop_words) 
                                            and (word not in string.punctuation)]


def get_datasets_task_1(train_raw, test_raw, corpus):
    dataset = train_raw + test_raw  
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    max_len = max([len(tokenizer.encode(sent)) for (sent, _) in dataset])
    lang = Lang(corpus.words(), corpus.categories(), cutoff=0)
    train_dataset = SubjAndObj(train_raw, max_len, tokenizer, lang)
    test_dataset = SubjAndObj(test_raw, max_len, tokenizer, lang)

    return train_dataset, test_dataset


def get_datasets_task_2(train_raw, test_raw, corpus):
    dataset = train_raw + test_raw  
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    max_len = max([len(tokenizer.encode(sent)) for (sent, _) in dataset])
    lang = Lang(corpus.words(), corpus.categories(), cutoff=0)
    train_dataset = PosAndNeg(train_raw, max_len, tokenizer, lang)
    test_dataset = PosAndNeg(test_raw, max_len, tokenizer, lang)

    return train_dataset, test_dataset


class Lang():
    def __init__(self, words, labels, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.label2id = self.lab2id(labels, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2label = {v:k for k, v in self.label2id.items()}
        
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
    

class SubjAndObj(data.Dataset):
    Max_len = 1
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, max_len, tokenizer, lang):
        SubjAndObj.Max_len = max_len
        self.sents = []
        self.labels = []
        self.tokenizer = tokenizer
        self.unk = "unk"
        
        for sent, label in dataset:
            self.sents.append(sent)
            self.labels.append(label)

        self.sent_ids = self.mapping_seq(self.sents, lang.word2id)
        self.label_ids = self.mapping_lab(self.labels, lang.label2id)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sent = self.sents[idx]

        encodings = self.tokenizer.encode_plus(
            sent,
            max_length=SubjAndObj.Max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        sent = torch.Tensor(self.sent_ids[idx])
        label = self.label_ids[idx]

        ids = encodings['input_ids'][0]
        mask = encodings['attention_mask'][0]
        
        sample = {'sent': sent, 'label': label, 'ids': ids, 'mask': mask}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


class PosAndNeg(data.Dataset):
    Max_len = 1
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, max_len, tokenizer, lang):
        SubjAndObj.Max_len = max_len
        self.sents = []
        self.labels = []
        self.tokenizer = tokenizer
        self.unk = "unk"
        
        for sent, label in dataset:
            self.sents.append(sent)
            self.labels.append(label)

        self.sent_ids = self.mapping_seq(self.sents, lang.word2id)
        self.label_ids = self.mapping_lab(self.labels, lang.label2id)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sent = self.sents[idx]

        encodings = self.tokenizer.encode_plus(
            sent,
            max_length=SubjAndObj.Max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        sent = torch.Tensor(self.sent_ids[idx])
        label = self.label_ids[idx]

        ids = encodings['input_ids'][0]
        mask = encodings['attention_mask'][0]
        
        sample = {'sent': sent, 'label': label, 'ids': ids, 'mask': mask}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res