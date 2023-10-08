from functions import cosine_similarity
import torch.nn as nn
import numpy as np
import numpy as np
from numpy.linalg import norm


class LM(nn.Module):
    def __init__(self, 
                emb_size, 
                hidden_size, 
                output_size,
                lstm=False,
                dropout=False, 
                pad_index=0, 
                out_dropout=0.1,
                emb_dropout=0.1, 
                n_layers=1):
        
        super(LM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        if lstm:
            self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        else:
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False)

        if dropout:
            self.emb_dropout = nn.Dropout(emb_dropout)
            self.out_dropout = nn.Dropout(out_dropout)
        
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)


    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        
        return output
    

    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()


    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        
        return (indexes, top_scores)


class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}


    def get_vocab(self, corpus, special_tokens=[]):
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