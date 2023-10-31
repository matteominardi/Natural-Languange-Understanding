import torch
import torch.nn as nn
from transformers import BertModel


class BERT_task1(nn.Module):
    def __init__(self):
        super(BERT_task1, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.label_out = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ids, mask):
        output = self.bert(input_ids=ids, attention_mask=mask)
        pool_output = output.pooler_output 
        
        label = self.dropout(self.label_out(pool_output))
        return label
    

def save_model(model):
    path = "bin/model.pt"
    torch.save(model.state_dict(), path)