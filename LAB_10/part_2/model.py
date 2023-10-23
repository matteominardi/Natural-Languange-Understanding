import torch
import torch.nn as nn
from transformers import BertModel


class MyBERT(nn.Module):
    def __init__(self, out_slot, out_int):
        super(MyBERT, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ids, mask):
        output = self.bert(input_ids=ids, attention_mask=mask)
        last_hidden = output.last_hidden_state # for slot filling
        pool_output = output.pooler_output # for intent classification
        
        slots = self.dropout(self.slot_out(last_hidden))
        # print("\n\nmodel slots", slots.size())
        intent = self.dropout(self.intent_out(pool_output))
        # print("model intent", intent.size())

        # Slot size: seq_len, batch size, calsses 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    

def save_model(model):
    path = "bin/model.pt"
    torch.save(model.state_dict(), path)