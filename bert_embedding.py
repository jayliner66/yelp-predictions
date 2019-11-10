import torch
from transformers import *

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

model = BertModel.from_pretrained(pretrained_weights)
input_ids = torch.tensor([tokenizer.encode("Seven is greater than eight", add_special_tokens=True)])
with torch.no_grad():
    enc = model(input_ids)[0]

print(enc)
