#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:28:06 2019

@author: adam
"""

import torch
from transformers import *

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

model = BertModel.from_pretrained(pretrained_weights)

string1="Seven is greater than eight"
string2="Seven is greater than nine"
string3="Apples are red and blue and orange"

def to_bert(string):
    input_ids = torch.tensor([tokenizer.encode(string, add_special_tokens=True)])
    with torch.no_grad():
        enc = model(input_ids, token_type_ids=None)[0]
    return enc

e1 = to_bert(string1)
e2 = to_bert(string2)
e3 = to_bert(string3)

print(e3.size())

