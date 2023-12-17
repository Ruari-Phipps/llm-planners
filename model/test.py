import time
import numpy as np
from os.path import join, dirname, realpath
from scipy.spatial.distance import cosine 

from transformers import AutoTokenizer, AutoModel, logging

from train_bert_model_classifier import BertTrain

import torch

from models import ResNet18FiLM, ResNet18FiLMBert
bert="bert-base-uncased"
location = dirname(realpath(__file__))
MODEL_PATH = join(location, "models", "weights",)
# tokenizer = AutoTokenizer.from_pretrained(bert)
# model = AutoModel.from_pretrained(
#             bert,
#             output_hidden_states=True,  # Whether the model returns all hidden-states.
#         )
model = ResNet18FiLMBert(c_in=4, c_out=4)
model.load_state_dict(torch.load(join(MODEL_PATH, "full_policy", "full_policy_bert.model"), map_location=torch.device('cpu')))
model.eval()

input=["Pick up red cube", "Place by red cube", "Place on red cube"]
tokenised = model.tokenizer.batch_encode_plus(
            input,
            max_length = 25,
            padding='max_length',
            truncation=True
        )
film = model.bert_model(torch.tensor(tokenised['input_ids']), attention_mask = torch.tensor(tokenised['attention_mask']), return_dict=False)[1].detach()
print(cosine(film[0], film[1]))
print(cosine(film[0], film[2]))
print(cosine(film[1], film[2]))

bert_model = AutoModel.from_pretrained(
            bert,
            output_hidden_states=True,  # Whether the model returns all hidden-states.
        )
tokenizer = AutoTokenizer.from_pretrained(bert)

tokenised = tokenizer.batch_encode_plus(
            input,
            max_length = 25,
            padding='max_length',
            truncation=True
        )
print("-----------------")
film = bert_model(torch.tensor(tokenised['input_ids']), attention_mask = torch.tensor(tokenised['attention_mask']), return_dict=False)[1].detach()
print(cosine(film[0], film[1]))
print(cosine(film[0], film[2]))
print(cosine(film[1], film[2]))
