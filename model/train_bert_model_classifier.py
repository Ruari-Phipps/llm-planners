import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()

import math

import glob

import numpy as np

import math

from os.path import join
import os

location = os.path.dirname(os.path.realpath(__file__))

import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine 

import datetime

from random import randint, choice

from .action_sentences import place_by_sentences_inc, place_by_sentences_non_inc, place_on_sentences_inc, place_on_sentences_non_inc, grab_sentences, articles, place_in_sentences_inc, place_in_sentences_non_inc

COLOURS = {
    "white": [1.0, 1.0, 1.0],
    "grey": [0.5, 0.5, 0.5],
    "black": [0.0, 0.0, 0.0],
    "red": [1.0, 0.1, 0.1],
    "green": [0.1, 0.7, 0.1],
    "blue": [0.1, 0.1, 1.0],
    "yellow": [1.0, 1.0, 0.1],
}


OBJECT_TYPES = [
    {
        "type": "cylinder",
        "size": [0.04, 0.04, 0.05],
        "names": ["cylinder", "tube", "can", "tin"]
    },
    {
        "type": "cube",
        "size": [0.05, 0.05, 0.05],
        "names": ["cube", "block"]
    },
    {
        "type": "box",
        "size": [0.03, 0.05, 0.06],
        "names": ["box", "cuboid"]
    },
    {
        "type": "sphere",
        "size": [0.05, 0.05, 0.05],
        "names": ["sphere", "ball", "orb", "globe"]
    },
    {
        "type": "slice",
        "size": [0.05, 0.05, 0.02],
        "names": ["slice"]
    }
]


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
class BertTrain(nn.Module):
    def __init__(
        self, bert="bert-base-uncased", bert_size = 768
    ) -> None:
        super().__init__()

        self.bert_model = AutoModel.from_pretrained(
            bert,
            output_hidden_states=True,  # Whether the model returns all hidden-states.
        )
        self.tokenizer = AutoTokenizer.from_pretrained(bert)

        self.fc1= nn.Linear(bert_size, 4)
        self.colour = nn.Linear(bert_size, 7)
        self.o = nn.Linear(bert_size, 7)
        self.soft = nn.Softmax(dim=1)

    def forward(self, X):
        tokenised = self.tokenizer.batch_encode_plus(
            X,
            max_length = 25,
            padding='max_length',
            truncation=True
        )
        bert = self.bert_model(torch.tensor(tokenised['input_ids']).to(device), attention_mask = torch.tensor(tokenised['attention_mask']).to(device), return_dict=False)[1]

        X = self.soft(self.fc1(bert))
        Y = self.soft(self.colour(bert))
        Z = self.soft(self.o(bert))

        return X, Y, Z

def train(
    X, Y, batch_size=64, epochs=20, lr=0.001, wd=1e-2
):
    now = datetime.datetime.now()
    n = X.shape[0]
    idx = torch.randperm(n)
    X = X[idx]
    Y = Y[idx]

    train_amount = int(0.9 * n)
    X_train, X_val = X[:train_amount], X[train_amount:]
    Y_train, Y_val = Y[:train_amount], Y[train_amount:]

    batches = math.ceil(train_amount / batch_size)
    batches_val = math.ceil((n - train_amount) / batch_size)

    model = BertTrain().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss = torch.nn.CrossEntropyLoss(reduction="mean")

    v_losses = []

    for epoch in range(epochs):
        t_loss = 0
        print(f"EPOCH {epoch + 1}/{epochs}")
        model.train()
        for i in range(batches):
            loc = i * batch_size
            print(f"Batch {i+1}/{batches}", end="\r", flush=True)
            batch_X = list(X_train[loc : loc + batch_size])

            batch_Y = (
                torch.tensor(Y_train[loc : loc + batch_size])
                .long()
                .to(device)
            )

            optimiser.zero_grad()
            a, c, o = model.forward(batch_X)
            l = loss(a, batch_Y[:,0]) + loss(c, batch_Y[:,1]) + loss(o, batch_Y[:,2])
            l.backward()
            optimiser.step()
            t_loss += l.item()

        t_loss = t_loss / batches
        model.eval()
        with torch.no_grad():
            v_loss = 0
            for i in range(batches_val):
                loc = i * batch_size
                batch_X = list(X_val[loc : loc + batch_size])

                batch_Y = (
                    torch.tensor(Y_val[loc : loc + batch_size])
                    .long()
                    .to(device)
                )
                a, c, o = model.forward(batch_X)
                l = loss(a, batch_Y[:,0]) + loss(c, batch_Y[:,1]) + loss(o, batch_Y[:,2])
                v_loss += l.item()

        v_loss = v_loss / batches_val
        v_losses.append(v_loss)
        print(f"\nTraining loss = {t_loss} Validation loss = {v_loss}")
        print(torch.argmax(a[:5], dim=1), torch.argmax(c[:5], dim=1), torch.argmax(o[:5], dim=1))
        print(batch_Y.detach().cpu().numpy()[:5])


    print("\nSAVING")
    torch.save(
        model.to(torch.device("cpu")).state_dict(),
        join(
            location,
            "models",
            "weights",
            "classifier",
            f"input_classifier.model",
        ),
    )


if __name__ == "__main__":
    data_count = 1000
    print("Loading data")
    X = []
    Y = []
    col = list(COLOURS.keys())
    for b in range(data_count):
        y = randint(0, 3)
        ob_colour = randint(0, len(col) - 1)
        tar_colour = randint(0, len(col) - 1)

        obj = randint(0, len(OBJECT_TYPES) - 1)
        tar = randint(0, len(OBJECT_TYPES) + 1)
        obj_description =f"{col[ob_colour]} {choice(OBJECT_TYPES[obj]['names'])}"

        if tar < 5:
            tar_obj_desc = choice(OBJECT_TYPES[tar]['names'])
        elif tar == 5:
            tar_obj_desc = "plate"
        else:
            tar_obj_desc = "bin"
        target_description =f"{col[tar_colour]} {tar_obj_desc}"

        

        if y == 0:
            x = choice(grab_sentences).format(choice(articles), obj_description)
            y = [y, ob_colour, obj]
        elif y == 1:
            if randint(0,1):
                x = choice(place_by_sentences_non_inc).format(choice(articles), target_description)
            else:
                x = choice(place_by_sentences_inc).format(choice(articles), obj_description, choice(articles), target_description)
            y = [y, tar_colour, tar]
        elif y == 2:
            if randint(0,1):
                x = choice(place_on_sentences_non_inc).format(choice(articles), target_description)
            else:
                x = choice(place_on_sentences_inc).format(choice(articles), obj_description, choice(articles), target_description)
            y = [y, tar_colour, tar]
        else:
            if randint(0,1):
                x = choice(place_in_sentences_non_inc).format(choice(articles), target_description)
            else:
                x = choice(place_in_sentences_inc).format(choice(articles), obj_description, choice(articles), target_description)
            y = [y, tar_colour, tar]
        X.append(x)
        Y.append(y)


    print("Done           ")
    train(
        np.array(X),
        np.array(Y),
        epochs=7,
        batch_size=32,
        lr=2e-4,
        wd=1e-3,
    )
