import torch
from torchvision import transforms

from PIL import Image
import glob
import pickle

import numpy as np

import math

from os.path import join
import os
import datetime

location = os.path.dirname(os.path.realpath(__file__))

import matplotlib.pyplot as plt

from models import ResNet18, ResNet18FiLM, ResNet18FiLMBert

use_gpu = True
device = torch.device(
    "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
)
print(device)


def train(
    camera, encodings, actions, texts=False, batch_size=64, epochs=20, lr=0.001, wd=1e-2
    #camera, actions, texts=False, batch_size=64, epochs=20, lr=0.001, wd=1e-2
):
    now = datetime.datetime.now()
    name = "full_policy_bert"
    n = actions.shape[0]
    idx = torch.randperm(n)
    camera = camera[idx]
    actions = actions[idx]
    encodings = encodings[idx]

    train_amount = int(0.9 * n)
    camera_train, camera_val = camera[:train_amount], camera[train_amount:]
    encoding_train, encoding_val = (
        encodings[:train_amount],
        encodings[train_amount:],
    )
    actions_train, actions_val = actions[:train_amount], actions[train_amount:]

    batches = math.ceil(train_amount / batch_size)
    batches_val = math.ceil((n - train_amount) / batch_size)
    c_in = camera.shape[1]
    c_out = actions.shape[1]
    # if not texts:
    #     film_size = encodings.shape[1]

    model = ResNet18FiLMBert(c_in=c_in, c_out=c_out).to(device)
    # model = ResNet18FiLMBert(c_in=c_in, c_out=c_out, bert_init=join(location, "models", "weights", "bert", "base.model")).to(device)
    # model = ResNet18FiLM(c_in=c_in, c_out=c_out, film_size=768).to(device)
    # model = ResNet18(c_in=c_in, c_out=c_out).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss = torch.nn.MSELoss(reduction="mean")

    t_losses = []
    v_losses = []

    for epoch in range(epochs):
        t_loss = 0
        print(f"EPOCH {epoch + 1}/{epochs}")
        model.train()
        for i in range(batches):
            loc = i * batch_size
            print(f"Batch {i+1}/{batches}", end="\r", flush=True)
            batch_cam = (
                torch.tensor(camera_train[loc : loc + batch_size])
                .float()
                .to(device)
            )
            if texts:
                batch_encoding = list(encoding_train[loc : loc + batch_size])
            else:
                batch_encoding = (
                    torch.tensor(encoding_train[loc : loc + batch_size])
                    .float()
                    .to(device)
                )
            batch_actions = (
                torch.tensor(actions_train[loc : loc + batch_size])
                .float()
                .to(device)
            )

            optimiser.zero_grad()
            results = model.forward(batch_cam, batch_encoding)
            l = loss(results, batch_actions)
            l.backward()
            optimiser.step()
            t_loss += l.item()


        t_loss = t_loss / batches
        t_losses.append(t_loss)
        model.eval()
        with torch.no_grad():
            v_loss = 0
            for i in range(batches_val):
                loc = i * batch_size
                batch_cam = (
                    torch.tensor(camera_val[loc : loc + batch_size])
                    .float()
                    .to(device)
                )
                if texts:
                    batch_encoding = list(encoding_val[loc : loc + batch_size])
                else:
                    batch_encoding = (
                        torch.tensor(encoding_val[loc : loc + batch_size])
                        .float()
                        .to(device)
                    )
                batch_actions = (
                    torch.tensor(actions_val[loc : loc + batch_size])
                    .float()
                    .to(device)
                )
                results = model.forward(batch_cam, batch_encoding)
                v_loss += loss(results, batch_actions).item()

        v_loss = v_loss / batches_val
        v_losses.append(v_loss)
        print(f"\nTraining loss = {t_loss} Validation loss = {v_loss}")
        print(results.detach().cpu().numpy()[:5])
        print(batch_actions.detach().cpu().numpy()[:5])

    print("\nSAVING")
    torch.save(
        model.to(torch.device("cpu")).state_dict(),
        join(
            location,
            "models",
            "weights",
            "full_policy",
            f"{name}.model",
        ),
    )
    with open(join(location, "models", "run_info", f"{name}_train.ob"), 'wb') as f:
        pickle.dump(t_losses, f)
    with open(join(location, "models", "run_info", f"{name}_val.ob"), 'wb') as f:
        pickle.dump(v_losses, f)




if __name__ == "__main__":
    data_count = len(
       glob.glob(join(location, "data", "full_policy", "actions", "*.npy"))
    )
    print(data_count)
    depths = []
    rgbs = []
    inputs = []
    actions = []
    encs = []
    texts = []
    print("Loading data")
    for i in range(0, data_count):
        print(f"{i}/{data_count}", end="\r", flush=True)
        inputs.append(np.load(
            join(location, "data", "full_policy", "inputs", f"{i}.npy")
        ))
        actions.append(np.load(
            join(location, "data", "full_policy", "actions", f"{i}.npy")
        ))
        encs.append(np.load(
            join(location, "data", "full_policy", "text_encoding", f"{i}.npy")
        ))
        with open(join(location, "data", "full_policy", "raw_text", f"{i}.ob"), "rb") as f:
            texts += pickle.load(f)
    print("Done           ")
    train(
        np.concatenate(inputs, axis=0),
        # np.concatenate(encs, axis=0),
        np.array(texts),
        np.concatenate(actions, axis=0),
        texts=True,
        epochs=30,
        batch_size=128,
        lr=5e-4,
        wd=1e-3,
    )

