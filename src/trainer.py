import torch
import time
import os
import datasets
import yaml
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

import config_env
import data
from data import Corpus
from model import RNN_nlp


def train(model, train_data, criterion, optimize, num_epochs=5, **train_params):
    batch_size = train_params.get("batch_size", train_data.size(1))
    n_token = train_params.get("n_token", model.decoder.out_features)
    stride = train_params.get("stride", 32)
    log_interval = train_params.get("log_interval", int(int(len(train_data) / stride) / 10))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")
    model = model.to(device)

    df_log = pd.DataFrame()

    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        hidden = model.init_hidden(bsz=batch_size)

        for batch, i in enumerate(
            tqdm(
                range(0, train_data.size(0), stride),
                desc=f"Epoch {epoch}|",
                unit=" batchs",
            )
        ):
            inputs, target = data.get_batch(data=train_data, i=i, stride=stride)
            inputs, target = inputs.to(device=device), target.to(device=device)

            model.zero_grad()
            hidden = hidden.detach()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, n_token), target.view(-1))
            df_log = pd.concat(
                [
                    df_log,
                    pd.DataFrame(
                        {
                            "epoch": [epoch],
                            "batch": [batch],
                            "loss": [loss.item()],
                            "ppl": [torch.exp(loss).item()]
                        }
                    )
                ]
            )
            if batch % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | Batch {batch}/{int(len(train_data) / stride)} | "
                    f"ms/batch: {elapsed:.2f} | loss: {loss.item():.4f} | ppl: {torch.exp(loss).item():.4f}"
                )
                start_time = time.time()  # Reset timer after logging

            loss.backward()
            optimize.step()
            optimize.zero_grad()
    
    if not os.path.exists(os.path.join(config_env.ROOT_PATH, "result/rnn")):
        os.makedirs(os.path.join(config_env.ROOT_PATH, "result/rnn"))
    df_log.to_csv(os.path.join(config_env.ROOT_PATH, "result/rnn/log.csv"))

def run():
    # Download dataset
    if os.path.exists(os.path.join(config_env.ROOT_PATH, "data/wikitext-103/")):
        print("Data exited")
    else:
        print("Downloading data set ...")
        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        for type_ds in ["train", "validation", "test"]:
            lines = ds[type_ds]["text"]
            if not os.path.exists(
                os.path.join(config_env.ROOT_PATH, "data/wikitext-103")
            ):
                os.makedirs(os.path.join(config_env.ROOT_PATH, "data/wikitext-103"))
            with open(
                os.path.join(config_env.ROOT_PATH, f"data/wikitext-103/{type_ds}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                for line in tqdm(lines, desc=f"Saving {type_ds}.txt", unit=" lines"):
                    f.write(line)

    # Load dataset
    if os.path.exists(
        os.path.join(config_env.ROOT_PATH, "data/wikitext-103/corpus.pt")
    ):
        print("Loading corpus ...")
        corpus = torch.load(
            os.path.join(config_env.ROOT_PATH, "data/wikitext-103/corpus.pt")
        )
    else:
        print("Creating corpus ...")
        corpus = Corpus(path=os.path.join(config_env.ROOT_PATH, "data/wikitext-103"))
        torch.save(
            corpus, os.path.join(config_env.ROOT_PATH, "data/wikitext-103/corpus.pt")
        )

    print("Training model ...")
    with open(os.path.join(config_env.ROOT_PATH, "config/train_params.yaml"), "r") as f:
        train_params = yaml.safe_load(f)

    # Batchify
    train_data = data.batchify(data=corpus.train, batch_size=train_params.get("batch_size", 128))

    # Init model
    print("Initing model ...")
    with open(os.path.join(config_env.ROOT_PATH, "config/model_params.yaml"), "r") as f:
        model_params = yaml.safe_load(f)
    model_params["ntoken"] = corpus.vocab_size
    model = RNN_nlp(**model_params)

    train_params = {
        "criterion": nn.CrossEntropyLoss(),
        "optimize": torch.optim.SGD(model.parameters(), lr=float(train_params["lr"])),
        "n_token": corpus.vocab_size,
    }

    train(model=model, train_data=train_data, **train_params)