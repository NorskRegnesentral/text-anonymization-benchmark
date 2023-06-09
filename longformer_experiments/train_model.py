from typing_extensions import TypedDict
import torch.nn.functional as F
from typing import List,Any
from transformers import LongformerTokenizerFast
from tokenizers import Encoding
import itertools
from torch import nn
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import json
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from data_handling import *
from longformer_model import Model
from data_manipulation import training_raw, dev_raw, test_raw
import collections
import random
import argparse

if __name__ == "__main__":

    bert = "allenai/longformer-base-4096"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    label_set = LabelSet(labels=["MASK"])

    training = Dataset(data=training_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=4096)
    dev = Dataset(data=dev_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=4096)
    test = Dataset(data=test_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=4096)

    trainloader = DataLoader(training, collate_fn=TrainingBatch,batch_size=1, shuffle=True)
    devloader = DataLoader(dev, collate_fn=TrainingBatch, batch_size=1, shuffle=True)
    testloader = DataLoader(test, collate_fn=TrainingBatch, batch_size=1)

    model = Model(model = bert, num_labels = len(training.label_set.ids_to_label.values()))
    model = model.to(device)

    if device == 'cuda':
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]).cuda())
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]))

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    total_val_loss = 0
    total_train_loss, epochs = [], []
    for epoch in range(2):
        epochs.append(epoch)
        model.train()
        for X in tqdm.tqdm(trainloader):
            y = X['labels']
            optimizer.zero_grad()
            y_pred = model(X)
            y_pred = y_pred.permute(0,2,1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        total_train_loss.append(loss.item())
        print('Epoch: ', epoch + 1)
        print('Training loss: {0:.2f}'.format(loss.item()))

    predictions , true_labels, offsets = [], [], []
    inputs, test_pred , test_true, offsets = [], [], [], []
    for X in tqdm.tqdm(devloader):
        model.eval()
        with torch.no_grad():
            y = X['labels']
            y_pred = model(X)
            y_pred = y_pred.permute(0,2,1)
            val_loss = criterion(y_pred, y)
            pred = y_pred.argmax(dim=1).cpu().numpy()
            true = y.cpu().numpy()
            offsets.extend(X['offsets'])
            predictions.extend([list(p) for p in pred])
            true_labels.extend(list(p) for p in true)
            total_val_loss += val_loss.item()

    avg_loss = total_val_loss / len(devloader)
    print('Validation loss: {0:.2f}'.format(avg_loss))

    out = []
    ## Getting entity level predictions
    for i in range(len(offsets)):
        if -1 in offsets[i]:
            count = offsets[i].count(-1)
            offsets[i] = offsets[i][:(len(offsets[i])-count)] # Remove the padding, each i is a different batch
            predictions[i] = predictions[i][:len(offsets[i])] # Remove the padding, [CLS] ... [SEP] [PAD] [PAD]...

    l1 = [item for sublist in predictions for item in sublist] # Unravel predictions if it has multiple batches
    l2 = [item for sublist in offsets for item in sublist] # Unravel subsets if it has multiple batches

    it = enumerate(l1+[0])
    sv = 0

    ## Uses the sequences of 1s and 2s in the predictions in combination with the token offsets to return the entity level start and end offset.
    try:
        while True:
            if sv==1: # If an entity followed by another entity, fi, fv marks the beginning of an entity
                fi,fv = si,sv
            else:
                while True:
                    fi,fv = next(it)
                    if fv:
                        break
            while True: # Whenever it finds an 1, it tries to find the boundary for this entity (stops at 0 or 1)
                si,sv = next(it)
                if sv == 0 or sv == 1:
                    break
            out.append((l2[fi][0],l2[fi][1],l2[si-1][2]))

    except StopIteration:
        pass

    d = {}
    for i in out: # save the updated out {id: [(start,end)}
        if i[0] not in d:
            d[i[0]] = []
            d[i[0]].append((i[1],i[2]))
        else:
            d[i[0]].append((i[1],i[2]))

    ##Filter
    out_dev = {}
    for i in d:
        out_dev[i] = []
        d[i] = list(map(list, OrderedDict.fromkeys(map(tuple, d[i])).keys()))
        out_dev[i] = d[i]

    f = open("preds_dev.json", "w")
    json.dump(out_dev, f)
    f.close()                               

    predictions , true_labels, offsets = [], [], []
    model.eval()
    for X in tqdm.tqdm(testloader):
        with torch.no_grad():
            y = X['labels']
            y_pred = model(X)
            y_pred = y_pred.permute(0,2,1)
            pred = y_pred.argmax(dim=1).cpu().numpy()
            true = y.cpu().numpy()
            offsets.extend(X['offsets'])
            predictions.extend([list(p) for p in pred])
            true_labels.extend(list(p) for p in true)

    out = []
    for i in range(len(offsets)):
        if -1 in offsets[i]:
            count = offsets[i].count(-1)
            offsets[i] = offsets[i][:(len(offsets[i])-count)]
            predictions[i] = predictions[i][:len(offsets[i])]

    l1 = [item for sublist in predictions for item in sublist]
    l2 = [item for sublist in offsets for item in sublist]

    it = enumerate(l1+[0])
    sv = 0

    try:
        while True:
            if sv==1:
                fi,fv = si,sv
            else:
                while True:
                    fi,fv = next(it)
                    if fv:
                        break
            while True:
                si,sv = next(it)
                if sv == 0 or sv == 1:
                    break
            out.append((l2[fi][0],l2[fi][1],l2[si-1][2]))

    except StopIteration:
        pass

    d = {}
    for i in out:
        if i[0] not in d:
            d[i[0]] = []
            d[i[0]].append((i[1],i[2]))
        else:
            d[i[0]].append((i[1],i[2]))

    ##Filter
    out_test = {}
    for i in d:
        out_test[i] = []
        d[i] = list(map(list, OrderedDict.fromkeys(map(tuple, d[i])).keys()))
        out_test[i] = d[i]

    f = open("preds_test.json", "w")
    json.dump(out_test, f)
    f.close()


    PATH = "long_model.pt"
    torch.save(model.state_dict(), PATH)