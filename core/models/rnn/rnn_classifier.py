import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import time, gc
import pickle

import torch.utils.data
from torch.utils import data
from torch.nn import functional as F, Dropout
from keras.preprocessing import text, sequence


NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 220


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets, max_features, num_classes=2, dropout=0.1):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.num_classes = num_classes
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.1)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear_combine = nn.Linear(2 * DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.dropout = Dropout(dropout)
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, num_classes)

    def pass_through_pipeline(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))

        hidden = h_conc + h_conc_linear1

        return hidden

    def forward(self, x):
        qa = x[0]
        question = x[1]

        hidden_qa = self.pass_through_pipeline(qa)
        hidden_question = self.pass_through_pipeline(question)

        combine = torch.cat((hidden_qa, hidden_question), 1)

        result = self.linear_combine(combine)
        result = self.dropout(result)
        result = F.relu(result)
        result = self.linear_out(result)
        return result


def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class TextDataset(data.Dataset):
    def __init__(self, text, lens, y=None):
        self.text = text
        self.lens = lens
        self.y = y

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, idx):
        if self.y is None:
            return self.text[idx], self.lens[idx]
        return self.text[0][idx], self.text[1][idx], self.lens[idx], self.y[idx]


class Collator(object):
    def __init__(self, test=False, percentile=100):
        self.test = test
        self.percentile = percentile

    def __call__(self, batch):
        global MAX_LEN

        batch_zip = list(zip(*batch))
        if self.test:
            lens = batch_zip[-1]
            texts = batch_zip[:-1]
        else:
            target = batch_zip[-1]
            lens = batch_zip[-2]
            texts = batch_zip[:-2]

        text_input = []
        lens_aux = np.array([l[0] for l in lens]) + np.array([l[1] for l in lens])
        max_len = min(int(np.percentile(lens_aux, self.percentile)), MAX_LEN)
        for i in range(len(texts)):
            text_input.append(sequence.pad_sequences(texts[i], maxlen=max_len))

        text_input = torch.Tensor(text_input).long().cpu()
        if self.test:
            return text_input

        return text_input, torch.Tensor([t.numpy() for t in target]).cpu()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_model(model, train_loader, test_loader, loss_fn, lr=0.01, lr_gamma =0.8, lr_step=3,
                n_epochs=4, title="", version="ULPC", debug=False, path=""):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_gamma ** (epoch // lr_step))
    best_f1 = 0
    best_report = ""
    best_f1_list = []
    for epoch in range(n_epochs):
        start_time = time.time()

        scheduler.step()

        model.train()
        avg_loss = 0.
        pred = []
        target = []
        if debug:
            print(f"{optimizer.param_groups[0]['lr']} LR")
        for step, (seq_batch, y_batch) in enumerate(train_loader):
            seq_batch = seq_batch.to("cuda")
            y_batch = y_batch.to("cuda")
            y_pred = model(seq_batch)
            y_target = y_batch
            loss = loss_fn(y_pred, y_target)
            optimizer.zero_grad()
            loss.backward()

            target += torch.argmax(y_batch.detach(), 1)
            pred += torch.argmax(y_pred.detach(), 1)

            optimizer.step()
            avg_loss += loss.item()
        if debug:
            print("Train Accuracy = ", sum([1 if pred[i] == target[i] else 0 for i in range(len(target))]) / len(target))
        model.eval()
        pred = []
        target = []
        for step, (seq_batch, y_batch) in enumerate(test_loader):
            seq_batch = seq_batch.to("cuda")
            y_batch = y_batch.to("cuda")
            y_pred = model(seq_batch)
            target += torch.argmax(y_batch, 1)
            pred += torch.argmax(y_pred, 1)

        target = torch.Tensor(target)
        pred = torch.Tensor(pred)
        #
        f1 = f1_score(target, pred, average="weighted")
        raw_f1 = f1_score(target, pred, average=None)
        if debug:
            print("Test Accuracy = ", sum([1 if pred[i] == target[i] else 0 for i in range(len(target))]) / len(target))
            print("f1 = ", str(f1))

        if best_f1 < f1:
            best_f1 = f1
            best_report = classification_report(target, pred, digits=3)
            best_f1_list = (f1, raw_f1, epoch)
            pickle.dump(model, open(f"{path}best_nn_{title}_{version}.bin", "wb"))

        elapsed_time = time.time() - start_time
        if debug:
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss / len(train_loader), elapsed_time))
            print("=============")
    print(best_report)
    print(best_f1_list)


def test_model(model, test_loader, tokenizer=None):
        model.eval()
        pred = []
        target = []
        for step, (seq_batch, y_batch) in enumerate(test_loader):
            seq_batch = seq_batch.to("cuda")
            y_batch = y_batch.to("cuda")
            y_pred = model(seq_batch)

            if tokenizer is not None:
                seq_list = seq_batch.cpu().numpy().tolist()
                for i in range(len(seq_list[0])):
                    print(tokenizer.sequences_to_texts([seq_list[0][i]]))
                    print(tokenizer.sequences_to_texts([seq_list[1][i]]))
                    print(torch.argmax(y_pred, 1)[i])
                    print(y_batch[i])
                    print("=" * 30)
            target += torch.argmax(y_batch, 1)
            pred += torch.argmax(y_pred, 1)

        target = torch.Tensor(target)
        pred = torch.Tensor(pred)
        #
        f1 = f1_score(target, pred, average=None)
        print("f1 = ", str(f1), f1.mean())

        print(classification_report(target, pred, digits=3))
        print("\n".join(["\t".join([str(y) for y in x]) for x in confusion_matrix(target, pred)]))
