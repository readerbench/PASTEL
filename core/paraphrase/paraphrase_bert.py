import pickle
import random

from sklearn.metrics import f1_score, classification_report
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer, AdamW
import torch
import numpy as np
import pandas as pd
from torch import nn

from core.models.bert.classifier import BERTClassifier
from core.models.bert.p_dataset import create_data_loader
from core.paraphrase.utils import seed_everything


DATA_FOLDER = "../../data/"
RESULTS_FOLDER = "../../data/results/"


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    i = 0
    agg_loss = 0
    for d in data_loader:
        i += 1
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        agg_loss += loss

    return correct_predictions.double() / n_examples, np.mean(losses)

def get_exp_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, step_size, gamma=0.9, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.01, gamma ** ((current_step - num_warmup_steps) // step_size))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def resample(df_train, label):
    len_pos_orig = len_pos = len(df_train[df_train[label] > 0])
    len_neg_orig = len_neg = len(df_train[df_train[label] == 0])
    df_train_neg = df_train[df_train[label] == 0]
    df_train_pos = df_train[df_train[label] > 0]
    # print(len_neg, len_pos)
    seen = []

    seed_everything(1234)
    if len_neg < len_pos:
        while len_neg < len_pos:
            # print(len_neg, len_pos)
            sample_id = random.randint(0, len(df_train_neg) - 1)
            if sample_id in seen:

                continue
            seen.append(sample_id)
            if len_neg_orig == len(seen):
                seen = []
            df_train = df_train.append(df_train_neg.iloc[sample_id])
            len_neg = len(df_train[(df_train[label] == 0)])
    else:
        while len_neg > len_pos:
            sample_id = random.randint(0, len(df_train_pos) - 1)
            if sample_id in seen:
                continue
            seen.append(sample_id)
            if len_pos_orig == len(seen):
                seen = []
            df_train = df_train.append([df_train_pos.iloc[sample_id]])
            len_pos = len(df_train[(df_train[label] > 0)])
    return df_train


def train_model(model, train_data_loader, test_data_loader, device, num_examples_train=1, num_examples_test=1,
                epochs=40, num_classes=2, metric="", version="ULPC", debug=False):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and n.find("bert") != -1],
         'weight_decay': 0.0001, 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find("bert") != -1],
         'weight_decay': 0.0, 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and n.find("bert") == -1],
         'weight_decay': 0.0001},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find("bert") == -1],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-2)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_exp_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0 * len(train_data_loader),
        num_training_steps=total_steps,
        step_size=len(train_data_loader) * 2,
        gamma=0.9 + (num_classes - 2) * 0.05
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    best_accuracy = 0
    best_split = None
    best_epoch = 0
    best_report = ""
    for epoch in range(epochs):
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler,
                                            num_examples_train)

        val_acc, val_loss, (val_f1, raw_f1, report) = eval_model(model, test_data_loader, loss_fn, device, num_examples_test,)

        if debug:
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            print(f'Train loss {train_loss} accuracy {train_acc}')
            print(f'Val loss {val_loss} accuracy {val_acc} f1 {val_f1}')

        if val_f1 > best_accuracy:
            torch.save(model, f'{RESULTS_FOLDER}best_bert_{metric}_{version}.bin')
            best_accuracy = val_f1
            best_split = raw_f1
            best_epoch = epoch
            best_report = report
        if val_f1 == 1:
            print("Quick break")
            break
    print(f"Best f1 {best_accuracy} {best_split}, epoch: {best_epoch}")
    print(best_report)
    return model

def read_datasets(dataset):
    if dataset == "ULPC":
        df = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_train_data.csv', encoding='latin1', sep='\t')
        targets_list = [
            ["Paraphrase_quality", "Paraphrase_quality_tri"],
            ["Paraphrase_quality", "Paraphrase_quality_bin"],
            ["Semantic_completeness", "Semantic_completeness_bin"],
            ["Syntactic_similarity", "Syntactic_similarity_bin"],
            ["Lexical_similarity", "Lexical_similarity_bin"],
        ]

        df_train = df[df['trn_test_val'] == 1]
        df_test = df[df['trn_test_val'] == 3]
    elif dataset == "msrp":
        df = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_msrp.csv', encoding='latin1', sep='\t')
        targets_list = [
            ["Paraphrase_quality_bin", "Paraphrase_quality_bin"],
        ]
        df_train = df[df['trn_test_val'] == 1]
        df_test = df[df['trn_test_val'] == 3]
    elif dataset == "children" or dataset == "adults":
        df = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_{dataset}.csv', encoding='latin1', sep='\t')
        df = df[df["Paraphrase_quality_tri"] != 9]
        df = df[df["Semantic_completeness_bin"] != 9]
        targets_list = [
            ["Paraphrase_quality", "Paraphrase_quality_tri"],
            # ["Paraphrase_quality", "Paraphrase_quality_bin"],
            ["Semantic_completeness", "Semantic_completeness_bin"],
            ["Syntactic_similarity", "Syntactic_similarity_bin"],
            ["Lexical_similarity", "Lexical_similarity_bin"],
        ]
        sources = df["Source"].unique().tolist()
        msk = np.random.rand(len(sources)) < 0.4
        train_sources = [sources[i] for i in range(len(sources)) if msk[i]]
        test_sources = [sources[i] for i in range(len(sources)) if not msk[i]]
        if len(df[df["Source"].str.contains("|".join(train_sources))]) < len(
                df[df["Source"].str.contains("|".join(test_sources))]):
            aux = train_sources
            train_sources = test_sources
            test_sources = aux

        df_train = df[df["Source"].str.contains("|".join(train_sources))]
        df_test = df[df["Source"].str.contains("|".join(test_sources))]

        # df_train = df[df['trn_test_val'] == 1]
        # df_test = df[df['trn_test_val'] == 3]
    return df_train, df_test, targets_list

def train(version="ULPC", initial_version="msrp", mode="train"):
    seed_everything(1234)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df_train, df_test, targets_list = read_datasets(version)
    df_train_orig = df_train
    df_test_orig = df_test
    for targets in targets_list:
        print("==========================================")
        print(f"============ {targets[1]} - {version} - {mode} - ({initial_version if mode == 'transfer_learn' else ''})")
        print("==========================================")
        df_train = df_train_orig[["Source", "Production", targets[1]]]
        df_train["bin"] = df_train[targets[1]]
        df_test = df_test_orig[["Source", "Production", targets[1]]]
        df_test["bin"] = df_test[targets[1]]

        df_train_new = resample(df_train, targets[1])
        PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        MAX_LEN_P = 75
        BATCH_SIZE = 32
        train_data_loader = create_data_loader(df_train_new, tokenizer, MAX_LEN_P, BATCH_SIZE, targets[1])
        test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, targets[1])

        num_classes = 2 if targets[1].endswith("bin") else 3
        if mode == "train":
            model = BERTClassifier(num_classes, PRE_TRAINED_MODEL_NAME)
        elif mode == "fine_tune":
            model = torch.load(f"{RESULTS_FOLDER}best_bert_{targets[1]}_{initial_version}.bin")
        elif mode == "transfer_learn":
            load_metric = "Paraphrase_quality_bin" if initial_version != "children" else "Paraphrase_quality_tri"
            if initial_version.find("to"):
                old_model = torch.load(f"{RESULTS_FOLDER}best_bert_tl_{load_metric}_{initial_version}.bin")
            else:
                old_model = torch.load(f"{RESULTS_FOLDER}best_bert_{load_metric}_{initial_version}.bin")

            old_state_dict = old_model.state_dict()
            del old_state_dict['out.weight']
            del old_state_dict['out.bias']
            model = BERTClassifier(num_classes, PRE_TRAINED_MODEL_NAME)
            model.load_state_dict(old_state_dict, strict=False)
        model = model.to(device)

        title = targets[1]
        if mode == "fine_tune":
            title = f"ft_{title}"
        if mode == "transfer_learn":
            title = f"tl_{title}"
        working_version = version if mode != "transfer_learn" else f"{initial_version}_to_{version}"
        train_model(model, train_data_loader, test_data_loader, device,
                    num_examples_train=len(df_train_new),
                    num_examples_test=len(df_test),
                    epochs=20, num_classes=num_classes, metric=title, version=working_version, debug=True)


if __name__ == '__main__':
    train("msrp", mode="train")
    train("ULPC", initial_version="msrp", mode="transfer_learn")
    train("children", initial_version="msrp_to_ULPC", mode="transfer_learn")