import pickle
import random
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from core.utils.utils import build_tokenizer, seed_everything
from core.models.rnn.rnn_classifier import Collator, \
    NeuralNet, train_model, test_model
from core.models.rnn.rnn_classifier import TextDataset


DATA_FOLDER = "../../data/"
RESULTS_FOLDER = "../../data/results/"

HPARAMS = {
    "msrp": {
        "train": {
            "lr": 0.009,
            "lr_gamma": 0.95,
            "lr_step": 1
        },
        "fine_tune": {
            "lr": 0.001,
            "lr_gamma": 0.8,
            "lr_step": 3
        },
        "transfer_learn": {
            "lr": 0.001,
            "lr_gamma": 0.8,
            "lr_step": 3
        }
    },
    "ULPC": {
        "train": {
            "lr": 0.01,
            "lr_gamma": 0.9,
            "lr_step": 1
        },
        "fine_tune": {
            "lr": 0.002,
            "lr_gamma": 0.9,
            "lr_step": 3
        },
        "transfer_learn": {
            "lr": 0.002,
            "lr_gamma": 0.9,
            "lr_step": 3
        }
    },
    "children": {
        "train": {
            "lr": 0.01,
            "lr_gamma": 0.8,
            "lr_step": 3
        },
        "fine_tune": {
            "lr": 0.001,
            "lr_gamma": 0.8,
            "lr_step": 3
        },
        "transfer_learn": {
            "lr": 0.001,
            "lr_gamma": 0.8,
            "lr_step": 3
        }
    }
}


def pre_process_input(df, metric_c, tokenizer, version="ULPC"):
    offset = 0 if version in ["msrp", "children"] else 1
    X = df.values[:, [0 + offset, 1 + offset]]
    y_c = df[metric_c].values.reshape(-1, 1)

    if version == "ULPC" or version == "msrp":
        X_train, X_test = X[df['trn_test_val'] == 1], X[df['trn_test_val'] == 3]
        y_c_train, y_c_test = y_c[df['trn_test_val'] == 1].astype(float), y_c[df['trn_test_val'] == 3].astype(
            float)
    elif version == "children":
        sources = df["Source"].unique().tolist()
        msk = np.random.rand(len(sources)) < 0.5
        train_sources = [sources[i] for i in range(len(sources)) if msk[i]]
        test_sources = [sources[i] for i in range(len(sources)) if not msk[i]]
        if len([line for line in X if line[0] in train_sources]) < len(
                [line for line in X if line[0] not in train_sources]):
            aux = train_sources
            train_sources = test_sources
            test_sources = aux

        train_msk = [X[id][0] in train_sources for id in range(len(X))]
        test_msk = [X[id][0] in test_sources for id in range(len(X))]

        X_test = X[test_msk]
        X_train = X[train_msk]
        y_c_test = y_c[test_msk].astype(float)
        y_c_train = y_c[train_msk].astype(float)
    else:
        X_train, X_test = X, X[df['trn_test_val'] == 3]
        y_c_train, y_c_test = y_c.astype(float), y_c[df['trn_test_val'] == 3].astype(float)
    len_pos = len(y_c_train[y_c_train > 0])
    len_neg = len(y_c_train[y_c_train == 0])
    X_train_neg = X_train[(y_c_train == 0)[:, 0]]
    y_c_train_neg = y_c_train[(y_c_train == 0)[:, 0]]

    if len_neg < len_pos:
        while len_neg < len_pos:
            sample_id = random.randint(0, len(X_train_neg) - 1)
            X_train = np.append(X_train, [X_train_neg[sample_id, :]], axis=0)
            y_c_train = np.append(y_c_train, [y_c_train_neg[sample_id, :]], axis=0)
            len_neg = len(X_train[(y_c_train == 0)[:, 0]])
    else:
        X_train_pos = X_train[(y_c_train == 1)[:, 0]]
        y_c_train_pos = y_c_train[(y_c_train == 1)[:, 0]]
        while len_neg > len_pos:
            sample_id = random.randint(0, len(X_train_pos) - 1)
            X_train = np.append(X_train, [X_train_pos[sample_id, :]], axis=0)
            y_c_train = np.append(y_c_train, [y_c_train_pos[sample_id, :]], axis=0)
            len_pos = len(X_train[(y_c_train == 1)[:, 0]])

    # x_train = [tokenizer.texts_to_sequences([x[i] for x in X_train]) for i in range(1, 3)]
    # x_test = [tokenizer.texts_to_sequences([x[i] for x in X_test]) for i in range(1, 3)]

    x_train = [tokenizer.texts_to_sequences([x[i] for x in X_train]) for i in range(2)]
    x_test = [tokenizer.texts_to_sequences([x[i] for x in X_test]) for i in range(2)]

    return (x_train, y_c_train), (x_test, y_c_test)


def build_dataloaders(x_train, y_c_train, x_test, y_c_test, num_classes, batch_size=256):
    classes = torch.arange(num_classes).unsqueeze(0)

    x_train_lens = [(len(x_train[0][j]), len(x_train[1][j])) for j in range(len(x_train[0]))]
    x_test_lens = [(len(x_test[0][j]), len(x_test[1][j])) for j in range(len(x_test[0]))]
    y_c_train = torch.Tensor(np.array([x for x in y_c_train]))
    y_c_test = torch.Tensor(np.array([x for x in y_c_test]))
    y_c_train = (y_c_train == classes)
    y_c_test = (y_c_test == classes)

    train_collate = Collator(percentile=96)
    train_dataset = TextDataset(x_train, x_train_lens, y_c_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate)

    test_collate = Collator(test=False)
    test_dataset = TextDataset(x_test, x_test_lens, y_c_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_collate)
    return train_loader, test_loader


def train(version="children", initial_version="msrp", mode="train"):
    if version == "ULPC":
        df = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_train_data.csv', encoding='latin1', sep='\t')
        X = df.values[:, [1, 2]]
        metric_list = [
            ["Semantic_completeness", "Semantic_completeness_bin"],
            ["Paraphrase_quality", "Paraphrase_quality_tri"],
            ["Paraphrase_quality", "Paraphrase_quality_bin"],
            ["Syntactic_similarity", "Syntactic_similarity_bin"],
            ["Lexical_similarity", "Lexical_similarity_bin"]
        ]
    elif version == "msrp":
        df = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_msrp.csv', encoding='latin1', sep='\t')
        X = df.values[:, [0, 1]]
        metric_list = [
            ["Paraphrase_quality", "Paraphrase_quality_bin"],
        ]
    elif version == "children" or version == "adults":
        df = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_{version}.csv', encoding='latin1', sep='\t')
        df = df[df["Paraphrase_quality_tri"] != 9]
        df = df[df["Semantic_completeness_bin"] != 9]
        X = df.values[:, [0, 1]]
        metric_list = [
            ["Semantic_completeness", "Semantic_completeness_bin"],
            ["Paraphrase_quality", "Paraphrase_quality_tri"],
            ["Syntactic_similarity", "Syntactic_similarity_bin"],
            ["Lexical_similarity", "Lexical_similarity_bin"]
        ]
    tokenizer, glove_matrix = build_tokenizer(X.reshape(-1).tolist(), f"nn-{version}", False, generic_tokenizer=True)

    for metric_r, metric_c in metric_list:
        print("==========================================")
        print(f"============ {metric_c} - {version} - {mode} - ({initial_version if mode == 'transfer_learn' else ''})")
        print("==========================================")

        num_classes = 3
        if metric_c.endswith("bin"):
            num_classes = 2

        (x_train, y_c_train), (x_test, y_c_test) = pre_process_input(df, metric_c, tokenizer, version)
        train_loader, test_loader = build_dataloaders(x_train, y_c_train, x_test, y_c_test, num_classes, batch_size=64)

        seed_everything(1234)

        if mode == "train":
            model = NeuralNet(glove_matrix, y_c_train.shape[-1] - 1, len(tokenizer.word_index) + 1,
                              num_classes=num_classes)
        elif mode == "fine_tune":
            model = pickle.load(open(f"{RESULTS_FOLDER}best_nn_{metric_c}_{initial_version}.bin", "rb"))
        elif mode == "transfer_learn":
            load_metric = "Paraphrase_quality_bin" if initial_version != "children" else "Paraphrase_quality_tri"
            model_old = pickle.load(open(f"{RESULTS_FOLDER}best_nn_{load_metric}_{initial_version}.bin", "rb"))
            old_state_dict = model_old.state_dict()
            del old_state_dict['linear_out.weight']
            del old_state_dict['linear_out.bias']
            model = NeuralNet(glove_matrix, y_c_train.shape[-1] - 1, len(tokenizer.word_index) + 1,
                              num_classes=num_classes)
            model.load_state_dict(old_state_dict, strict=False)
        model.cuda()

        if mode != "train":
            print("Initial validation on train.")
            test_model(model, train_loader)
            print("Initial validation on test")
            test_model(model, test_loader)

        hp = HPARAMS[version][mode]
        title = metric_c
        if mode == "fine_tune":
            title = f"ft_{title}"
        if mode == "transfer_learn":
            title = f"tl_{title}"
        working_version = version if mode != "transfer_learn" else f"{initial_version}_to_{version}"
        test_model(model, test_loader, tokenizer)
        train_model(model, train_loader, test_loader, n_epochs=40, lr=hp["lr"], lr_gamma=hp["lr_gamma"],
                    lr_step=hp["lr_step"], loss_fn=nn.BCEWithLogitsLoss(reduction='mean'), title=title,
                    version=working_version, path=RESULTS_FOLDER)


def pretrain_generic_tokenizer():
    df_ulpc = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_train_data.csv', encoding='latin1', sep='\t')
    X_ulpc = df_ulpc.values[:, [1, 2]]

    df_msrp = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_msrp.csv', encoding='latin1', sep='\t')
    X_msrp = df_msrp.values[:, [0, 1]]

    df_children = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_children.csv', encoding='latin1', sep='\t')
    X_children = df_children.values[:, [0, 1]]

    X = X_ulpc.reshape(-1).tolist() + X_msrp.reshape(-1).tolist() + X_children.reshape(-1).tolist()
    _, _ = build_tokenizer(X, "", True, generic_tokenizer=True)


if __name__ == '__main__':
    pretrain_generic_tokenizer()
    train("ULPC", mode="train")
    train("children", initial_version="ULPC", mode="fine_tune")
    # train("msrp", mode="train")
    # train("children", mode="train")
    # train("children", initial_version="ULPC", mode="transfer_learn")
    # train("ULPC", initial_version="children", mode="transfer_learn")
    # train("children", initial_version="msrp", mode="transfer_learn")