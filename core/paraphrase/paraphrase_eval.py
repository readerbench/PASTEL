import torch
from sklearn.metrics import f1_score, classification_report
import numpy as np
import pandas as pd
from torch import nn

from transformers import BertTokenizer
from core.models.bert.classifier import BERTClassifier

from core.models.bert.p_dataset import create_data_loader, encode_paraphrase_pair

from core.paraphrase.paraphrase_bert import read_datasets


MAX_LEN_P = 75

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    global_preds = []
    global_targets = []
    res_list = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            for i in range(len(d["text_p"])):
                res_list.append((d["text_s"][i], d["text_p"][i], preds[i].item(), d["item"][i].item()))
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            global_preds.append(preds)
            global_targets.append(targets)
            losses.append(loss.item())

    global_preds = torch.cat(global_preds).cpu()
    global_targets = torch.cat(global_targets).cpu()
    f1 = f1_score(global_targets, global_preds, average="weighted")

    best_report = classification_report(global_targets, global_preds, digits=3)
    return correct_predictions.double() / n_examples, np.mean(losses), (f1, f1_score(global_targets, global_preds, average=None), best_report)


def validate_model_on_dataset(model_path, dataset):
    df_train, df_test, targets_list = read_datasets(dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df_test_orig = df_test
    for targets in targets_list:
        df_test = df_test_orig[["Source", "Production", targets[1]]]

        df_test["bin"] = df_test[targets[1]]
        PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, 1, targets[1])
        model = torch.load(model_path)

        _, _, (val_f1, raw_f1, report) = eval_model(model, test_data_loader, nn.CrossEntropyLoss().to(device), device,
                                                    len(df_test))
        print(f"Validation on test. {val_f1} - {raw_f1}")

def validate_model_on_str_input(model_path, paraphrase_pairs_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = torch.load(model_path)

    for (source, production) in paraphrase_pairs_list:
        input_ids, att_mask = encode_paraphrase_pair(source, production, tokenizer, MAX_LEN_P)
        input_ids = input_ids.reshape(1, -1).to(device)
        attention_mask = att_mask.reshape(1, -1).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        print(f"source: {source}\nparaphrase: {production}\nassessment: {preds[0].item()}\n")

if __name__ == '__main__':
    examples = [
        # contains 2-element list of source, paraphrase pairs
        ["Scientists can tell stars apart by their color.", "Colors help scientists tell stars apart."],
        ["Scientists can tell stars apart by their color.", "Scientists know which stars are which by their colors."],
        ["Scientists can tell stars apart by their color.", "Scientists use their color by the stars apart by their color"],
        ["Scientists can tell stars apart by their color.",
         "Protostars are not stars yet. They are baby stars, just like humans they grow."]
    ]
    model_path = "/home/bogdan/projects/PASTEL/data/results/best_bert_tl_Paraphrase_quality_tri_msrp_to_ULPC_to_children.bin"
    validate_model_on_str_input(model_path, examples)