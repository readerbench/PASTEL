import os.path

import pytorch_lightning as pl
from transformers import BertTokenizer
import torch
import transformers
from core.models.bert.mtl import BERTMTL
from core.data_processing.se_dataset import SelfExplanations, create_data_loader
from sklearn.model_selection import train_test_split

transformers.logging.set_verbosity_error()

def get_train_test_IDs(IDs):
    ID_file = "../data/mtl/se_ID_file.txt"

    if os.path.exists(ID_file):
        train_line, test_line = open(ID_file, "r").readlines()
        train_IDs = train_line.split("\t")
        test_IDs = test_line.split("\t")
    else:
        train_IDs, test_IDs = train_test_split(IDs, test_size=0.2, random_state=42)
        f = open(ID_file, "w")
        f.write("\t".join(train_IDs))
        f.write("\n")
        f.write("\t".join(test_IDs))
        f.close()

    return train_IDs, test_IDs

def map_train_test(x):
    if x['Dataset'] in ['NIU 1', 'ASU 5']:
        return 'train'
    if x['Dataset'] == 'NIU 3':
        return 'dev'
    if x['Dataset'] == 'ASU 1':
        if x['PrePost'] == 'post':
            return 'train'
        return 'dev'
    if x['Dataset'] == 'ASU 4' and not str(x['ID']).startswith('ISTARTREF'):
        return 'train'
    print(x['Dataset'], x['ID'])
    return 'test'

def get_new_train_test_split(df):
    df['EntryType'] = df.apply(lambda x: map_train_test(x), axis=1)
    df = df[df[SelfExplanations.OVERALL] > 0]
    # df[SelfExplanations.OVERALL] -= 1
    return df[df['EntryType'] == 'train'], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']

if __name__ == '__main__':
    num_tasks = 7
    predefined_version = ""
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    MAX_LEN_P = 80
    BATCH_SIZE = 64
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self_explanations = SelfExplanations()
    # target_sent_enhanced = self_explanations.parse_se_from_csv(
    #     "../data/results/results_paraphrase_se_aggregated_dataset_v2.csv")
    target_sent_enhanced = self_explanations.parse_se_from_csv(
        "/home/bogdan/projects/PASTEL/data/results/results_paraphrase_se_aggregated_dataset_2.csv")

    text_list = self_explanations.df['TextID'].unique().tolist()
    # text_list = ['ASU 1 - HD', 'ASU 1 - RBC', 'ASU 4 - NS 1', 'ASU 5 - CD', 'ASU 5 - HD', 'CRaK - CD', 'NIU 1 - 320', 'NIU 1 - 321', 'NIU 1 - 337', 'NIU 1 - 338 ']
    dev_mask = self_explanations.df['TextID'].isin(text_list[:2])
    test_mask = self_explanations.df['TextID'].isin(text_list[2:3])
    train_mask = self_explanations.df['TextID'].isin(text_list[3:])

    # old split
    df_train = self_explanations.df[train_mask]
    df_dev = self_explanations.df[dev_mask]
    df_test = self_explanations.df[test_mask]

    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df)
    # random deterministic split
    # IDs = self_explanations.df['ID'].unique().tolist()
    # train_IDs, test_IDs = get_train_test_IDs(IDs)
    # df_train = self_explanations.df[self_explanations.df['ID'].isin(train_IDs)]
    # df_test = self_explanations.df[self_explanations.df['ID'].isin(test_IDs)]

    # toggle 0 or 1 for using rb_features
    rb_feats = 0
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=False)
    val_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=False)
    rb_feats = 0#train_data_loader.dataset.rb_feats.shape[1]
    model = BERTMTL(num_tasks, PRE_TRAINED_MODEL_NAME, rb_feats=rb_feats)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # limit_train_batches=100,
        max_epochs=50)
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)