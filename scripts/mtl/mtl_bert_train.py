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
    ID_file = "/data/mtl/se_ID_file.txt"

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

if __name__ == '__main__':
    num_tasks = 7
    predefined_version = ""
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    MAX_LEN_P = 80
    BATCH_SIZE = 64
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self_explanations = SelfExplanations()
    target_sent_enhanced = self_explanations.parse_se_from_csv(
        "/home/bogdan/projects/self-explanations/git_resources/new_english_se2_enhanced.csv")

    IDs = self_explanations.df['ID'].unique().tolist()
    train_IDs, test_IDs = get_train_test_IDs(IDs)
    df_train = self_explanations.df[self_explanations.df['ID'].isin(train_IDs)]
    df_test = self_explanations.df[self_explanations.df['ID'].isin(test_IDs)]

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks)
    val_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks)
    model = BERTMTL(num_tasks, PRE_TRAINED_MODEL_NAME)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # limit_train_batches=100,
        max_epochs=50)
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)