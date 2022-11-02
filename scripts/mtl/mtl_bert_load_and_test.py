from transformers import BertTokenizer

from core.data_processing.se_dataset import SelfExplanations, create_data_loader
from core.models.bert.mtl import BERTMTL
from scripts.mtl.mtl_bert_train import get_train_test_IDs, get_new_train_test_split
import pytorch_lightning as pl
import torch

if __name__ == '__main__':
    num_tasks = 7
    predefined_version = ""
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    MAX_LEN_P = 80
    BATCH_SIZE = 16
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self_explanations = SelfExplanations()
    target_sent_enhanced = self_explanations.parse_se_from_csv(
        "../../data/results_paraphrase_se_aggregated_dataset_2.csv")

    # IDs = self_explanations.df['ID'].unique().tolist()
    # _, test_IDs = get_train_test_IDs(IDs)
    # df_test = self_explanations.df[self_explanations.df['ID'].isin(test_IDs)]
    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df)

    val_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True)
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True)
    model = BERTMTL(num_tasks, PRE_TRAINED_MODEL_NAME, rb_feats=0)#val_data_loader.dataset.rb_feats.shape[1])
    model = model.load_from_checkpoint("./lightning_logs/version_4/checkpoints/epoch=49-step=2600.ckpt",
                                       num_tasks=num_tasks,
                                       pretrained_bert_model=PRE_TRAINED_MODEL_NAME,
                                       rb_feats=val_data_loader.dataset.rb_feats.shape[1])
    trainer = pl.Trainer(

        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # limit_train_batches=100,
        max_epochs=50)
    trainer.test(model, dataloaders=val_data_loader)
    trainer.test(model, dataloaders=train_data_loader)