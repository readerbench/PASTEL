import os.path

import pytorch_lightning as pl
from transformers import BertTokenizer, RobertaTokenizer
import torch
import transformers
from core.models.bert.mtl import BERTMTL
from core.data_processing.se_dataset import SelfExplanations, create_data_loader
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint

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
    if x['Dataset'] in ['ASU 5']:
        return 'test'
    if x['Dataset'] == 'CRaK':
        return 'train'
    if x['Dataset'] == 'ASU 1':
        print(x['PrePost'])
        if x['PrePost'] == 'Post':
            return 'dev'
        return 'train'
    if x['Dataset'] == 'ASU 4':
        print("ID", x['ID'])
        if not str(x['ID']).startswith('ISTARTREF'):
            return 'train'
        else:
            return 'dev'
    return 'dump'

def get_new_train_test_split(df):
    df['EntryType'] = df.apply(lambda x: map_train_test(x), axis=1)
    print(df['EntryType'].describe())
    df = df[(df[SelfExplanations.OVERALL] > 0) & (df[SelfExplanations.OVERALL] < 9)]
    df[SelfExplanations.OVERALL] -= 1
    return df[(df['EntryType'] == 'train') | (df['EntryType'] == 'dev')], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']
    # return df[(df['EntryType'] == 'train')], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']

def experiment(task_level_weights=[], bert_model="bert-base-cased", lr=1e-3, num_epochs=20, task_name="none"):
    if task_name == "none":
        num_tasks = 4
    else:
        num_tasks = 1
    predefined_version = ""
    MAX_LEN_P = 80
    BATCH_SIZE = 128
    if "roberta" in bert_model:
        tokenizer = RobertaTokenizer.from_pretrained(bert_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model)

    self_explanations = SelfExplanations()
    # target_sent_enhanced = self_explanations.parse_se_from_csv(
    #     "../data/results/results_paraphrase_se_aggregated_dataset_v2.csv")
    target_sent_enhanced = self_explanations.parse_se_from_csv(
        "../../data/results_paraphrase_se_aggregated_dataset_2.csv")


    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df)
    print(f"len train {len(df_train)}")
    print(f"len dev {len(df_dev)}")
    print(f"len test {len(df_test)}")
    # random deterministic split
    # IDs = self_explanations.df['ID'].unique().tolist()
    # train_IDs, test_IDs = get_train_test_IDs(IDs)
    # df_train = self_explanations.df[self_explanations.df['ID'].isin(train_IDs)]
    # df_test = self_explanations.df[self_explanations.df['ID'].isin(test_IDs)]

    # toggle 0 or 1 for using rb_features
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True, task_name=task_name)
    val_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True, task_name=task_name)
    rb_feats = train_data_loader.dataset.rb_feats.shape[1]
    task_weights = []
    if num_tasks > 1:
        task_names = [SelfExplanations.MTL_TARGETS[task_id] for task_id in range(num_tasks)]
        for task in range(num_tasks):
            df_aux = df_train[df_train[SelfExplanations.MTL_TARGETS[task]] < 9]
            values = df_aux[SelfExplanations.MTL_TARGETS[task]].value_counts()
            total = len(df_aux[SelfExplanations.MTL_TARGETS[task]]) * 1.0
            task_weights.append(torch.Tensor([total / values[i] if i in values else 0 for i in range(SelfExplanations.MTL_CLASS_DICT[SelfExplanations.MTL_TARGETS[task]])]))
    else:
        task_names = [task_name]
        df_aux = df_train[df_train[task_name] < 9]
        values = df_aux[task_name].value_counts()
        total = len(df_aux[task_name]) * 1.0
        task_weights.append(torch.Tensor([total / values[i] if i in values else 0 for i in range(SelfExplanations.MTL_CLASS_DICT[task_name])]))

    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_loss")
    model = BERTMTL(task_names, bert_model, rb_feats=rb_feats, task_weights=task_weights, task_level_weights=task_level_weights, lr=lr, num_epochs=num_epochs)

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[checkpoint_callback],
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # limit_train_batches=100,
        max_epochs=num_epochs)
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

if __name__ == '__main__':
    print("=" * 33)
    experiment([2, 2, 1, 5], bert_model="roberta-base", lr=2e-4, num_epochs=50, task_name="paraphrasepresence")
    print("=" * 33)
    experiment([2, 2, 1, 5], bert_model="roberta-base", lr=2e-4, num_epochs=50, task_name="bridgepresence")
    print("=" * 33)
    experiment([2, 2, 1, 5], bert_model="roberta-base", lr=2e-4, num_epochs=50, task_name="elaborationpresence")
    print("=" * 33)
    experiment([2, 2, 1, 5], bert_model="roberta-base", lr=2e-4, num_epochs=50, task_name="overall")
    print("=" * 33)
    experiment([2, 2, 1, 5], bert_model="roberta-base", lr=2e-4, num_epochs=50)
    #experiment([2, 2, 1, 5], bert_model="roberta-base", lr=5e-4)
    print("=" * 33)
    #experiment([2, 2, 1, 5], bert_model="roberta-base", lr=8e-4)
    print("=" * 33)
    #experiment([2, 2, 1, 5], bert_model="roberta-base", lr=1e-3)
    print("=" * 33)
    # experiment([2, 2, 1, 5], bert_model="bert-base-cased")
