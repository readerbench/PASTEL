import numpy as np
import torch
import pandas
import re

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from os.path import exists

def clear_text(text: str):
  return re.sub('[^a-zA-Z0-9 \n.]', '', str(text))


def get_matching_sentence(df_target_sentences, dataset_row, sentence_index):
  return df_target_sentences.loc[
    # TEXT_ID is a string type (e.g. NIU 1 - 338) and needs to be stripped to overcome any cleanup dataset issues
    (df_target_sentences[SelfExplanations.TEXT_ID].str.strip() == dataset_row[SelfExplanations.TEXT_ID].strip()) &
    (df_target_sentences[SelfExplanations.SENT_NO] == dataset_row[SelfExplanations.SENT_NO])
    ][sentence_index].values[0]


class SelfExplanations:
  TEXT_ID = "TextID"
  SENT_NO = "SentNo"

  SE = "SelfExplanation"
  TARGET_SENTENCE = "TargetSentence"
  PREVIOUS_SENTENCE = "PreviousSentence"
  TOO_SHORT = "tooshort"
  NON_SENSE = "nonsense"
  IRRELEVANT = "irrelevant"
  COPY_PASTE = "copypaste"
  MISCONCEPTION = "misconception"
  MONITORING = "monitoring"
  PARAPHRASE = "paraphrasepresence"
  PR_LEXICAL_CHANGE = "lexicalchange"
  PR_SYNTACTIC_CHANGE = "syntacticchange"
  BRIDGING = "bridgepresence"
  BR_CONTRIBUTION = "bridgecontribution"
  ELABORATION = "elaborationpresence"
  EL_LIFE_EVENT = "lifeevent"
  OVERALL = "overall"

  MTL_TARGETS = [PARAPHRASE, PR_LEXICAL_CHANGE, PR_SYNTACTIC_CHANGE, BRIDGING,
                BR_CONTRIBUTION, ELABORATION, OVERALL]

  MTL_CLASS_DICT = {
    PARAPHRASE: 3,
    PR_LEXICAL_CHANGE: 2,
    PR_SYNTACTIC_CHANGE: 2,
    BRIDGING: 4,
    BR_CONTRIBUTION: 3,
    ELABORATION: 3,
    OVERALL: 4
  }

  def parse_se_scoring_from_csv(self, path_to_csv_file: str):
    df = pandas.read_csv(path_to_csv_file, delimiter=',', dtype={self.SENT_NO: "Int64"}).dropna(how='all')
    # print(df.sample(5))
    df[self.SE] = df[self.SE].map(clear_text)
    self.df = df

    enhanced_dataset_file = '/home/bogdan/projects/self-explanations/git_resources/new_english_se2_enhanced.csv'
    # compute target and previous sentences if necessary
    if not exists(enhanced_dataset_file):
      self.df_target_sentences = pandas.read_csv(
        "/home/bogdan/projects/self-explanations/git_resources/targetsentences.csv", delimiter=',',
        dtype={self.SENT_NO: "Int64"})
      self.df[self.TARGET_SENTENCE] = self.df.apply(
        lambda x: get_matching_sentence(self.df_target_sentences, x, SelfExplanations.TARGET_SENTENCE), axis=1)
      self.df[self.PREVIOUS_SENTENCE] = self.df.apply(
        lambda x: get_matching_sentence(self.df_target_sentences, x, SelfExplanations.PREVIOUS_SENTENCE), axis=1)
      self.df.to_csv(enhanced_dataset_file, index=False)

    return enhanced_dataset_file


  def parse_se_from_csv(self, path_to_csv_file: str):
    df = pandas.read_csv(path_to_csv_file, delimiter=',', dtype={self.SENT_NO: "Int64"}).dropna(how='all')
    self.df = df
    self.df['Production'] = self.df['SelfExplanation']
    self.df['Source'] = self.df['TargetSentence']
    for val in self.MTL_TARGETS:

      self.df[val][self.df[val] == 'BLANK '] = 9
      self.df[val][self.df[val] == 'BLANK'] = 9
      self.df[val][self.df[val] == 'blANK'] = 9
      self.df[val] = self.df[val].astype(int)

      # print(val, self.df[val].unique())
      # mask = self.df[val] != 9
      # print(self.df[val][mask].describe())
    return df


class SEDataset(Dataset):
  def __init__(self, source, production, targets, tokenizer, max_len, rb_feats=None):
    self.source = source
    self.production = production
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.rb_feats = rb_feats.astype(float) if rb_feats is not None else None

    self.targets[self.targets == 'BLANK '] = 9
    self.targets[self.targets == 'BLANK'] = 9
    self.targets[self.targets == 'blANK'] = 9
    self.targets = np.vectorize(int)(self.targets)

  def __len__(self):
    return len(self.source)

  def __getitem__(self, item):
    source = str(self.source[item])
    production = str(self.production[item])
    target = self.targets[item]
    rb_feats = self.rb_feats[item] if self.rb_feats is not None else []

    encoding = self.tokenizer.encode_plus(
      text=source,
      text_pair=production,
      truncation=True,
      truncation_strategy="longest_first",
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'text_s': source,
      'text_p': production,
      'rb_feats': rb_feats,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.LongTensor(target),
      'item': item
    }

def create_data_loader(df, tokenizer, max_len, batch_size, num_tasks, use_rb_feats=False):
  targets = SelfExplanations.MTL_TARGETS[:num_tasks]
  feats = df[df.columns[38:]].to_numpy() if use_rb_feats else None

  ds = SEDataset(
    source=df['Source'].to_numpy(),
    production=df['Production'].to_numpy(),
    rb_feats=feats,
    targets=np.array([df[t] for t in targets]).transpose(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
  )