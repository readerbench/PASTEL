from torch.nn.functional import relu, softmax, sigmoid
from transformers import BertModel
import torch.nn as nn


class BERTClassifier(nn.Module):
  def __init__(self, n_classes, pretrained_bert_model):
    super(BERTClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(pretrained_bert_model)
    self.drop = nn.Dropout(p=0.2)
    self.tmp = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)