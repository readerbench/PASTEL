from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch import optim
from transformers import BertModel
from torch.nn import functional as F
from torch.nn import ModuleList
from torchmetrics import F1Score, Accuracy

from core.data_processing.se_dataset import SelfExplanations

class BERTMTL(pl.LightningModule):
  def __init__(self, num_tasks, pretrained_bert_model, rb_feats=0):
    super().__init__()
    self.bert = BertModel.from_pretrained(pretrained_bert_model, return_dict=False)
    self.drop = nn.Dropout(p=0.2)
    self.tmp1 = nn.Linear(self.bert.config.hidden_size, 100)
    self.task_names = SelfExplanations.MTL_TARGETS[:num_tasks]
    task_classes = [SelfExplanations.MTL_CLASS_DICT[x] for x in self.task_names]
    self.num_tasks = num_tasks
    self.iteration_stat_aggregator = OrderedDict()
    self.train_iter_counter = 0
    self.val_iter_counter = 0

    self.rb_feats = rb_feats
    if self.rb_feats > 0:
      self.rb_feats_in = nn.Linear(self.rb_feats, 100)
      self.out = ModuleList([nn.Linear(200, task_classes[i]) for i in range(num_tasks)])
    else:
      self.out = ModuleList([nn.Linear(100, task_classes[i]) for i in range(num_tasks)])
    self.reset_iteration_stat_aggregator()

  def reset_iteration_stat_aggregator(self):
    self.iteration_stat_aggregator[f"train_loss"] = 0
    self.iteration_stat_aggregator[f"val_loss"] = 0
    for key in self.task_names:
      self.iteration_stat_aggregator[f"{key}_train_loss"] = 0
      self.iteration_stat_aggregator[f"{key}_val_loss"] = 0
    self.train_iter_counter = 0
    self.val_iter_counter = 0

  def forward(self, input_ids, attention_mask, rb_feats_data=None):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    x = self.drop(pooled_output)
    x = F.tanh(self.tmp1(x))
    
    if self.rb_feats > 0:
      feats = F.tanh(self.rb_feats_in(rb_feats_data))
      x = F.tanh(torch.cat([feats, x], dim=1))

    x = [F.softmax(f(x)) for f in self.out]

    return x

  def process_batch_get_losses(self, batch):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    targets = batch['targets']
    if self.rb_feats > 0:
      rb_feats_data = batch['rb_feats'].to(torch.float32)
      outputs = self(input_ids, attention_mask, rb_feats_data)
    else:
      outputs = self(input_ids, attention_mask)
    loss_f = nn.CrossEntropyLoss()
    partial_losses = [0 for _ in range(self.num_tasks)]

    transp_targets = targets.transpose(1, 0)
    for task_id in range(self.num_tasks):
      task_mask = transp_targets[task_id] != 9
      partial_losses[task_id] += loss_f(outputs[task_id][task_mask], transp_targets[task_id][task_mask])
    loss = sum(partial_losses)

    return loss, partial_losses, transp_targets, outputs

  def training_step(self, batch, batch_idx):
    loss, partial_losses, _, _ = self.process_batch_get_losses(batch)

    # Logging to TensorBoard by default
    self.iteration_stat_aggregator["train_loss"] += loss
    for i, task in enumerate(self.task_names):
      self.iteration_stat_aggregator[f"{task}_train_loss"] += partial_losses[i]
    self.train_iter_counter += 1

    return loss

  def validation_step(self, batch, batch_idx):
    loss, partial_losses, transp_targets, outputs = self.process_batch_get_losses(batch)

    out_targets = transp_targets.cpu()
    out_outputs = torch.Tensor([[y.argmax() for y in x] for x in outputs]).cpu()
    # Logging to TensorBoard by default
    self.iteration_stat_aggregator["val_loss"] += loss
    for i, task in enumerate(self.task_names):
      self.iteration_stat_aggregator[f"{task}_val_loss"] += partial_losses[i]
    self.val_iter_counter += 1
    return (out_targets, out_outputs)
  def test_step(self, batch, batch_idx):
    _, _, transp_targets, outputs = self.process_batch_get_losses(batch)

    out_targets = transp_targets.cpu()
    out_outputs = torch.Tensor([[y.argmax() for y in x] for x in outputs]).cpu()
    return (out_targets, out_outputs)

  def test_epoch_end(self, validation_step_outputs):
    targets = [x[0] for x in validation_step_outputs]
    outputs = [x[1] for x in validation_step_outputs]

    task_targets = [[] for _ in range(self.num_tasks)]
    task_outputs = [[] for _ in range(self.num_tasks)]

    for i in range(self.num_tasks):
      print("=" * 5 + self.task_names[i] + "=" * 20)
      for j in range(len(targets)):
        task_targets[i].append(targets[j][i])
        task_outputs[i].append(outputs[j][i])

      task_targets[i] = torch.cat(task_targets[i])
      task_outputs[i] = torch.cat(task_outputs[i])
      task_mask = task_targets[i] != 9
      filtered_targets = task_targets[i][task_mask].int()
      filtered_outputs = task_outputs[i][task_mask].int()
      print(confusion_matrix(filtered_targets, filtered_outputs))
      print(classification_report(filtered_targets, filtered_outputs))


  def validation_epoch_end(self, validation_step_outputs):
    targets = [x[0] for x in validation_step_outputs]
    outputs = [x[1] for x in validation_step_outputs]

    task_targets = [[] for _ in range(self.num_tasks)]
    task_outputs = [[] for _ in range(self.num_tasks)]

    for i in range(self.num_tasks):
      for j in range(len(targets)):
        task_targets[i].append(targets[j][i])
        task_outputs[i].append(outputs[j][i])

      task_targets[i] = torch.cat(task_targets[i])
      task_outputs[i] = torch.cat(task_outputs[i])
      task_mask = task_targets[i] != 9
      filtered_targets = task_targets[i][task_mask].int()
      filtered_outputs = task_outputs[i][task_mask].int()
      f1 = F1Score(num_classes=SelfExplanations.MTL_CLASS_DICT[self.task_names[i]])
      acc = Accuracy()

      for key in self.iteration_stat_aggregator:
        if key.endswith("val_loss") and self.val_iter_counter > 0:
          self.log(key, self.iteration_stat_aggregator[key] / self.val_iter_counter)
        elif self.train_iter_counter > 0:
          self.log(key, self.iteration_stat_aggregator[key] / self.train_iter_counter)
      self.reset_iteration_stat_aggregator()
      self.log(f"acc_{self.task_names[i]}", acc(filtered_outputs, filtered_targets))

  def configure_optimizers(self):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      # BERT params - with WD
      {'params': [p for n, p in self.named_parameters() if
                  not any(nd in n for nd in no_decay) and n.find("bert") != -1],
       'weight_decay': 0.0001, 'lr': 1e-5},
      # BERT params - no WD
      {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and n.find("bert") != -1],
       'weight_decay': 0.0, 'lr': 1e-5},
      # non-BERT params - with WD
      {'params': [p for n, p in self.named_parameters() if
                  not any(nd in n for nd in no_decay) and n.find("bert") == -1],
       'weight_decay': 0.0001},
      # non-BERT params - no WD
      {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and n.find("bert") == -1],
       'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-3)

    # optimizer = optim.Adam(self.parameters(), lr=1e-3)
    return optimizer