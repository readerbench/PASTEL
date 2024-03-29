import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class ParaphraseDataset(Dataset):
  def __init__(self, source, production, targets, tokenizer, max_len):
    self.source = source
    self.production = production
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.source)

  def __getitem__(self, item):
    source = str(self.source[item])
    production = str(self.production[item])

    input_ids, att_mask = encode_paraphrase_pair(source, production, self.tokenizer, self.max_len)
    result_dict = {
      'text_s': source,
      'text_p': production,
      'input_ids': input_ids,
      'attention_mask': att_mask,
      'item': item
    }
    if self.targets is not None:
      result_dict['targets'] = torch.tensor(self.targets[item], dtype=torch.long)
    return result_dict

def encode_paraphrase_pair(source, production, tokenizer, max_len):
  encoding = tokenizer.encode_plus(
    text=source,
    text_pair=production,
    truncation=True,
    truncation_strategy="longest_first",
    add_special_tokens=True,
    max_length=max_len,
    return_token_type_ids=False,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
  )

  return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()

def create_data_loader(df, tokenizer, max_len, batch_size, class_name):
  ds = ParaphraseDataset(
    source=df["Source"].to_numpy(),
    production=df["Production"].to_numpy(),
    targets=df[class_name].to_numpy() if class_name is not None else None,
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
  )