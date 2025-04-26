import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SHOPEETextDataset(Dataset):
    def __init__(self, df, tokenizer=None):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        text = row.title
        text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        return input_ids, attention_mask, torch.tensor(row.label_group).float()
