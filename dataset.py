"""
dataset.py - Loading and preprocess the data
Contains data loader for the BioLaySumm project
"""

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BioLaySumm(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = "Translate into plain language for a general audience: "

        inputs = self.tokenizer(
            prompt + item['radiology_report'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        targets = self.tokenizer(
            item['layman_report'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }



