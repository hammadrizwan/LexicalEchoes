import torch
from torch.utils.data import Dataset, DataLoader
import json
# Example single sample from your ANLI-like dataset

# Map labels to integers
label2id = {"e": 0, "c": 1, "n": 2}  # entailment, contradiction, neutral

class ANLIDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["context"], item["hypothesis"],torch.tensor(self.label2id[item["label"]], dtype=torch.long)

