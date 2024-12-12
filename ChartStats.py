import json
from torch.utils.data import Dataset
import torch
import math

class chartStats(Dataset):
    def __init__(self, token_file, boundary):
        self.boundary = boundary
        with open(token_file, 'r') as fp:
            self.data = json.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]['tokens']
        label = int(self.data[idx]['rating_num'] < self.boundary)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.int32)

    def get_name(self, idx):
        return self.data[idx]['name']

    def get_rate(self, idx):
        return self.data[idx]['rating_num']
    
    def max(self):
        cur = 0
        for i in range(len(self)):
            if cur < self[i][1]:
                cur = self[i][1]
        return cur
        
    def min(self):
        cur = math.inf
        for i in range(len(self)):
            if cur > self[i][1]:
                cur = self[i][1]
        return cur
    
    