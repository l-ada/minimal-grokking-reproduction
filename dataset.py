import random

import torch
from torch.utils.data import Dataset


def create_data(modulus):
    data_list=[]
    for a in range(modulus):
        for b in range(modulus):
            data_list.append((a,b, modulus+1,modulus+2, (a+b)%modulus))
    return data_list

class GrokkingDataset(Dataset):
    def __init__(self, data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_point = self.data[index]
        x, y = data_point[0:4], data_point[-1]
        return torch.tensor(x, dtype=torch.long),torch.tensor(y, dtype=torch.long)

def setup_dataset(modulus):
    data = create_data(modulus)
    random.shuffle(data)
    split_ratio = 0.8
    train_len = round(len(data) * split_ratio)
    # train_len = 10
    train_data = data[:train_len]
    val_data = data[train_len:]
    train_dataset = GrokkingDataset(train_data)
    val_dataset = GrokkingDataset(val_data)
    return train_dataset, val_dataset