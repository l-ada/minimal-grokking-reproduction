import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED=42
torch.manual_seed(SEED)
random.seed(SEED)
MODULUS=97
def create_data(modulus):
    data_list=[]
    for a in range(modulus):
        for b in range(modulus):
            data_list.append((a,b, modulus+1,modulus+2, (a+b)%modulus))
    return data_list
data=create_data(MODULUS)
random.shuffle(data)
split_ratio=0.5
train_len = round(len(data)*split_ratio)
val_len = len(data)-train_len
train_data=data[:train_len]
val_data=data[train_len:]
class GrokkingDataset(Dataset):
    def __init__(self, data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_point = self.data[index]
        x, y = data_point[0:4], data_point[-1]
        return torch.tensor(x, dtype=torch.long),torch.tensor(y, dtype=torch.long)
train_dataset = GrokkingDataset(train_data)
val_dataset = GrokkingDataset(val_data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(MODULUS+3,128)
        self.positional_embedding = nn.Parameter(torch.randn(1,4,128))
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first=True,
            dropout=0.0)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.linear = nn.Linear(128,MODULUS)

    def forward(self, x):
        x = self.embedding(x)
        x = x+self.positional_embedding
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x

## training loop
# good advice
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
model = Net()
model.to(device)
weight_decay=1e-2
EPOCHS=100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=weight_decay)
for epoch in range(EPOCHS):
    for x,y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")