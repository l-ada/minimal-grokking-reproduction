from torch import nn
import torch
class Net(nn.Module):
    def __init__(self, MODULUS):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(MODULUS+3,128)
        self.positional_embedding = nn.Parameter(torch.randn(1,4,128)*0.1)
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
