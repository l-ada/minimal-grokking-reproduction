import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from IPython.display import clear_output # Added for dynamic plot updating

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
split_ratio=0.8
train_len = round(len(data)*split_ratio)
# train_len = 10
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

## training loop
# good advice
batch_size = 10000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
model = Net()
model.to(device)
weight_decay=2.0
learning_rate=0.0005
EPOCHS=14000
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
def plot_results(history):
    clear_output(wait=True) # Clear previous output for dynamic updating
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Loss (Log Scale is better for Grokking)
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.legend()
    ax1.set_title('Grokking Progress')

    # Plot Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()

    # plt.savefig('grokking_plot.png') # Removed as per user request
    # print("Plot saved as grokking_plot.png") # Removed as per user request
    plt.show() # Display the updated plot
    plt.close(fig) # Close the figure to free up memory
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for x,y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        # print(f"Epoch {epoch} | Loss: {loss.item():.4f}") # Commented out to reduce output verbosity
        # Data Collection
        running_loss += loss.item()
        # Optional: Train Accuracy (for the plot)
        preds = torch.argmax(y_pred, dim=1)
        correct_train += (preds == y).sum().item()
        total_train += y.size(0)
    # Average for the epoch
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = correct_train / total_train

    # --- VALIDATION PHASE (The part you requested) ---
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # Saves memory/time by not tracking gradients
        for x_v, y_v in val_loader:
            x_v, y_v = x_v.to(device), y_v.to(device)
            y_v_pred = model(x_v)
            v_loss = criterion(y_v_pred, y_v)

            val_running_loss += v_loss.item()
            v_preds = torch.argmax(y_v_pred, dim=1)
            correct_val += (v_preds == y_v).sum().item()
            total_val += y_v.size(0)

    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_acc = correct_val / total_val

    # --- DATA COLLECTION & LOGGING ---
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)

    print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    if epoch % 50 == 0:
    # Update the plot file every epoch
      plot_results(history)