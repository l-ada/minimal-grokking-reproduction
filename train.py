import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import setup_dataset, GrokkingDataset
from model import Net
from tqdm.auto import tqdm
import math # Required for the bias correction in your MSAM class
from optimizer import AdamW_MSAM
from viz import plot_results
from utils import save_run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED=42
torch.manual_seed(SEED)
random.seed(SEED)
MODULUS=97

train_dataset, val_dataset = setup_dataset(MODULUS)
indices = list(range(100))
debug_train_dataset = Subset(train_dataset, indices)
debug_val_dataset = Subset(val_dataset, indices)


def setup_optimizer(model, config):
    # Extract name, default to 'adamw' if not found
    opt_name = config.get('optimizer_name', 'adamw').lower()

    params = model.parameters()
    lr = config['lr']
    wd = config['weight_decay']

    if opt_name == 'msam':
        return AdamW_MSAM(
            params,
            lr=lr,
            weight_decay=wd,
            rho=config.get('rho', 0.05)
        )
    if opt_name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    # Default to standard AdamW
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
def train(config):
    model = config['model']
    model = model.to(device)
    model.train()
    initial_params = [p.detach().clone() for p in model.parameters()]
    batch_size = config['batch_size']
    train_dataset = config['train_dataset']
    val_dataset = config['val_dataset']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = config['num_epochs']
    history = {'train_loss': [], 'val_loss': [],
                'train_acc': [], 'val_acc': [], 'param_distance': [], 'stopping_time': None}
    parameter_distance = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = setup_optimizer(model, config)
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for x, y in train_loader:
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

        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad(): 
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
        if history['stopping_time'] is None and epoch_val_acc == 1.0:
            print(f"Validation accuracy reached 100% at epoch {epoch}!")
            history['stopping_time'] = epoch
            # break # Uncomment to stop training when val acc hits 100%
        # --- DATA COLLECTION & LOGGING ---

        if epoch % 25 == 0:
            parameter_distance = 0
            total_sq_dist = 0.0
            with torch.no_grad():
                for init_param, param in zip(initial_params, model.parameters()):
                    # Sum the squared errors across all layers
                    total_sq_dist += torch.sum((init_param - param) ** 2).item()

            # The final Euclidean distance
            parameter_distance = total_sq_dist ** 0.5
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['param_distance'].append(parameter_distance)
        if epoch % 100 == 0:
        # Update the plot file every epoch
            # plot_results(history)
            pass


        pbar.set_postfix({
            'loss': f"{epoch_train_loss:.4e}",
            'train_acc': f"{epoch_train_acc:4f}",
            'val_acc': f"{epoch_val_acc:.4f}",
            'par_dist' : f"{parameter_distance:.4f}"
                })
    save_run(config, history)
# groks around epoch 980
config = {'batch_size': 10000, 'num_epochs': 1100, 'lr': 0.0005,
          'weight_decay': 2.0, 'model': Net(MODULUS),
          'train_dataset': train_dataset, 'val_dataset': val_dataset}
config_wd = {'batch_size': 10000, 'num_epochs': 1100, 'lr': 0.0005,
          'weight_decay': 1.0, 'model': Net(MODULUS),
          'train_dataset': train_dataset, 'val_dataset': val_dataset}
debug_config = {'batch_size': 10000, 'num_epochs': 2, 'lr': 0.0005,
          'weight_decay': 2.0, 'model': Net(MODULUS),
          'train_dataset': train_dataset, 'val_dataset': val_dataset}
overfit_config = {'batch_size': 10000, 'num_epochs': 1000, 'lr': 0.0005,
          'weight_decay': 2.0, 'model': Net(MODULUS),
          'train_dataset': debug_train_dataset, 'val_dataset': debug_val_dataset}
if __name__ == "__main__":
    train(config)
    train(config_wd)
