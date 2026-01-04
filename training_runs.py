import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import setup_dataset, GrokkingDataset
from model import Net
from tqdm.auto import tqdm



def train_sharpness(config, device):
    model = config['model']
    model = model.to(device)
    model.train()
    initial_params = [p.detach.clone() for p in model.parameters()]
    batch_size = config['batch_size']
    train_dataset = config['train_dataset']
    val_dataset = config['val_dataset']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    weight_decay = config['weight_decay']
    learning_rate = config['lr']
    num_epochs = config['num_epochs']
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
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

        if epoch % 100 == 0:
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
        if epoch % 100 == 0:
        # Update the plot file every epoch
        #   plot_results(history)
            pass


        pbar.set_postfix({
            'loss': f"{epoch_train_loss:.4e}",
            'val_acc': f"{epoch_val_acc:.4f}",
            'parameter_distance' : f"{parameter_distance:.4f}"
        })
