# ✅ 1. PyTorch Dataset: Construct multi-task time series data
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns

sns.set(style="whitegrid")

import torch
print(torch.__version__)

# ✅ Centered Moving Average (CMA) function
def compute_cma(volume, window=5):
    half = window // 2
    padded = np.pad(volume, (half, half), mode='edge')
    cma = np.convolve(padded, np.ones(window)/window, mode='valid')
    return cma

# ✅ Structure compression function (inflection point extraction)
def extract_structure_points(series):
    extrema = []
    for i in range(1, len(series)-1):
        if (series[i] > series[i-1] and series[i] > series[i+1]) or \
           (series[i] < series[i-1] and series[i] < series[i+1]):
            extrema.append((i, series[i]))
    return np.array(extrema)

# ✅ PyTorch dataset class
class StockMultiTaskDataset11(Dataset):
    def __init__(self, close_series, volume_series, seq_len=20):
        self.seq_len = seq_len

        # Volume CMA smoothing
        volume_cma = compute_cma(volume_series, window=5)

        # Closing price structural compression and interpolation recovery
        compressed_close = np.zeros_like(close_series)
        structure_points = extract_structure_points(close_series)
        for i, v in structure_points:
            compressed_close[int(i)] = v
        nonzero_idx = np.where(compressed_close != 0)[0]
        compressed_close = np.interp(np.arange(len(close_series)), nonzero_idx, compressed_close[nonzero_idx])

        # Construct DataFrame and normalize
        processed_df = pd.DataFrame({"close": compressed_close, "volume": volume_cma})
        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(processed_df)
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]                # [seq_len, 2]
        next_price = self.data[idx + self.seq_len, 0]          # Closing price
        next_volume = self.data[idx + self.seq_len, 1]         # Trading volume
        return torch.tensor(seq, dtype=torch.float32), \
               torch.tensor([next_price], dtype=torch.float32), \
               torch.tensor([next_volume], dtype=torch.float32)

class StockMultiTaskDataset(Dataset):
    def __init__(self, close_series, volume_series, seq_len=20):
        self.seq_len = seq_len

        # ✅ 1. Volume CMA smoothing
        volume_cma = compute_cma(volume_series, window=5)

        # ✅ 2. Closing price structural compression and interpolation recovery
        compressed_close = np.zeros_like(close_series)
        structure_points = extract_structure_points(close_series)
        for i, v in structure_points:
            compressed_close[int(i)] = v
        nonzero_idx = np.where(compressed_close != 0)[0]
        compressed_close = np.interp(np.arange(len(close_series)), nonzero_idx, compressed_close[nonzero_idx])

        # ✅ 3. Normalize separately
        self.close_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()

        close_norm = self.close_scaler.fit_transform(compressed_close.reshape(-1, 1))
        volume_norm = self.volume_scaler.fit_transform(volume_cma.reshape(-1, 1))

        # ✅ 4. Merge into [N, 2]
        self.data = np.hstack([close_norm, volume_norm])

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]                # [seq_len, 2]
        next_price = self.data[idx + self.seq_len, 0]          # Closing price
        next_volume = self.data[idx + self.seq_len, 1]         # Trading volume
        return torch.tensor(seq, dtype=torch.float32), \
               torch.tensor([next_price], dtype=torch.float32), \
               torch.tensor([next_volume], dtype=torch.float32)


# ✅ create_dataloaders receives raw closing price and trading volume series
def create_dataloaders_from_series11(close_series, volume_series, seq_len=20, batch_size=32):
    dataset = StockMultiTaskDataset(close_series, volume_series, seq_len=seq_len)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, dataset.scaler

def create_dataloaders_from_series(close_series, volume_series, seq_len=10, batch_size=16):
    dataset = StockMultiTaskDataset(close_series, volume_series, seq_len=seq_len)
    total_size = len(dataset)
    
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # ✅ Sequential slicing
    train_indices = range(0, train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_size)

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    # ✅ shuffle=False to ensure time order
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




# ✅ 2. Transformer Encoder-only Multi-task model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerMultiTask(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.price_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.volume_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        # x: [B, T, 2]
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_encoder(x)  # [B, T, d_model]
        x = x.permute(1, 0, 2)  # [T, B, d_model] for Transformer
        encoded = self.encoder(x)  # [T, B, d_model]
        last_token = encoded[-1]  # [B, d_model]
        return self.price_head(last_token), self.volume_head(last_token)


# ✅ 3. Multi-task loss function
def multitask_loss(price_pred, price_true, vol_pred, vol_true, alpha=1.0, beta=0.2):
    loss_fn = nn.MSELoss()
    loss_price = loss_fn(price_pred, price_true)
    loss_vol = loss_fn(vol_pred, vol_true)
    return alpha * loss_price + beta * loss_vol


# ✅ 4. Data loading and splitting
from sklearn.model_selection import train_test_split


def create_dataloaders11(df, seq_len=20, batch_size=8):
    dataset = StockMultiTaskDataset(df, seq_len=seq_len)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def create_dataloaders(df, seq_len=20, batch_size=1):
    dataset = StockMultiTaskDataset(df, seq_len=seq_len)
    total_size = len(dataset)
    
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # ✅ Sequential slicing
    train_set = torch.utils.data.Subset(dataset, range(0, train_size))
    val_set = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_set = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))

    # ✅ Note: Time-series prediction cannot be shuffled
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader



# ✅ 5. Training and validation loop (with early stopping)
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    counter = 0
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y_price, y_vol in train_loader:
            y_price_pred, y_vol_pred = model(x)
            loss = multitask_loss(y_price_pred, y_price, y_vol_pred, y_vol)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y_price, y_vol in val_loader:
                y_price_pred, y_vol_pred = model(x)
                loss = multitask_loss(y_price_pred, y_price, y_vol_pred, y_vol)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        history["train"].append(avg_train)
        history["val"].append(avg_val)
        print(f"Epoch {epoch + 1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "best_model.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    return history


# ✅ 6. Visualization of loss curves
def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="Train Loss")
    plt.plot(history["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.eps", format="eps")
    plt.show()


# ✅ 7. Test set evaluation (MAE and RMSE) + prediction visualization

def evaluate_model(model, test_loader):
    model.eval()
    all_price_preds, all_price_trues = [], []
    all_vol_preds, all_vol_trues = [], []

    with torch.no_grad():
        for x, y_price, y_vol in test_loader:
            y_price_pred, y_vol_pred = model(x)
            all_price_preds.extend(y_price_pred.cpu().numpy())
            all_price_trues.extend(y_price.cpu().numpy())
            all_vol_preds.extend(y_vol_pred.cpu().numpy())
            all_vol_trues.extend(y_vol.cpu().numpy())

    mae_price = mean_absolute_error(all_price_trues, all_price_preds)
    mse_price = mean_squared_error(all_price_trues, all_price_preds)
    rmse_price = np.sqrt(mse_price)
    mae_vol = mean_absolute_error(all_vol_trues, all_vol_preds)
    mse_vol = mean_squared_error(all_vol_trues, all_vol_preds)
    rmse_vol = np.sqrt(mse_vol)
    #######################################################################
    #mae = mean_absolute_error(test_data, predictions)
    #rmse = np.sqrt(mean_squared_error(test_data, predictions))
    ################################################################################
    print(f"\nTest Results:")
    print(f"Price - MAE: {mae_price:.4f}, RMSE: {rmse_price:.4f}")
    print(f"Volume - MAE: {mae_vol:.4f}, RMSE: {rmse_vol:.4f}")
    np.savez("predictions_ours_update.npz", preds=np.array(all_price_preds), trues=np.array(all_price_trues))
    # Visualization: Closing price prediction vs true values
    plt.figure(figsize=(10, 4))
    plt.plot(all_price_trues[:200], label="True Price")
    plt.plot(all_price_preds[:200], label="Predicted Price")
    plt.title("Price Prediction vs True")
    plt.xlabel("Sample")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("price_prediction.eps", format="eps")
    plt.savefig("price_prediction.png", format="png")
    plt.show()

    # Visualization: Trading volume prediction vs true values
    plt.figure(figsize=(10, 4))
    plt.plot(all_vol_trues[:200], label="True Volume")
    plt.plot(all_vol_preds[:200], label="Predicted Volume")
    plt.title("Volume Prediction vs True")
    plt.xlabel("Sample")
    plt.ylabel("Normalized Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig("volume_prediction.eps", format="eps")
    plt.savefig("volume_prediction.png", format="png")
    plt.show()

# ✅ Usage example
file_path = '/home/featurize/data/data_information/information_list.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
close_series = df['Price'].astype(float).values[:-7]
volume_series = df['Volume'].astype(float).values[:-7]

print(len(close_series))
print(volume_series)
train_loader, val_loader, test_loader = create_dataloaders_from_series(close_series, volume_series)
model = TransformerMultiTask()
history = train_model(model, train_loader, val_loader)
model.load_state_dict(torch.load("best_model.pt"))
evaluate_model(model, test_loader)
