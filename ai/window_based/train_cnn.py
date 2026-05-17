"""
1D CNN klasifikacija za turn in hill z PyTorch.

Vhod:
    dataset_output/X_all.npy
    dataset_output/Y_turn_all.npy
    dataset_output/Y_hill_all.npy
    dataset_output/log_ids.npy

Izhod:
    dataset_output/cnn_model.pt
    dataset_output/cnn_loss_fold*.png
    dataset_output/cnn_loss_avg.png
"""

from pathlib import Path
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Poti ─────────────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent / 'dataset_output'

# ── Hiperparametri ────────────────────────────────────────────────────────────

EPOCHS     = 200
BATCH_SIZE = 32
LR         = 3e-4
PATIENCE   = 20

# ── Dataset ───────────────────────────────────────────────────────────────────

class IMUDataset(Dataset):
    def __init__(self, X, Y_turn, Y_hill):
        self.X      = torch.from_numpy(X).permute(0, 2, 1)  # (N,100,9) → (N,9,100)
        self.Y_turn = torch.from_numpy(Y_turn)
        self.Y_hill = torch.from_numpy(Y_hill)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_turn[idx], self.Y_hill[idx]

# ── Model ─────────────────────────────────────────────────────────────────────

class IMU_CNN(nn.Module):
    def __init__(self, n_channels=9, n_turn=3, n_hill=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head_turn = nn.Linear(64, n_turn)
        self.head_hill = nn.Linear(64, n_hill)

    def forward(self, x):
        f = self.fc_shared(self.encoder(x))
        return self.head_turn(f), self.head_hill(f)

# ── Training funkcije ─────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yt, yh in loader:
            xb, yt, yh = xb.to(device), yt.to(device), yh.to(device)
            out_t, out_h = model(xb)
            loss = criterion(out_t, yt) + criterion(out_h, yh)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_t, all_h = [], []
    for xb, _, _ in loader:
        out_t, out_h = model(xb.to(device))
        all_t.append(out_t.argmax(1).cpu())
        all_h.append(out_h.argmax(1).cpu())
    return torch.cat(all_t).numpy(), torch.cat(all_h).numpy()


def train_fold(X_tr, Yt_tr, Yh_tr, X_val, Yt_val, Yh_val, device):
    train_loader = DataLoader(IMUDataset(X_tr, Yt_tr, Yh_tr),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(IMUDataset(X_val, Yt_val, Yh_val),
                              batch_size=BATCH_SIZE, shuffle=False)

    model     = IMU_CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, patience_count = float('inf'), None, 0
    tr_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        tr  = run_epoch(model, train_loader, criterion, device, optimizer)
        val = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()
        tr_losses.append(tr)
        val_losses.append(val)

        if val < best_val:
            best_val       = val
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, val_loader, tr_losses, val_losses


def plot_loss(tr_losses, val_losses, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tr_losses,  label='Train loss', color='steelblue')
    ax.plot(val_losses, label='Val loss',   color='darkorange')
    ax.set_xlabel('Epoha')
    ax.set_ylabel('Izguba')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ── Log-based cross-validation ────────────────────────────────────────────────

def log_kfold(log_ids, n_folds=5):
    """
    Vrne seznam (train_idx, val_idx) parov kjer je val_idx
    vedno okna iz logov ki niso v train setu.
    """
    unique_logs = np.unique(log_ids)
    #n_logs      = len(unique_logs)

    # razdeli loge v n_folds skupin
    log_groups = np.array_split(unique_logs, n_folds)

    folds = []
    for val_logs in log_groups:
        val_mask  = np.isin(log_ids, val_logs)
        train_idx = np.where(~val_mask)[0]
        val_idx   = np.where(val_mask)[0]
        folds.append((train_idx, val_idx))
        print(f"  Val logi: {val_logs.tolist()}  "
              f"(train={len(train_idx)} oken, val={len(val_idx)} oken)")

    return folds

# ── Glavni program ────────────────────────────────────────────────────────────

def main():
    print("Nalagam dataset...")
    X       = np.load(DATASET_DIR / 'X_all.npy').astype(np.float32)
    Y_turn  = np.load(DATASET_DIR / 'Y_turn_all.npy').astype(np.int64)
    Y_hill  = np.load(DATASET_DIR / 'Y_hill_all.npy').astype(np.int64)
    log_ids = np.load(DATASET_DIR / 'log_ids.npy')
    print(f"  X: {X.shape},  logi: {np.unique(log_ids).tolist()}")
    print(f"  Y_turn: none={np.sum(Y_turn==0)}  left={np.sum(Y_turn==1)}  right={np.sum(Y_turn==2)}")
    print(f"  Y_hill: none={np.sum(Y_hill==0)}  up={np.sum(Y_hill==1)}    down={np.sum(Y_hill==2)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}\n")

    print("Log-based cross-validation split:")
    folds = log_kfold(log_ids, n_folds=5)

    all_yt_true, all_yt_pred = [], []
    all_yh_true, all_yh_pred = [], []
    all_tr_losses, all_val_losses = [], []

    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr,  X_val  = X[train_idx],      X[val_idx]
        Yt_tr, Yt_val = Y_turn[train_idx], Y_turn[val_idx]
        Yh_tr, Yh_val = Y_hill[train_idx], Y_hill[val_idx]

        model, val_loader, tr_l, val_l = train_fold(
            X_tr, Yt_tr, Yh_tr, X_val, Yt_val, Yh_val, device
        )

        yt_pred, yh_pred = predict(model, val_loader, device)
        all_yt_true.extend(Yt_val);  all_yt_pred.extend(yt_pred)
        all_yh_true.extend(Yh_val);  all_yh_pred.extend(yh_pred)
        all_tr_losses.append(tr_l)
        all_val_losses.append(val_l)

        f1t = f1_score(Yt_val, yt_pred, average='macro', zero_division=0)
        f1h = f1_score(Yh_val, yh_pred, average='macro', zero_division=0)
        print(f"  Fold {fold+1}: F1_turn={f1t:.3f}  F1_hill={f1h:.3f}  "
              f"F1_avg={((f1t+f1h)/2):.3f}  (ep={len(tr_l)})")

        plot_loss(tr_l, val_l,
                  f'1D CNN loss — Fold {fold+1}',
                  DATASET_DIR / f'cnn_loss_fold{fold+1}.png')

    # Povprečni loss graf čez vse folde
    max_ep  = max(len(l) for l in all_tr_losses)
    pad     = lambda ls: [l + [l[-1]] * (max_ep - len(l)) for l in ls]
    avg_tr  = np.mean(pad(all_tr_losses),  axis=0)
    avg_val = np.mean(pad(all_val_losses), axis=0)
    plot_loss(avg_tr, avg_val,
              '1D CNN loss — povprečje vseh foldov',
              DATASET_DIR / 'cnn_loss_avg.png')
    print(f"\n  Grafi shranjeni v: {DATASET_DIR}")

    # Končni rezultati
    print("\n" + "="*54)
    print("1D CNN — TURN  (log-based cross-validation)")
    print("="*54)
    print(classification_report(
        all_yt_true, all_yt_pred,
        target_names=['none', 'left', 'right'], zero_division=0
    ))

    print("="*54)
    print("1D CNN — HILL  (log-based cross-validation)")
    print("="*54)
    print(classification_report(
        all_yh_true, all_yh_pred,
        target_names=['none', 'up', 'down'], zero_division=0
    ))

    f1t = f1_score(all_yt_true, all_yt_pred, average='macro', zero_division=0)
    f1h = f1_score(all_yh_true, all_yh_pred, average='macro', zero_division=0)
    print(f"Skupna metrika za primerjavo z BiLSTM/GRU:")
    print(f"  F1_turn (macro) = {f1t:.3f}")
    print(f"  F1_hill (macro) = {f1h:.3f}")
    print(f"  F1_avg          = {((f1t+f1h)/2):.3f}  ← primerjaj s kolegi")

    # Finalni model na celotnem datasetu
    print("\nTreniram finalni model na celotnem datasetu...")
    full_loader = DataLoader(IMUDataset(X, Y_turn, Y_hill),
                             batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    final = IMU_CNN().to(device)
    opt   = torch.optim.Adam(final.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        run_epoch(final, full_loader, crit, device, opt)
        sch.step()

    save_path = DATASET_DIR / 'cnn_model.pt'
    torch.save(final.state_dict(), save_path)
    print(f"  Model shranjen: {save_path}")
    print(f"  Parametri: {sum(p.numel() for p in final.parameters()):,}")


if __name__ == '__main__':
    main()