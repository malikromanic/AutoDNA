"""
train_cnn.py
============
1D CNN klasifikacija za turn in hill -- DVA LOČENA MODELA.

Izboljšave v2:
    - 11 kanalov: gyro + accel + (preskočeno mag) + pitch_cf + roll_cf.
    - Per-channel z-score normalizacija (iz TRAIN, aplicirano na val).
    - **Focal Loss** namesto plain CE (boljši za hudo imbalance).
    - Class alpha uteži v Focal Loss.
    - Turn model: 2s okna, lažja arhitektura.
    - Hill model: 4s okna, dilated convs (večji receptive field).
    - StratifiedGroupKFold.
    - AdamW + OneCycleLR, gradient clipping.

Vhod:
    dataset_output/X_turn.npy, Y_turn.npy, log_ids_turn.npy
    dataset_output/X_hill.npy, Y_hill.npy, log_ids_hill.npy

Izhod:
    dataset_output/cnn_model_turn.pt, cnn_model_hill.pt
    dataset_output/cnn_loss_turn_fold*.png, cnn_loss_hill_fold*.png
    dataset_output/cnn_loss_turn_avg.png, cnn_loss_hill_avg.png
"""

from pathlib import Path
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Poti ─────────────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent / 'dataset_output'

# ── Hiperparametri ────────────────────────────────────────────────────────────

EPOCHS     = 200
BATCH_SIZE = 32
LR         = 3e-4
PATIENCE   = 25

# Kanali: gyro (0-2), accel (3-5), pitch_cf (9), roll_cf (10). Brez mag (6-8).
USE_CHANNELS = [0, 1, 2, 3, 4, 5, 9, 10]
N_CHANNELS   = len(USE_CHANNELS)

# Focal Loss parameter
# OPOMBA: gamma=0 je ekvivalentno navadni CE s class weights.
# Ekstremne vrednosti (2.5) so prej povzročile over-correction in padec F1.
FOCAL_GAMMA_TURN = 0.0   # uporabi navadno CE
FOCAL_GAMMA_HILL = 1.0   # zmerni focal effect

# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss z razrednimi alpha utežmi.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - alpha:  per-class uteži (npr. inverz class frekvenc)
    - gamma:  focus parameter; 0 = navadna CE, >0 znižuje kazen za "lahke" primere
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        # p_t = exp(-CE) -- verjetnost dodeljena pravemu razredu
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()

# ── Dataset ───────────────────────────────────────────────────────────────────

class IMUDataset(Dataset):
    def __init__(self, X, Y, mean=None, std=None):
        # X: (N, T, 11) -> izberi kanale -> permute (N, C, T)
        X = X[:, :, USE_CHANNELS]
        X = torch.from_numpy(X).permute(0, 2, 1).float()

        if mean is None:
            mean = X.mean(dim=(0, 2), keepdim=False)
            std  = X.std(dim=(0, 2),  keepdim=False) + 1e-6
        self.mean = mean
        self.std  = std

        X = (X - mean[None, :, None]) / std[None, :, None]
        self.X = X
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ── Modela ────────────────────────────────────────────────────────────────────

class IMU_CNN_Turn(nn.Module):
    """Za 2s okna -- turn je hitra dinamika."""
    def __init__(self, n_channels=N_CHANNELS, n_classes=3):
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class IMU_CNN_Hill(nn.Module):
    """
    Za 4s okna -- hill je počasna dinamika.
    Dilated convs povečajo receptive field brez izgube T resolucije.
    GAP + GMP concat za zajem tako povprečnega kot ekstremnega stanja.
    """
    def __init__(self, n_channels=N_CHANNELS, n_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        h = self.encoder(x)
        gap = h.mean(dim=-1)
        gmp = h.max(dim=-1).values
        return self.head(torch.cat([gap, gmp], dim=1))

# ── Training funkcije ─────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        out = model(xb.to(device))
        preds.append(out.argmax(1).cpu())
    return torch.cat(preds).numpy()


def train_fold(model_class, X_tr, Y_tr, X_val, Y_val, device,
               class_weights, focal_gamma):
    train_ds = IMUDataset(X_tr, Y_tr)
    val_ds   = IMUDataset(X_val, Y_val, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR*3,
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )

    alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # Brez label smoothing -- v kombinaciji s focal in class weights je bilo
    # preveč regularizacije in model je over-correction-al k manjšinskim razredom.
    criterion = FocalLoss(alpha=alpha, gamma=focal_gamma, label_smoothing=0.0)

    best_val, best_state, patience_count = float('inf'), None, 0
    tr_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        tr  = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        val = run_epoch(model, val_loader,   criterion, device)
        tr_losses.append(tr); val_losses.append(val)

        if val < best_val:
            best_val       = val
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, val_loader, tr_losses, val_losses, train_ds.mean, train_ds.std


def plot_loss(tr_losses, val_losses, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tr_losses,  label='Train loss', color='steelblue')
    ax.plot(val_losses, label='Val loss',   color='darkorange')
    ax.set_xlabel('Epoha'); ax.set_ylabel('Izguba (Focal)')
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ── Per-task pipeline ─────────────────────────────────────────────────────────

def train_task(X, Y, log_ids, task_name, model_class, target_names,
               focal_gamma, device):
    print(f"\n{'='*54}")
    print(f"TRENIRAM: {task_name.upper()}  -- model: {model_class.__name__}")
    print(f"  Focal gamma: {focal_gamma}")
    print(f"{'='*54}")
    print(f"  X: {X.shape}, Y dist: {[int(np.sum(Y==c)) for c in range(3)]} -> {target_names}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, Y, groups=log_ids))

    all_true, all_pred = [], []
    all_tr_losses, all_val_losses = [], []

    for fold, (train_idx, val_idx) in enumerate(folds):
        val_logs = np.unique(log_ids[val_idx])
        print(f"\n  fold {fold+1}: val_logi={val_logs.tolist()}  "
              f"train={len(train_idx)} val={len(val_idx)}")

        X_tr,  X_val  = X[train_idx],  X[val_idx]
        Y_tr,  Y_val  = Y[train_idx],  Y[val_idx]

        # 'balanced' uteži so lahko zelo agresivne pri 9:1 imbalance.
        # sqrt scaling jih ublaži (sklearn 'balanced' daje npr. [0.4, 3.8, 4.6],
        # sqrt scaling pa [0.6, 1.95, 2.15] -- še vedno favorizira manjšinske,
        # ampak ne tako agresivno da bi model napovedoval samo njih).
        cw_raw = compute_class_weight('balanced', classes=np.arange(3), y=Y_tr)
        cw = np.sqrt(cw_raw)
        cw = cw / cw.mean()   # normaliziraj okrog 1.0

        model, val_loader, tr_l, val_l, _, _ = train_fold(
            model_class, X_tr, Y_tr, X_val, Y_val, device, cw, focal_gamma
        )

        y_pred = predict(model, val_loader, device)
        all_true.extend(Y_val); all_pred.extend(y_pred)
        all_tr_losses.append(tr_l); all_val_losses.append(val_l)

        f1 = f1_score(Y_val, y_pred, average='macro', zero_division=0)
        print(f"    F1_{task_name}={f1:.3f}  (ep={len(tr_l)})")

        plot_loss(tr_l, val_l,
                  f'CNN {task_name} loss — fold {fold+1} (Focal γ={focal_gamma})',
                  DATASET_DIR / f'cnn_loss_{task_name}_fold{fold+1}.png')

    max_ep  = max(len(l) for l in all_tr_losses)
    pad     = lambda ls: [l + [l[-1]] * (max_ep - len(l)) for l in ls]
    avg_tr  = np.mean(pad(all_tr_losses),  axis=0)
    avg_val = np.mean(pad(all_val_losses), axis=0)
    plot_loss(avg_tr, avg_val,
              f'CNN {task_name} loss — povprečje vseh foldov',
              DATASET_DIR / f'cnn_loss_{task_name}_avg.png')

    print(f"\n  Classification report — {task_name}:")
    print(classification_report(all_true, all_pred,
        target_names=target_names, zero_division=0))

    f1_total = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"  F1_{task_name} (macro): {f1_total:.3f}")

    # Finalni model na celotnem datasetu
    print(f"\n  Treniram finalni {task_name} model na vsem...")
    cw_full_raw = compute_class_weight('balanced', classes=np.arange(3), y=Y)
    cw_full = np.sqrt(cw_full_raw)
    cw_full = cw_full / cw_full.mean()
    full_ds = IMUDataset(X, Y)
    mean_full, std_full = full_ds.mean, full_ds.std
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    final = model_class().to(device)
    opt   = torch.optim.AdamW(final.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR*3,
        total_steps=EPOCHS * len(full_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    alpha = torch.tensor(cw_full, dtype=torch.float32).to(device)
    crit = FocalLoss(alpha=alpha, gamma=focal_gamma, label_smoothing=0.0)
    for _ in range(EPOCHS):
        run_epoch(final, full_loader, crit, device, opt, sch)

    save_path = DATASET_DIR / f'cnn_model_{task_name}.pt'
    torch.save({
        'state_dict': final.state_dict(),
        'mean':       mean_full,
        'std':        std_full,
        'channels':   USE_CHANNELS,
        'model_class': model_class.__name__,
        'focal_gamma': focal_gamma,
    }, save_path)
    print(f"  Model shranjen: {save_path}")
    print(f"  Parametri: {sum(p.numel() for p in final.parameters()):,}")

    return f1_total

# ── Glavni program ────────────────────────────────────────────────────────────

def main():
    print("Nalagam dataset...")
    X_turn       = np.load(DATASET_DIR / 'X_turn.npy').astype(np.float32)
    Y_turn       = np.load(DATASET_DIR / 'Y_turn.npy').astype(np.int64)
    log_ids_turn = np.load(DATASET_DIR / 'log_ids_turn.npy')

    X_hill       = np.load(DATASET_DIR / 'X_hill.npy').astype(np.float32)
    Y_hill       = np.load(DATASET_DIR / 'Y_hill.npy').astype(np.int64)
    log_ids_hill = np.load(DATASET_DIR / 'log_ids_hill.npy')

    if X_turn.shape[2] < 11:
        print(f"[NAPAKA] Pričakujem 11 kanalov (z pitch_cf, roll_cf), dobil {X_turn.shape[2]}.")
        print(f"          Najprej poženi build_dataset.py.")
        return

    print(f"  turn: X={X_turn.shape}, logi={np.unique(log_ids_turn).tolist()}")
    print(f"  hill: X={X_hill.shape}, logi={np.unique(log_ids_hill).tolist()}")
    print(f"  Uporabljam kanale: {USE_CHANNELS} ({N_CHANNELS} kanalov)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    f1_turn = train_task(X_turn, Y_turn, log_ids_turn,
                         'turn', IMU_CNN_Turn,
                         ['none', 'left', 'right'],
                         FOCAL_GAMMA_TURN, device)

    f1_hill = train_task(X_hill, Y_hill, log_ids_hill,
                         'hill', IMU_CNN_Hill,
                         ['none', 'up', 'down'],
                         FOCAL_GAMMA_HILL, device)

    print("\n" + "="*54)
    print(f"FINAL: F1_turn={f1_turn:.3f}  F1_hill={f1_hill:.3f}  "
          f"F1_avg={(f1_turn+f1_hill)/2:.3f}")
    print("="*54)


if __name__ == '__main__':
    main()