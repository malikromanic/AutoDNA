"""
Popravljen minimalni 1D CNN z GroupKFold CV, utežmi za balansiranje,
Focal Loss funkcijo ter popravljeno časovno ločljivostjo za klančine.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
import torch.nn.functional as F

DATASET_DIR = Path(__file__).parent / 'dataset_output'
EPOCHS, BATCH_SIZE, LR = 40, 32, 1e-3

class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class ``alpha`` weighting.

    Down-weights easy examples via the ``(1 - pt) ** gamma`` factor so rare
    classes (e.g. hills) contribute more to the gradient.
    """

    def __init__(self, alpha=None, gamma=2.0):
        """Store the per-class weight tensor *alpha* and focusing parameter *gamma*."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Return the mean focal loss for logits *inputs* against class *targets*."""
        log_pt = F.log_softmax(inputs, dim=1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)
        
        loss = -((1 - pt) ** self.gamma) * log_pt
        
        if self.alpha is not None:
            at = self.alpha[targets]
            loss = loss * at
            
        return loss.mean()

class MinimalIMUDataset(Dataset):
    """Windowed IMU dataset yielding ``(C, T)`` tensors for :class:`SimpleCNN`.

    With *augment* enabled, the accel X/Y axes are randomly rotated to make the
    turn/hill classifier invariant to sensor mounting yaw.
    """

    def __init__(self, X, Y, augment=False):
        """Wrap the ``(N, T, C)`` windows *X* and labels *Y*; toggle rotation *augment*."""
        self.X = torch.from_numpy(X).float()  # (N, T, C)
        self.Y = torch.from_numpy(Y).long()
        self.augment = augment

    def __len__(self):
        """Return the number of windows in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return one window permuted to ``(C, T)`` (optionally yaw-augmented) and its label."""
        x = self.X[idx].clone()
        
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            ax_raw = x[:, 3].clone()
            ay_raw = x[:, 4].clone()
            
            x[:, 3] = ax_raw * cos_t - ay_raw * sin_t
            x[:, 4] = ax_raw * sin_t + ay_raw * cos_t

        # Permutacija v (C, T) za potrebe Conv1d
        return x.permute(1, 0), self.Y[idx]

class SimpleCNN(nn.Module):
    """Compact 1-D CNN classifier for fixed-length IMU windows.

    ``BatchNorm -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> AdaptiveAvgPool ->
    Linear``, mapping *n_channels* input signals to *n_classes* logits.
    """

    def __init__(self, n_channels=4, n_classes=3):
        """Build the conv stack for *n_channels* inputs and *n_classes* outputs."""
        super().__init__()
        self.net = nn.Sequential(
            # POPRAVEK: Ker podatki niso več uničeni z normalizacijo v preprocessing koraku,
            # stabilnost učenja zdaj elegantno rešujemo z BatchNorm slojem na vhodu.
            nn.BatchNorm1d(n_channels),
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # POPRAVEK: Povečana ločljivost temporalnih blokov iz 4 na 8 točk,
            # da model ne izgubi finih prehodov in dinamike začetka/konca klančine.
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            # 64 kanalov * 8 časovnih blokov = 512 vhodnih nevronov
            nn.Linear(64 * 8, n_classes)
        )

    def forward(self, x):
        """Run the conv stack on ``(N, C, T)`` input and return ``(N, n_classes)`` logits."""
        return self.net(x)

def train_simple_task(X, Y, log_ids, task_name):
    """Train the CNN for one task (turn or hill) with 5-fold GroupKFold CV and report OOF metrics."""
    print("\n==================================================")
    print(f"Treniram CNN (5-Fold CV + POPRAVKI): {task_name.upper()}")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gkf = GroupKFold(n_splits=5)
    
    oof_predictions = np.zeros(len(Y), dtype=np.int64)
    target_names = ['none', 'left', 'right'] if task_name == 'turn' else ['none', 'up', 'down']

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, Y, groups=log_ids)):
        print(f"-> Fold {fold + 1}/5...")
        
        train_dataset = MinimalIMUDataset(X[train_idx], Y[train_idx], augment=False)
        val_dataset   = MinimalIMUDataset(X[val_idx], Y[val_idx], augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        class_counts = np.bincount(Y[train_idx])
        class_weights = 1.0 / np.sqrt(class_counts)
        class_weights = class_weights / class_weights.sum() * 3.0
        class_weights = torch.FloatTensor(class_weights).to(device)

        if task_name == 'turn':
            focal_gamma = 0.0  # Navaden CrossEntropy za zavoje
        else:
            focal_gamma = 2.0  # Močan Focal Loss za reševanje težkih klančin

        model = SimpleCNN(n_channels=X.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()

        model.eval()
        fold_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                out = model(xb.to(device))
                fold_preds.extend(out.argmax(1).cpu().numpy())
        
        oof_predictions[val_idx] = fold_preds

    print(f"\n=== NOVO POPRAVLJENO POROČILO ZA CNN: {task_name.upper()} ===")
    print(classification_report(Y, oof_predictions, target_names=target_names, zero_division=0))
    torch.save(model.state_dict(), DATASET_DIR / f'cnn_minimal_{task_name}.pt')

def main():
    """Train and evaluate the CNN on both the turn and hill datasets."""
    X_turn = np.load(DATASET_DIR / 'X_turn.npy')
    Y_turn = np.load(DATASET_DIR / 'Y_turn.npy')
    ids_turn = np.load(DATASET_DIR / 'log_ids_turn.npy')
    train_simple_task(X_turn, Y_turn, ids_turn, 'turn')

    X_hill = np.load(DATASET_DIR / 'X_hill.npy')
    Y_hill = np.load(DATASET_DIR / 'Y_hill.npy')
    ids_hill = np.load(DATASET_DIR / 'log_ids_hill.npy')
    train_simple_task(X_hill, Y_hill, ids_hill, 'hill')

if __name__ == '__main__':
    main()