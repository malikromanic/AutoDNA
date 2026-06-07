import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# hyperparameters
num_layers    = 2
hidden_size   = 64
num_outputs   = 2        # turn_present, turn_dir
learning_rate = 0.001
num_epochs    = 200
patience      = 20
min_save_epoch = 15


def load_sample(filepath):
    data = np.load(filepath)
    gyro = data['gyro_rgb'].astype(np.float32) / 255.0  # (F, T, 3)
    gyro_z_mean = data['gyro_z_mean_per_window']        # (T,)
    y = data['labels'].astype(np.float32)[:, [0, 1]]   # (T, 2)

    F, T, _ = gyro.shape
    gyro_z_spec = gyro[:, :, 2].T                       # (T, F) — magnitude
    
    # normalize magnitude
    mn, mx = gyro_z_spec.min(), gyro_z_spec.max()
    if mx > mn:
        gyro_z_spec = (gyro_z_spec - mn) / (mx - mn)

    # normalize mean signal to [-1, 1]
    max_abs = np.abs(gyro_z_mean).max()
    if max_abs > 0:
        gyro_z_mean = gyro_z_mean / max_abs
    
    # append mean as extra feature column: (T, F+1)
    #accel = data['accel_rgb'].astype(np.float32) / 255.0
    accel_x_mean = data.get('accel_x_mean_per_window', None)
    
    if accel_x_mean is not None:
        max_abs_a = np.abs(accel_x_mean).max()
        if max_abs_a > 0:
            accel_x_mean = accel_x_mean / max_abs_a
        x = np.concatenate([
            gyro_z_spec,
            gyro_z_mean.reshape(-1, 1),
            accel_x_mean.reshape(-1, 1)
        ], axis=1)
    else:
        x = np.concatenate([gyro_z_spec, gyro_z_mean.reshape(-1, 1)], axis=1)
        
    #x = np.concatenate([gyro_z_spec, gyro_z_mean.reshape(-1, 1)], axis=1)

    return torch.tensor(x), torch.tensor(y)


def load_sample_g(filepath):   #samo gyro
    data = np.load(filepath)
    gyro = data['gyro_rgb'].astype(np.float32) / 255.0  # (F, T, 3)
    gyro_z_mean = data['gyro_z_mean_per_window']        # (T,)
    y = data['labels'].astype(np.float32)[:, [0, 1]]   # (T, 2)

    F, T, _ = gyro.shape
    gyro_z_spec = gyro[:, :, 2].T                       # (T, F) — magnitude
    
    # normalize magnitude
    mn, mx = gyro_z_spec.min(), gyro_z_spec.max()
    if mx > mn:
        gyro_z_spec = (gyro_z_spec - mn) / (mx - mn)

    # normalize mean signal to [-1, 1]
    max_abs = np.abs(gyro_z_mean).max()
    if max_abs > 0:
        gyro_z_mean = gyro_z_mean / max_abs
      
    x = np.concatenate([gyro_z_spec, gyro_z_mean.reshape(-1, 1)], axis=1)

    return torch.tensor(x), torch.tensor(y)


class DriveDataset(Dataset):
    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_sample_g(self.files[idx])


class BiLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_outputs)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.fc(out)          # raw logits, no sigmoid


def compute_loss(pred, target):
    """Simple BCE loss — no masking, no pos_weight complexity."""
    return nn.BCEWithLogitsLoss()(pred, target)


"""def compute_loss_custom(pred, target):
    bce = nn.BCEWithLogitsLoss(reduction='none')  # per-timestep loss
    loss_per_timestep = bce(pred, target)          # (N, 2)

    # presence loss — always active
    presence_loss = loss_per_timestep[:, 0].mean()

    # direction loss — weighted by whether turn is present
    turn_weight = 1.0 + target[:, 0] * 4.0  # 1.0 normally, 5.0 during turns
    direction_loss = (loss_per_timestep[:, 1] * turn_weight).mean()

    return presence_loss + direction_loss"""


def compute_f1(pred_prob, target, threshold=0.5):
    """F1 for turn_present only (col 0)."""
    pred_bin = (pred_prob[:, 0] > threshold).float()
    tgt      = target[:, 0]

    tp = ((pred_bin == 1) & (tgt == 1)).float().sum()
    fp = ((pred_bin == 1) & (tgt == 0)).float().sum()
    fn = ((pred_bin == 0) & (tgt == 1)).float().sum()

    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1.item(), p.item(), r.item()


def train(model, train_files, val_files):
    start_time = time.time()
    print("Training...")
    train_set = DriveDataset(train_files)
    val_set   = DriveDataset(val_files)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False)

    train_losses, val_losses, f1s, precisions, recalls = [], [], [], [], []
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # train
        model.train()
        t_loss = 0.0
        for x, y in train_loader:
            pred = model(x)
            loss = compute_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        train_losses.append(t_loss / len(train_loader))

        # validate
        model.eval()
        v_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                raw  = model(x)
                prob = torch.sigmoid(raw)
                v_loss += compute_loss(raw, y).item()
                all_preds.append(prob.squeeze(0))
                all_targets.append(y.squeeze(0))
        val_losses.append(v_loss / len(val_loader))

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        f1, p, r    = compute_f1(all_preds, all_targets)
        f1s.append(f1 * 100)
        precisions.append(p * 100)
        recalls.append(r * 100)

        # save best and early stopping — only after min_save_epoch
        if epoch >= min_save_epoch:
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save(model.state_dict(), 'bilstm_best_1d.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}, best F1: {best_f1*100:.1f}%")
                    break
                
        epoch_end_time = time.time()
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:3d}  "
                  f"train={train_losses[-1]:.3f}  "
                  f"val={val_losses[-1]:.3f}  "
                  f"F1={f1*100:.1f}%  P={p*100:.1f}%  R={r*100:.1f}% "
                  f"time={epoch_end_time - epoch_start_time:.1f}")

    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(train_losses, label='Train loss')
    ax1.plot(val_losses,   label='Val loss')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(f1s,        label='F1')
    #ax2.plot(precisions, label='Precision')
    #ax2.plot(recalls,    label='Recall')
    ax2.set_ylabel('%')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training.png')
    plt.show()
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time}s\n") 
    
    return best_f1


def evaluate(model, val_files):
    """Detailed evaluation on validation set after training."""
    model.load_state_dict(torch.load('bilstm_best_1d.pth'))
    model.eval()

    val_set    = DriveDataset(val_files)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            prob = torch.sigmoid(model(x))
            all_preds.append(prob.squeeze(0))
            all_targets.append(y.squeeze(0))

    all_preds   = torch.cat(all_preds,   dim=0)  # (N, 2)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 2)

    pred_bin = (all_preds > 0.5).float()

    print("\n--- Evaluation (best model) ---")
    print(f"Pred distribution:  turn={pred_bin[:,0].mean()*100:.1f}%  "
          f"left={((pred_bin[:,0]==1)&(pred_bin[:,1]==0)).float().mean()*100:.1f}%  "
          f"right={((pred_bin[:,0]==1)&(pred_bin[:,1]==1)).float().mean()*100:.1f}%")
    print(f"True distribution:  turn={all_targets[:,0].mean()*100:.1f}%  "
          f"left={((all_targets[:,0]==1)&(all_targets[:,1]==0)).float().mean()*100:.1f}%  "
          f"right={((all_targets[:,0]==1)&(all_targets[:,1]==1)).float().mean()*100:.1f}%")

    # turn_present metrics
    for col, name in [(0, 'turn_present')]:
        tp = ((pred_bin[:,col]==1) & (all_targets[:,col]==1)).float().sum()
        fp = ((pred_bin[:,col]==1) & (all_targets[:,col]==0)).float().sum()
        tn = ((pred_bin[:,col]==0) & (all_targets[:,col]==0)).float().sum()
        fn = ((pred_bin[:,col]==0) & (all_targets[:,col]==1)).float().sum()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2*p*r / (p + r + 1e-8)
        acc = (tp + tn) / (tp + fp + tn + fn + 1e-8)
        print(f"\n{name}:")
        print(f"  TP={tp:.0f}  FP={fp:.0f}  TN={tn:.0f}  FN={fn:.0f}")
        print(f"  Accuracy={acc*100:.1f}%  F1={f1:.2f}  P={p:.2f}  R={r:.2f}")

    # direction accuracy — only where turn is actually present
    turn_mask = all_targets[:, 0] > 0.5
    if turn_mask.any():
        correct_dir = (pred_bin[:,1][turn_mask] == all_targets[:,1][turn_mask]).float()
        print(f"\nturn_direction accuracy (where turn=1): {correct_dir.mean()*100:.1f}%")
    else:
        print("\nNo turns in validation set.")


def main():
    sample_files = sorted(Path('../input_data').glob('*_training.npz'))
    
    mode = 2 
    
    if mode == 1:
        f1s = []
        for seed in [11, 42, 85, 63, 97]:
            print("\n\nSEED: ", seed, "\n")
            random.seed(seed)
            files = list(sample_files)
            random.shuffle(files)
            val_count   = max(1, int(len(files) * 0.2))
            val_files   = files[:val_count]
            train_files = files[val_count:]
            
            x0, _ = load_sample(train_files[0])
            input_size = x0.shape[1]
            print(f"Files: {len(train_files)} train, {len(val_files)} val")
            print(f"Input size: {input_size}")
        
            model = BiLSTM(input_size)
            best = train(model, train_files, val_files)
            evaluate(model, val_files)
            print(f"Best F1: {best*100:.1f}%")
            f1s.append(best)
        
        print(f"OVERALL: avg f1: {sum(f1s)/len(f1s)}")
        
    else:
        seeds = []
        for i in range(5):
            seeds.append(random.randint(0, 200))
        
        f1s = []
        for seed in seeds:
            print("\n\nSEED: ", seed, "\n")
            random.seed(seed)
            files = list(sample_files)
            random.shuffle(files)
            val_count   = max(1, int(len(files) * 0.2))
            val_files   = files[:val_count]
            train_files = files[val_count:]
            
            x0, _ = load_sample(train_files[0])
            input_size = x0.shape[1]
            print(f"Files: {len(train_files)} train, {len(val_files)} val")
            print(f"Input size: {input_size}")
        
            model = BiLSTM(input_size)
            best = train(model, train_files, val_files)
            evaluate(model, val_files)
            print(f"Best F1: {best*100:.1f}%")
            f1s.append(best)
        
        print(f"OVERALL: avg f1: {sum(f1s)/len(f1s)}")


if __name__ == '__main__':
    main()