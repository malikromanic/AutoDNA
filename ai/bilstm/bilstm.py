# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:45:16 2026
 
@author: mihal
"""
 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
 
#hiperparametri
num_layers   = 2
hidden_size  = 64
num_outputs  = 7        
learning_rate = 0.0001
batch_size   = 1        
num_epochs   = 200


MAX_ANGLE_TURN = 100
MAX_ANGLE_HILL = 15  
 
 
def load_training_sample(filepath):
    """
    Load one training sample from a .npz file.
 
    :param filepath: Path to the .npz file with preprocessed sensor data.
    :returns: Tuple (accel, gyro, mag, y) where accel/gyro/mag are RGB
              spectrograms of shape (F, T, 3) normalized to [0, 1],
              and y is the label matrix of shape (T, 7).
    """
    data = np.load(filepath)
    accel = data['accel_rgb'].astype(np.float32) / 255.0   #(F, T, 3)
    gyro  = data['gyro_rgb'].astype(np.float32)  / 255.0
    mag   = data['mag_rgb'].astype(np.float32)   / 255.0
    y     = data['labels']                                 #(T, 7)
    
    #print(f"accel: {accel.shape}, gyro: {gyro.shape}, mag: {mag.shape}, y: {y.shape}")
 
    return accel, gyro, mag, y
 
 
def flatten_sensors(accel, gyro, mag):
    """
    Combine RGB spectrograms from all three sensors into one time series.
 
    Each sensor has shape (F, T, 3). The frequency and color dimensions
    are flattened into one vector and transposed to (T, F*3).
    All three sensors are concatenated along the time axis.
 
    :param accel: Accelerometer RGB spectrogram of shape (F, T, 3).
    :param gyro: Gyroscope RGB spectrogram of shape (F, T, 3).
    :param mag: Magnetometer RGB spectrogram of shape (F, T, 3).
    :returns: Matrix of shape (T, F*9) where each row represents
              one timestep with all sensor values.
    """
    F, T, _ = accel.shape
    accel_flat = accel.reshape(F*3, T).T
    gyro_flat  = gyro.reshape(F*3, T).T
    mag_flat   = mag.reshape(F*3, T).T

    def minmax(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return arr

    # scale each sensor independently to preserve relative differences
    return np.concatenate([minmax(accel_flat), minmax(gyro_flat), minmax(mag_flat)], axis=-1)
 
 
class DriveDataset(Dataset):
    """
    PyTorch Dataset for loading training samples from .npz files.
 
    Each sample is one drive sequence with all sensor data
    and labels for every timestep.
 
    :param sample_files: List of paths to .npz training files.
    """
 
    def __init__(self, sample_files):
        """
        Initialize the dataset with a list of files.
 
        :param sample_files: List or generator of paths to .npz files.
        """
        self.files = list(sample_files)
 
    def __len__(self):
        """
        Return the number of samples in the dataset.
 
        :returns: Number of .npz files.
        """
        return len(self.files)
 
    def __getitem__(self, idx):
        """
        Load and prepare one sample by index.
 
        :param idx: Index of the sample in the file list.
        :returns: Tuple (x, y) where x is a tensor of shape (T, input_size)
                  and y is a label tensor of shape (T, 7).
        """
        accel, gyro, mag, y = load_training_sample(self.files[idx])
        x = flatten_sensors(accel, gyro, mag)            
        return torch.tensor(x), torch.tensor(y)
    
 
class BiLSTM(nn.Module):
    """
    Bidirectional LSTM network for per-timestep classification of driving maneuvers.
 
    The network accepts a time series of sensor data and predicts 7 values
    for each timestep: turn presence and direction, hill presence and direction,
    their respective angles, and a straight driving indicator.
 
    Architecture: BiLSTM -> Dropout -> Linear -> Sigmoid
 
    :param input_size: Size of the input vector per timestep.
    """
 
    def __init__(self, input_size):
        """
        Initialize the BiLSTM model with the given input size.
 
        :param input_size: Number of input features per timestep
                           (F * 3 channels * 3 sensors).
        """
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_outputs)
 
    def forward(self, x):
        """
        Perform one forward pass through the network.
 
        :param x: Input tensor of shape (batch, T, input_size).
        :returns: Output tensor of shape (batch, T, 7) with values in [0, 1].
                  Outputs are:
                  - col 0: turn_present
                  - col 1: turn_dir (0=left, 1=right)
                  - col 2: turn_angle normalized to [0, 1]
                  - col 3: hill_present
                  - col 4: hill_dir (0=down, 1=up)
                  - col 5: hill_angle normalized to [0, 1]
                  - col 6: straight
        """
        out, _ = self.lstm(x)          
        out = self.dropout(out)
        out = self.fc(out)             
        #return torch.sigmoid(out)
        return out


def get_labeled_mask(target):
    """Returns True for timesteps that have at least one label."""
    # sum across all 7 outputs — if all zero, timestep is unlabeled
    return target.sum(dim=-1) > 0  # shape (batch, T)


def stratified_split(sample_files, val_ratio=0.2, seed=42):
    """Split files ensuring val set has similar class distribution to train."""
    random.seed(seed)
    
    # load label distribution per file
    file_distributions = []
    for f in sample_files:
        data = np.load(f)
        y = data['labels']
        labeled = y.sum(axis=1) > 0
        y_labeled = y[labeled]
        if len(y_labeled) == 0:
            dist = 0.0
        else:
            # use straight ratio as proxy — high = mostly straight, low = eventful
            dist = y_labeled[:, 6].mean()
        file_distributions.append((f, dist))
    
    # sort by straight ratio and interleave into val/train
    file_distributions.sort(key=lambda x: x[1])
    
    val_files = []
    train_files = []
    val_count = max(1, int(len(sample_files) * val_ratio))
    
    # pick every Nth file for val to ensure spread across distribution
    step = len(sample_files) // val_count
    for i, (f, dist) in enumerate(file_distributions):
        if i % step == 0 and len(val_files) < val_count:
            val_files.append(f)
        else:
            train_files.append(f)
    
    return train_files, val_files


def compute_pos_weights(dataset, max_weight=3.0):
    """
    Calculate pos_weight for each binary output from training data.
    Capped at max_weight to prevent extreme imbalance dominating loss.
    """
    all_labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labeled = get_labeled_mask(y.unsqueeze(0)).squeeze(0)
        all_labels.append(y[labeled])
    
    all_labels = torch.cat(all_labels, dim=0)
    
    weights = {}
    for col, name in [(0,'turn'), (3,'hill'), (6,'straight'), (1,'turn_dir'), (4,'hill_dir')]:
        pos = all_labels[:, col].sum()
        neg = (1 - all_labels[:, col]).sum()
        pw = neg / (pos + 1e-8)
        pw = min(max(pw.item(), 0.8), max_weight)  # keep between 0.8 and 3.0
        weights[name] = torch.tensor([pw])
        print(f"  pos_weight {name}: {pw:.2f}  (pos={pos:.0f}, neg={neg:.0f})")
    
    return weights


def masked_loss(pred, target, pw):
    def bce_turn(p, t):
        return nn.BCEWithLogitsLoss(pos_weight=pw['turn'])(p, t)
    def bce_hill(p, t):
        return nn.BCEWithLogitsLoss(pos_weight=pw['hill'])(p, t)
    def bce_straight(p, t):
        return nn.BCEWithLogitsLoss(pos_weight=pw['straight'])(p, t)
    def bce_dir(p, t):
        return nn.BCEWithLogitsLoss(pos_weight=pw['turn_dir'])(p, t)

    # L1 instead of MSE for angle regression — more robust to outliers
    l1 = nn.L1Loss()

    pred_prob = torch.sigmoid(pred)

    labeled = get_labeled_mask(target)
    if not labeled.any():
        return torch.tensor(0.0, requires_grad=True)

    pred_l      = pred[labeled]
    target_l    = target[labeled]
    pred_prob_l = pred_prob[labeled]

    loss = bce_turn(pred_l[:, 0],      target_l[:, 0])
    loss += bce_hill(pred_l[:, 3],     target_l[:, 3])
    loss += bce_straight(pred_l[:, 6], target_l[:, 6])

    turn_mask = target_l[:, 0] > 0.5
    if turn_mask.any():
        loss += bce_dir(pred_l[:, 1][turn_mask], target_l[:, 1][turn_mask])

    hill_mask = target_l[:, 3] > 0.5
    if hill_mask.any():
        loss += bce_dir(pred_l[:, 4][hill_mask], target_l[:, 4][hill_mask])

    # L1 loss with higher weight so angle regression competes with presence losses
    if turn_mask.any():
        loss += 3.0 * l1(pred_prob_l[:, 2][turn_mask], target_l[:, 2][turn_mask])
    if hill_mask.any():
        loss += 3.0 * l1(pred_prob_l[:, 5][hill_mask], target_l[:, 5][hill_mask])

    return loss


def train_model(model, train_set, val_set, print_info):
    if print_info:
        print("Training...\n")
        
    pw = compute_pos_weights(train_set)
    
        
    start_time = time.time()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5  # mode='max' because we track accuracy
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)  
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False) 

    train_losses = []
    val_losses = []
    accs = []
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 25

    for epoch in range(num_epochs):   
        model.train()
        train_loss_sum = 0.0
        for x, y in train_loader:
            pred = model(x)
            loss = masked_loss(pred, y, pw)
            optimizer.zero_grad()       
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        train_losses.append(train_loss_sum / len(train_loader))
        
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                raw = model(x)
                pred = torch.sigmoid(raw)
                all_preds.append(pred.squeeze(0))
                all_targets.append(y.squeeze(0))
                loss = masked_loss(raw, y, pw)
                val_loss_sum += loss.item()
        
        val_losses.append(val_loss_sum / len(val_loader))
        current_acc = measure_epoch_acc(all_preds, all_targets)
        accs.append(current_acc)

        # lr scheduler steps on accuracy
        scheduler.step(current_acc)

        # early stopping on accuracy
        if current_acc > best_val_acc:
            best_val_acc = current_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'bilstm_best.pth')  # save best
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if print_info:
                    print(f"Early stopping at epoch {epoch+1}, best acc: {best_val_acc:.1f}%")
                break
        
    end_time = time.time()
    
    if print_info:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(train_losses, label='Train loss')
        ax1.plot(val_losses, label='Val loss')
        ax1.set_xlabel('Epoha')
        ax1.set_ylabel('Izguba')
        ax1.legend()

        ax2.plot(accs, label='Val accuracy')
        ax2.set_xlabel('Epoha')
        ax2.set_ylabel('Točnost (%)')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('training.png')
        plt.show()
        plt.close()

        print(f"Best val accuracy: {best_val_acc:.1f}%")
        print(f"Last train loss: {train_losses[-1]}")
        print(f"Last validation loss: {val_losses[-1]}")
        elapsed_time = round(end_time - start_time, 2)
        print(f"Training time: {elapsed_time}s\n") 
    
    return train_losses, val_losses, accs


def measure_epoch_acc(all_preds, all_targets):
    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    labeled = get_labeled_mask(all_targets)
    all_preds   = all_preds[labeled]
    all_targets = all_targets[labeled]

    pred_binary = (all_preds > 0.5).float()

    """pct_turn     = pred_binary[:, 0].mean().item() * 100
    pct_hill     = pred_binary[:, 3].mean().item() * 100
    pct_straight = pred_binary[:, 6].mean().item() * 100
    #print(f"  pred: turn={pct_turn:.1f}%  hill={pct_hill:.1f}%  straight={pct_straight:.1f}%")

    true_turn     = all_targets[:, 0].mean().item() * 100
    true_hill     = all_targets[:, 3].mean().item() * 100
    true_straight = all_targets[:, 6].mean().item() * 100
    #print(f"  true: turn={true_turn:.1f}%  hill={true_hill:.1f}%  straight={true_straight:.1f}%")"""

    # F1 instead of accuracy — penalizes degenerate all-zero or all-one predictions
    f1s = []
    for col in [0, 3, 6]:
        tp = ((pred_binary[:, col] == 1) & (all_targets[:, col] == 1)).float().sum()
        fp = ((pred_binary[:, col] == 1) & (all_targets[:, col] == 0)).float().sum()
        fn = ((pred_binary[:, col] == 0) & (all_targets[:, col] == 1)).float().sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1.item())

    return np.mean(f1s) * 100
 
 
def into_deterministic(pred):
    """
    Convert raw model predictions into deterministic decisions.

    Applies sigmoid to convert raw logits to probabilities,
    then thresholds binary outputs to 0 or 1.
    Angle outputs are converted from normalized values to degrees.

    :param pred: Raw logit tensor of shape (batch, T, 7).
    :returns: Tensor of shape (batch, T, 7) with binary decisions
              and angles in degrees.
    """
    pred_prob = torch.sigmoid(pred)

    pred_det = (pred_prob > 0.5).float()
    pred_det[:, :, 2] = pred_prob[:, :, 2] * MAX_ANGLE_TURN
    pred_det[:, :, 5] = pred_prob[:, :, 5] * MAX_ANGLE_HILL

    return pred_det
        
        
def evaluate_model(model, validation_files):
    """
    Run detailed evaluation of the model on validation files.
 
    Measures and prints:
    - Presence accuracy for turn, hill and straight.
    - Direction accuracy for turn and hill (only where events are present).
    - Mean absolute angle error for turn and hill in degrees.
 
    :param model: Trained BiLSTM model in eval mode.
    :param validation_files: List of paths to validation .npz files.
    """
    print("Evaluating best saved model (bilstm_best.pth)")
    val_dataset = DriveDataset(validation_files)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model.eval() 
    all_preds = []
    all_targets = []
    with torch.no_grad():  
        for x, y in val_loader:
            raw = model(x)
            pred = torch.sigmoid(raw)  #v 0-1 za predictione
            all_preds.append(pred.squeeze(0))    
            all_targets.append(y.squeeze(0))     
    
    all_preds   = torch.cat(all_preds,   dim=0) 
    all_targets = torch.cat(all_targets, dim=0)  
    labeled = get_labeled_mask(all_targets)
    all_preds   = all_preds[labeled]
    all_targets = all_targets[labeled]
    
    pred_binary = (all_preds > 0.5).float()
    print("Best model pred distribution:")
    print(f"  turn={pred_binary[:,0].mean()*100:.1f}%  hill={pred_binary[:,3].mean()*100:.1f}%  straight={pred_binary[:,6].mean()*100:.1f}%")

    print(f"After mask — turn: {all_targets[:,0].mean()*100:.1f}%  hill: {all_targets[:,3].mean()*100:.1f}%  straight: {all_targets[:,6].mean()*100:.1f}%")
    print(f"Total labeled timesteps: {labeled.sum()}")

    pred_binary = (all_preds > 0.5).float()
    pred_binary[:, 2] = all_preds[:, 2]  # restore turn angle
    pred_binary[:, 5] = all_preds[:, 5]  # restore hill angle
    
    for col, name in [(0, 'turn_present'), (3, 'hill_present'), (6, 'straight')]:
        tp = ((pred_binary[:, col] == 1) & (all_targets[:, col] == 1)).float().sum()
        fp = ((pred_binary[:, col] == 1) & (all_targets[:, col] == 0)).float().sum()
        fn = ((pred_binary[:, col] == 0) & (all_targets[:, col] == 1)).float().sum()
        correct = (pred_binary[:, col] == all_targets[:, col]).float()
        
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy  = correct.mean().item() * 100
        
        print(f"{name}: acc={accuracy:.1f}%  F1={f1:.2f}  P={precision:.2f}  R={recall:.2f}")
    
    #print("num of left turns:  ", len(np.where(pred_binary[:, 1] == 0)[0]))
    #print("num of right turns: ", len(np.where(pred_binary[:, 1] == 1)[0]))
    
    for col, name in [(0, 'turn_present'), (3, 'hill_present'), (6, 'straight')]:
        correct = (pred_binary[:, col] == all_targets[:, col]).float()
        accuracy = correct.mean().item() * 100
        print(f"{name} accuracy: {accuracy:.1f}%")
 
    turn_mask = all_targets[:, 0] > 0.5
    if turn_mask.any():
        correct = (pred_binary[:, 1][turn_mask] == all_targets[:, 1][turn_mask]).float()
        print(f"turn_direction accuracy: {correct.mean().item()*100:.1f}%")
 
    hill_mask = all_targets[:, 3] > 0.5
    if hill_mask.any():
        correct = (pred_binary[:, 4][hill_mask] == all_targets[:, 4][hill_mask]).float()
        print(f"hill_direction accuracy: {correct.mean().item()*100:.1f}%")
 
    if turn_mask.any():
        mae = (pred_binary[:, 2][turn_mask] - all_targets[:, 2][turn_mask]).abs().mean()
        print(f"turn_angle average error: {mae.item() * MAX_ANGLE_TURN:.1f} degrees")
 
    if hill_mask.any():
        mae = (pred_binary[:, 5][hill_mask] - all_targets[:, 5][hill_mask]).abs().mean()
        print(f"hill_angle average error: {mae.item() * MAX_ANGLE_HILL:.1f} degrees")
 
    
def train_once(sample_files, val_ratio=0.2, seed=42):
    if seed is not None:
        random.seed(seed)

    files = sample_files.copy()
    """random.shuffle(files)

    val_count = max(1, int(len(files) * val_ratio))
    val_files = files[:val_count]
    train_files = files[val_count:]"""
    
    train_files, val_files = stratified_split(files, val_ratio, seed)
    
    accel, gyro, mag, _ = load_training_sample(sample_files[0])
    F = accel.shape[0]
    input_size = F * 3 * 3

    train_set = DriveDataset(train_files)
    val_set = DriveDataset(val_files)
    model = BiLSTM(input_size)
    
    train_model(model, train_set, val_set, print_info=True)

    model.load_state_dict(torch.load('bilstm_best.pth'))
    torch.save(model.state_dict(), 'bilstm.pth')  # copy best to final
    
    print("Val files for this seed:")
    for f in val_files:
        data = np.load(f)
        y = data['labels']
        labeled = y.sum(axis=1) > 0
        y_l = y[labeled]
        print(f"  {f.name}: turn={y_l[:,0].mean()*100:.0f}%  hill={y_l[:,3].mean()*100:.0f}%  straight={y_l[:,6].mean()*100:.0f}%")
    evaluate_model(model, val_files)
    
    
def load_and_test(sample_files, val_ratio=0.2, seed=42):
    files = sample_files.copy()
    train_files, val_files = stratified_split(files, val_ratio, seed)  # match train_once
    
    accel, gyro, mag, _ = load_training_sample(sample_files[0])
    F = accel.shape[0]
    input_size = F * 3 * 3

    loaded_model = BiLSTM(input_size)
    state_dict = torch.load("bilstm.pth")
    loaded_model.load_state_dict(state_dict)
    evaluate_model(loaded_model, val_files)
    
    
def main():
    """
    Main entry point of the program.
 
    Loads all training files and offers a choice between three modes:
    1. Train the model once on a fixed split.
    3. Load and test a previously saved model.
    """
    
    sample_files = list(Path('../input_data').glob('*_training.npz'))
    mode = int(input("Choose mode:\n\n 1: Train model once\n 2: Train model with multiple seeds\n 3: Load and test saved model\n\n Type your choice: "))
    
    SEED = 123
    if mode == 1:
        train_once(sample_files, seed=SEED)
    elif mode == 2:
        for seed in [42, 123, 7, 99, 17]:
            print(f"\n--- Seed {seed} ---")
            train_once(sample_files, seed=seed)
            print("\n\n")
    elif mode == 3:
        load_and_test(sample_files, seed=SEED)
 
 
if __name__ == "__main__":
    main()