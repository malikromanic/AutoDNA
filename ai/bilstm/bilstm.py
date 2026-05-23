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
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
 
#hiperparametri
num_layers   = 2
hidden_size  = 64
num_outputs  = 7        
learning_rate = 0.001
batch_size   = 1        
num_epochs   = 50
MAX_ANGLE_TURN = 180.0  
MAX_ANGLE_HILL = 45.0  
 
 
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
        #out = self.dropout(out)
        out = self.fc(out)             
        return torch.sigmoid(out)     
 
 
def masked_loss(pred, target):
    """
    Combined loss for multi-label and regression outputs.
 
    Uses BCE for presence, direction and straight driving outputs.
    Uses MSE for turn and hill angle, but only where the corresponding
    event is present (masked computation).
 
    :param pred: Model predictions of shape (batch, T, 7).
    :param target: Ground truth labels of shape (batch, T, 7).
    :returns: Scalar tensor of total loss.
    """
    bce = nn.BCELoss()
    mse = nn.MSELoss()
 
    loss = bce(pred[:, :, 0], target[:, :, 0])   # turn_present
    loss += bce(pred[:, :, 3], target[:, :, 3])   # hill_present
    loss += bce(pred[:, :, 6], target[:, :, 6])   # straight
 
    turn_mask = target[:, :, 0] > 0.5
    if turn_mask.any():
        loss += bce(pred[:, :, 1][turn_mask], target[:, :, 1][turn_mask])
 
    hill_mask = target[:, :, 3] > 0.5
    if hill_mask.any():
        loss += bce(pred[:, :, 4][hill_mask], target[:, :, 4][hill_mask])
 
    if turn_mask.any():
        loss += mse(pred[:, :, 2][turn_mask], target[:, :, 2][turn_mask])
 
    if hill_mask.any():
        loss += mse(pred[:, :, 5][hill_mask], target[:, :, 5][hill_mask])
 
    return loss
 
 
def train_model(model, train_set, val_set, print_info):
    """
    Train the BiLSTM model while tracking validation loss and accuracy each epoch.
 
    For each epoch performs:
    1. Training step with backpropagation on train_set.
    2. Evaluation without weight updates on val_set.
    3. Measurement of average event presence accuracy.
 
    If print_info=True, plots loss and accuracy graphs and
    prints training time.
 
    :param model: Initialized BiLSTM model.
    :param train_set: DriveDataset with training samples.
    :param val_set: DriveDataset with validation samples.
    :param print_info: If True, plots graphs and prints statistics.
    :returns: Tuple (train_losses, val_losses, accs) — lists of
              length num_epochs with per-epoch values.
    """
    if print_info:
        print("Training...\n")
        
    start_time = time.time()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)  #weight deacy = l2
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)  
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False) 
 
    train_losses = []
    val_losses = []
    accs = []
    
    for epoch in range(num_epochs):   
        model.train()
        train_loss_sum = 0.0
        for x, y in train_loader:       #treniranje
            pred = model(x)             #forward
            loss = masked_loss(pred, y)
 
            optimizer.zero_grad()       
            loss.backward()             #backward
            optimizer.step()
            train_loss_sum += loss.item()
 
        train_losses.append(train_loss_sum / len(train_loader))   #avg ker so batchi lahko razlicno veliki
        
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():               #evaluacija
            for x, y in val_loader:
                pred = model(x)             #samo forward
                all_preds.append(pred.squeeze(0))    #za accuracy
                all_targets.append(y.squeeze(0))    
                loss = masked_loss(pred, y)
                val_loss_sum += loss.item()
        
        val_losses.append(val_loss_sum / len(val_loader))   #avg ker so batchi lahko razlicno veliki
        accs.append(measure_epoch_acc(all_preds, all_targets))
        
    end_time = time.time()
    
    if print_info:
        plt.figure()
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses,   label='Val loss')
        plt.xlabel('Epoha')
        plt.ylabel('Izguba')
        plt.legend()
        plt.show()
        plt.close()
        print(f"Last train loss: {train_losses[-1]}")
        print(f"Last validation loss: {val_losses[-1]}")
        
        plt.figure()
        plt.plot(accs)
        plt.xlabel('Epoha')
        plt.ylabel('Točnost (%)')
        plt.show()
        
        elapsed_time = round(end_time - start_time, 2)
        print(f"Training time for {num_epochs} epochs: {elapsed_time}s\n") 
    
    return train_losses, val_losses, accs    #len = num_epochs
 
 
def measure_epoch_acc(all_preds, all_targets):
    """
    Measure average event presence accuracy for one epoch.
 
    Compares binary predictions (threshold 0.5) against ground truth
    for turn_present, hill_present and straight (columns 0, 3, 6).
    Returns the average accuracy across all three as a percentage.
 
    :param all_preds: List of prediction tensors from the validation loop.
    :param all_targets: List of ground truth label tensors.
    :returns: Average accuracy as a percentage (0-100).
    """
    all_preds   = torch.cat(all_preds,   dim=0) 
    all_targets = torch.cat(all_targets, dim=0) 
    
    pred_binary = (all_preds > 0.5).float()  #vse v 0 ali 1 -> dobimo predictione
 
    accs = []
    for col in [0, 3, 6]:         # turn_present, hill_present, straight
        correct = (pred_binary[:, col] == all_targets[:, col]).float()
        accs.append(correct.mean().item())
        
    return np.mean(accs) * 100  #accuracy je povprecje vseh predictionov
 
 
def into_deterministic(pred):
    """
    Convert raw model predictions into deterministic decisions.
 
    Binary outputs (presence, direction, straight) are thresholded to 0 or 1
    using a threshold of 0.5. Angle outputs (columns 2 and 5) are converted
    from normalized values back to actual degrees.
 
    :param pred: Prediction tensor of shape (batch, T, 7) with values in [0, 1].
    :returns: Tensor of shape (batch, T, 7) with binary decisions
              and angles in degrees.
    """
    print(pred.shape)
    
    pred_det = (pred > 0.5).float()                          #vse odlocilne pretvorimo v odlocitev
    pred_det[:, :, 2] = pred[:, :, 2] * MAX_ANGLE_TURN       #turn in hill angle v kot
    pred_det[:, :, 5] = pred[:, :, 5] * MAX_ANGLE_HILL
 
 
def predict(x, input_size):
    """
    Load a saved model and run inference on input data.
 
    Loads weights from bilstm.pth, switches the model to eval mode and
    performs a forward pass without computing gradients.
    Converts predictions to deterministic decisions via into_deterministic.
 
    :param x: Input tensor of shape (1, T, input_size).
    :param input_size: Input vector size — must match the saved model.
    :returns: Deterministic predictions of shape (1, T, 7).
    """
    model = BiLSTM(input_size)
    model.load_state_dict(torch.load('bilstm.pth'))     #loada shranjen nateniran model
    model.eval()        #mode za predictione
    
    with torch.no_grad():  
        pred = model(x)
        pred = into_deterministic(pred)
        
        
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
    val_dataset = DriveDataset(validation_files)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model.eval() 
    all_preds = []
    all_targets = []
    with torch.no_grad():  
        for x, y in val_loader:
            pred = model(x)  
            all_preds.append(pred.squeeze(0))    
            all_targets.append(y.squeeze(0))     
    
    all_preds   = torch.cat(all_preds,   dim=0) 
    all_targets = torch.cat(all_targets, dim=0)  
    
    pred_binary = (all_preds > 0.5).float()
    pred_binary[:, 2] = all_preds[:, 2]  # restore turn angle
    pred_binary[:, 5] = all_preds[:, 5]  # restore hill angle
    
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
 
 
def train_once(sample_files):
    """
    Train the model once on a fixed train/val file split.
 
    The last two files (indices 10 and 11) are used as the validation set,
    all others as the training set.
    After training, saves the model to bilstm.pth and runs evaluation.
 
    :param sample_files: List of all available .npz files.
    """
    train_files = sample_files.copy()
    val1 = train_files.pop(11)
    val2 = train_files.pop(10)
    
    val_files = []
    val_files.append(val1)
    val_files.append(val2)
    
    accel, gyro, mag, _ = load_training_sample(sample_files[0])
    F = accel.shape[0]
    input_size = F * 3 * 3   # x,y,z * acc,gy,mag
 
    train_set = DriveDataset(train_files)
    val_set = DriveDataset(val_files)
    model = BiLSTM(input_size)
    
    train_model(model, train_set, val_set, print_info=True)
    torch.save(model.state_dict(), "bilstm.pth")    
    evaluate_model(model, val_files)
    
    
def train_ksplit(sample_files):
    """
    Train the model using k-fold cross validation (7 folds).
 
    For each fold initializes a new model, trains it and stores
    losses and accuracies. At the end plots average loss and accuracy
    curves across all folds and prints overall average metrics.
 
    K-fold reduces the impact of a specific split on results and makes
    better use of a small dataset.
 
    :param sample_files: List of all available .npz files.
    """
    print("Training...\n")
    kf = KFold(n_splits=7, shuffle=True)
    all_t_losses = []
    all_v_losses = [] 
    all_accs = []
    for train_idx, val_idx in kf.split(sample_files):
        train_files = [sample_files[i] for i in train_idx]
        val_files   = [sample_files[i] for i in val_idx]
        
        accel, gyro, mag, _ = load_training_sample(sample_files[0])
        F = accel.shape[0]
        input_size = F * 3 * 3   # x,y,z * acc,gy,mag
 
        train_set = DriveDataset(train_files)
        val_set = DriveDataset(val_files)
        model = BiLSTM(input_size)
        t_losses, v_losses, accs = train_model(model, train_set, val_set, print_info=False)
        all_t_losses.append(t_losses)
        all_v_losses.append(v_losses)
        all_accs.append(accs)
    
    plt.figure()
    plt.plot(np.mean(all_t_losses, axis=0), label='Train loss')
    plt.plot(np.mean(all_v_losses, axis=0),   label='Val loss')
    plt.xlabel('Epoha')
    plt.ylabel('Izguba')
    plt.legend()
    plt.show()
    plt.close()
     
    plt.figure()
    plt.plot(np.mean(all_accs, axis=0))
    plt.xlabel('Epoha')
    plt.ylabel('Točnost (%)')
    plt.show()   
            
    avg_t_loss = np.mean(all_t_losses)
    avg_v_loss = np.mean(all_v_losses)
    avg_acc = np.mean(all_accs)
    print(f"Avg train loss: {avg_t_loss}")
    print(f"Avg validation loss: {avg_v_loss}")
    print(f"Avg accuracy: {avg_acc}")
    
    
def load_and_test(sample_files):
    """
    Load a saved model from bilstm.pth and evaluate it on the validation set.
 
    Uses the same fixed split as train_once (indices 10 and 11 as validation).
    Useful for testing an already trained model without retraining.
 
    :param sample_files: List of all available .npz files.
    """
    train_files = sample_files.copy()
    val1 = train_files.pop(11)
    val2 = train_files.pop(10)
    
    val_files = []
    val_files.append(val1)
    val_files.append(val2)
    
    accel, gyro, mag, _ = load_training_sample(sample_files[0])
    F = accel.shape[0]
    input_size = F * 3 * 3   # x,y,z * acc,gy,mag
    
    loaded_model = BiLSTM(input_size) 
    state_dict = torch.load("bilstm.pth")
    loaded_model.load_state_dict(state_dict)    
    evaluate_model(loaded_model, val_files)
    
    
def main():
    """
    Main entry point of the program.
 
    Loads all training files and offers a choice between three modes:
    1. Train the model once on a fixed split.
    2. Train the model with 7-fold cross validation.
    3. Load and test a previously saved model.
    """
    sample_files = list(Path('../input_data').glob('*_training.npz'))
    
    mode = int(input("Choose mode:\n\n 1: Train model once\n 2: Train model with ksplit\n 3: Load and test saved model\n\n Type your choice: "))
    
    if mode == 1:
        train_once(sample_files)
    elif mode == 2:
        train_ksplit(sample_files)
    elif mode == 3:
        load_and_test(sample_files)
 
 
if __name__ == "__main__":
    main()