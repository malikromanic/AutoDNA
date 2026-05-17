from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


#hiperparametri modela
NUM_LAYERS = 2
HIDDEN_SIZE = 128
NUM_CLASSES = 5
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


def load_sample(filepath):
    #nalozi en npz sample
    data = np.load(filepath)

    accel = data["accel_rgb"]
    gyro = data["gyro_rgb"]
    mag = data["mag_rgb"]
    y = data["labels"]

    return accel, gyro, mag, y


def prepare(accel, gyro, mag, y):
    #flatten senzorje v casovno vrsto
    def flatten(x):
        f, t, c = x.shape
        return x.reshape(f * c, t).T

    accel = flatten(accel)
    gyro = flatten(gyro)
    mag = flatten(mag)

    #uskladi dolzine signalov
    min_len = min(len(accel), len(gyro), len(mag), len(y))

    accel = accel[:min_len]
    gyro = gyro[:min_len]
    mag = mag[:min_len]
    y = y[:min_len]

    #zdruzi vse senzorje v en input vektor
    x = np.concatenate([accel, gyro, mag], axis=1)

    #pretvori one hot v class label
    labels = np.argmax(y, axis=1)
    label = np.bincount(labels).argmax()

    return x.astype(np.float32), label


def load_data(folder):
    #nalozi vse sample iz dataset mape
    x_list = []
    y_list = []

    for f in sorted(Path(folder).glob("*.npz")):
        accel, gyro, mag, y = load_sample(f)
        x, label = prepare(accel, gyro, mag, y)

        x_list.append(x)
        y_list.append(label)

        print(f"loaded {f.name}")

    return x_list, y_list


def pad(x_list):
    #padding sekvenc na isto dolzino za batch processing
    max_len = max(len(x) for x in x_list)
    dim = x_list[0].shape[1]

    out = np.zeros((len(x_list), max_len, dim), dtype=np.float32)

    for i, x in enumerate(x_list):
        out[i, :len(x)] = x

    return out


def train_val_split(x, y):
    #random shuffle in split za train in validation set
    idx = np.arange(len(x))
    np.random.shuffle(idx)

    split = int(len(x) * 0.8)

    train_idx = idx[:split]
    val_idx = idx[split:]

    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]


class GRUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        #gru arhitektura za casovne sekvence
        self.gru = nn.GRU(
            input_size,
            HIDDEN_SIZE,
            NUM_LAYERS,
            batch_first=True
        )

        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        #inicijalizacija hidden stanja
        #pomembno: mora biti na istem device kot input
        device = x.device
        h = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE, device=device)

        out, _ = self.gru(x, h)

        #uporabimo zadnji timestep kot reprezentacijo sekvence
        out = out[:, -1]

        return self.fc(out)


def train(model, x_train, y_train, x_val, y_val):
    #loss funkcija in optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    #tensor pretvorba (enkrat, da ne delamo vsako epoch)
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_val = torch.tensor(x_val)
    y_val = torch.tensor(y_val)

    for epoch in range(NUM_EPOCHS):
        model.train()

        #train forward pass
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()

        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = loss_fn(val_pred, y_val)

            #validation accuracy
            acc = (val_pred.argmax(1) == y_val).float().mean().item()

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        #validation loss je pomemben indikator
        #ce train loss pada, val loss pa raste -> overfitting
        #ce oba padata ->> model se dobro generalizira
        print(
            f"epoch {epoch + 1} "
            f"train loss {loss.item():.4f} "
            f"val loss {val_loss.item():.4f} "
            f"val acc {acc:.4f}"
        )

    return train_losses, val_losses


def plot(train, val):
    #primerjava train vs validation loss
    plt.plot(train)
    plt.plot(val)
    plt.legend(["train", "val"])
    plt.show()


def main():
    #reproducibilnost rezultatov
    np.random.seed(42)
    torch.manual_seed(42)

    folder = Path(__file__).parent / "training_data"

    x_list, y_list = load_data(folder)

    x = pad(x_list)
    y = np.array(y_list)

    x_train, x_val, y_train, y_val = train_val_split(x, y)

    model = GRUModel(x.shape[2])

    train_losses, val_losses = train(
        model,
        x_train,
        y_train,
        x_val,
        y_val
    )

    plot(train_losses, val_losses)


if __name__ == "__main__":
    main()