import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import os

# ======== Device Configuration ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=== Using device: {device} ===")


# ======== Configuration ========
class Config:
    dataset = 'cifar10'  # 'mnist' or 'cifar10'
    type_rnn = 'diagonal'  # 'row' for Row LSTM, 'diagonal' for Diaganal BiLSTM
    batch_size_train = 16  # 16
    batch_size_test = 128  # 128
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    epochs = 500
    lr = 5*1e-3
    L=1e-6 # coefficient Regularization
    dropout=0.3
    step_size = 10 #lr decay step size
    lr_decay = 0.8
    patience=10


# ======== TensorBoard Setup ========
experiment_early_stop = f"lr_{Config.lr}_dropout_{Config.dropout}_decay_{Config.lr_decay}"
log_dir = f"/runs/{experiment_early_stop}"
writer = SummaryWriter(log_dir=log_dir)


# ======== Masked Convolution ========
class MaskedConv1d(nn.Conv1d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, device=device):
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)

        kernel_Width = kernel_size
        self.mask = torch.ones(out_channels, in_channels, kernel_Width).to(device)
        self.mask[:, :, kernel_Width // 2 + (mask_type == 'B'):] = 0

        self.device = device
        self.to(self.device)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=1, device=device):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)

        kernel_Height, kernel_Width = kernel_size, kernel_size
        self.mask = torch.ones(out_channels, in_channels, kernel_Height, kernel_Width).to(device)

        # Masking rule for RGB images
        if mask_type == 'A':  # Mask A (Used in First Layer)
            self.mask[:, :, kernel_Height // 2, kernel_Width // 2:] = 0  # Right part of the center pixel
            self.mask[:, :, kernel_Height // 2 + 1:] = 0  # Rows below
            self.mask[:, :, kernel_Height // 2, kernel_Width // 2] = 0  # Mask the pixel itself

        elif mask_type == 'B':  # Mask B (Used in Later Layers)
            self.mask[:, :, kernel_Height // 2, kernel_Width // 2 + 1:] = 0
            self.mask[:, :, kernel_Height // 2 + 1:] = 0

        self.device = device
        self.to(self.device)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


# ======== Row LSTM ========
class RowLSTMLayer(nn.Module):
    def __init__(self, in_channels, hidden_dim, image_size, device=device):
        super(RowLSTMLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.image_size = 32 if Config.dataset == "cifar10" else 28
        self.in_channels = in_channels
        self.num_units = self.hidden_dim * self.image_size
        self.output_size = self.num_units
        self.state_size = self.num_units * 2

        self.conv_i_s = MaskedConv1d('B', hidden_dim, 4 * self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_s_s = nn.Conv1d(in_channels, 4 * self.hidden_dim, kernel_size=3, padding=1, stride=1)

        self.device = device
        self.to(self.device)

    def forward(self, inputs, states):
        c_prev, h_prev = states

        h_prev = h_prev.view(-1, self.hidden_dim, self.image_size)
        B, C, _, W = inputs.shape  
        inputs = inputs.view(B, C, W)  # (Batch, Channels, Width)

        s_s = self.conv_s_s(h_prev)
        i_s = self.conv_i_s(inputs)

        s_s = s_s.view(-1, 4 * self.num_units)
        i_s = i_s.view(-1, 4 * self.num_units)

        lstm = s_s + i_s
        lstm = torch.sigmoid(lstm)

        i, g, f, o = torch.split(lstm, (4 * self.num_units) // 4, dim=1)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        new_state = (c, h)
        return h, new_state


class RowLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim, input_size, device=device):
        super(RowLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.image_size = 32 if Config.dataset == "cifar10" else 28
        self.init_state = (torch.zeros(1, self.image_size * hidden_dim).to(device),
                           torch.zeros(1, self.image_size * hidden_dim).to(device))
        self.lstm_layer = RowLSTMLayer(in_channels, hidden_dim, self.image_size)

        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, padding=0)
        self.device = device
        self.to(self.device)

    def forward(self, inputs, initial_state=None):
        n_batch, channel, height, width = inputs.size()
        if initial_state is None:
            hidden_init, cell_init = self.init_state
        else:
            hidden_init, cell_init = initial_state

        states = (hidden_init.repeat(n_batch, 1), cell_init.repeat(n_batch, 1))

        steps = []
        for row in range(height):
            row_input = inputs[:, :, row, :]  # Shape should be (B, C, W)
            if row_input.dim() == 3:
                row_input = row_input.unsqueeze(2)  # Ensure (B, C, 1, W)

            h, states = self.lstm_layer(row_input, states)
            steps.append(h.unsqueeze(1))

        output = torch.cat(steps, dim=1)
        output = output.view(n_batch, height, width, self.hidden_dim)
        output = output.permute(0, 3, 1, 2)  #(Batch, Hidden, Height, Width)

        # Residual connection
        residual = self.residual_conv(inputs)
        output = output + residual

        return output


# ======== Diagonal BiLSTM ========
def skew(tensor):
    B, C, H, W = tensor.shape
    output = tensor.new_zeros((B, C, H, H + W - 1))
    for row in range(H):
        columns = (row, row + W)
        output[:, :, row, columns[0]:columns[1]] = tensor[:, :, row, :]
    return output


def unskew(tensor):
    B, C, H, skew_W = tensor.shape
    W = skew_W - H + 1
    output = tensor.new_zeros((B, C, H, W))
    for row in range(H):
        columns = (row, row + W)
        output[:, :, row, :] = tensor[:, :, row, columns[0]:columns[1]]
    return output


class DiagonalLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim, device=device):
        super(DiagonalLSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.conv_is = MaskedConv2d(mask_type='B', in_channels=in_channels, out_channels=5 * hidden_dim, kernel_size=1,
                                    padding=0)
        self.conv_ss = nn.Conv1d(hidden_dim, 5 * hidden_dim, 2, padding=1)
        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(p=Config.dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.device = device
        self.to(self.device)

    def forward(self, inputs):
        skewed_input = skew(inputs)
        B, C, H, skew_W = skewed_input.shape

        i_s = self.conv_is(skewed_input)

        hStates = []
        cStates = []

        h_prev = skewed_input.new_zeros([B, self.hidden_dim, H])
        c_prev = skewed_input.new_zeros([B, self.hidden_dim, H])

        for i in range(skew_W):
            input_column = skewed_input[..., i]
            s_s = self.conv_ss(h_prev)[..., :-1]
            gates = i_s[..., i] + s_s

            o, f_left, f_up, i, g = torch.chunk(gates, 5, dim=1)
            o, f_left, f_up, i, g = self.sigmoid(o), self.sigmoid(f_left), self.sigmoid(f_up), self.sigmoid(
                i), self.tanh(g)

            c_prev_shifted = torch.cat([input_column.new_zeros([input_column.shape[0], self.hidden_dim, 1]), c_prev],
                                       2)[..., :-1]
            c = (f_left * c_prev + f_up * c_prev_shifted) + i * g
            h = o * self.tanh(c)

            h = self.dropout(h)

            hStates.append(h)
            cStates.append(c)

            h_prev = h
            c_prev = c

        total_hStates = unskew(torch.stack(hStates, dim=3))
        total_cStates = unskew(torch.stack(cStates, dim=3))

        # Adding residual connection
        residual = self.residual_conv(inputs)
        total_hStates = total_hStates + residual

        return total_hStates


# ======== Pixel RNN Architecture ========
class PixelRNN_model(nn.Module):
    def __init__(self, num_layers, num_filters, input_size, type_rnn='row', device=device):
        super(PixelRNN_model, self).__init__()

        self.conv1 = self.conv1 = MaskedConv2d('A', 3 if Config.dataset == 'cifar10' else 1, num_filters, kernel_size=7,
                                               stride=1, padding=3)

        if type_rnn == 'row':
            self.lstm_list = nn.ModuleList([RowLSTM(num_filters, num_filters, input_size) for _ in range(num_layers)])
        elif type_rnn == 'diagonal':
            self.lstm_list = nn.ModuleList([DiagonalLSTM(num_filters, num_filters) for _ in range(num_layers)])
        self.conv2 = MaskedConv2d('B', num_filters, 32 if Config.dataset == "cifar10" else 28, kernel_size=1, stride=1, padding=0)
        self.conv3 = MaskedConv2d('B', 32 if Config.dataset == "cifar10" else 28, 3 if Config.dataset == 'cifar10' else 1, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=Config.dropout)

        self.device = device
        self.to(self.device)

    def forward(self, inputs):
        x = self.conv1(inputs)
        for lstm in self.lstm_list:
            x = lstm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


# ======== Train Function ========
def train(model, train_loader, val_loader, epochs, optimizer, device, criterion, start_epoch=0, checkpoint_path=None):
    train_losses, val_losses = [], []

    best_val_loss = float("inf")
    epochs_no_improve = 0  # Counter for early stopping

    # Learning Rate Scheduler: Fixed Step Decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.step_size, gamma=Config.lr_decay)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch, _ in progress_bar:
            batch = batch.to(device)
            batch = (batch > 0.5).float()

            optimizer.zero_grad()
            output = model(batch)

            loss = Config.criterion(output, batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        writer.add_scalar("runs/experiment_early_stop/Loss/Train",
                          avg_train_loss, epoch)
        writer.add_scalar("runs/experiment_early_stop/Loss/Validation",
                          val_loss, epoch)
        writer.add_scalar("runs/experiment_early_stop/Learning Rate",
                          optimizer.param_groups[0]['lr'], epoch)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] -> Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if os.path.exists(checkpoint_path)==False:
            os.makedirs("/models", exist_ok=True)

        #Early Stop
        if val_loss < best_val_loss-0.001:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset patience counter
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Best model updated! Checkpoint saved at {checkpoint_path}")
        else:
            epochs_no_improve += 1  # Increment patience counter

        if epochs_no_improve >= Config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return train_losses, val_losses


# ======== Evaluate Function ========
def evaluate(model, loader, device, criterion):
    model.eval()
    tot_loss = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            batch = (batch > 0.5).float()  # Binarizing input
            output = model(batch)
            loss = criterion(output, batch)
            tot_loss.append(loss.item())
    return np.mean(tot_loss)


# ======== Negative Log-Likelihood (NLL) Test Evaluation ========
def test_nll(model, test_loader, dataset_type="mnist"):
    model.eval()
    log_probs = []
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.float().to(device)
            output = model(batch)
            nll = -torch.mean(torch.sum(torch.log(output + 1e-8), dim=[1, 2, 3]))  # Sum over all pixels
            log_probs.append(nll.item())

    mean_nll = np.mean(log_probs)

    if dataset_type == "cifar10":
        bits_per_dim = mean_nll / (32 * 32 * 3 * np.log(2))  # Calculate NLL in bits/dim
        print(f"NLL on Test Set: {bits_per_dim:.4f} Bits/Dim")
        return bits_per_dim
    else:
        print(f"NLL on Test Set: {mean_nll:.4f} (in nats)")
        return mean_nll


if __name__ == "__main__":
    if Config.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='data', train=False, transform=transform, download=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size_train, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size_test, shuffle=False)

    model = PixelRNN_model(num_layers=7, num_filters=16, input_size=32 if Config.dataset == "cifar10" else 28, type_rnn=Config.type_rnn).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=Config.lr, weight_decay=Config.L)

    if Config.dataset=='mnist':
        checkpoint_path = 'models/mnist_row.pth' if Config.type_rnn == 'row' else 'models/mnist_diagonal.pth'
    else :
        checkpoint_path = 'models/cifar10_row.pth' if Config.type_rnn == 'row' else 'models/cifar10_diagonal.pth'

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Continue from next epoch
        print(f"Model loaded. Resuming from epoch {start_epoch}")
    else:
        print("Starting training from scratch")
        start_epoch = 0  # Start fresh if no checkpoint exists

    train_losses, val_losses = train(
        model, train_loader, val_loader, epochs=Config.epochs, optimizer=optimizer, device=device,
        criterion=Config.criterion, start_epoch=start_epoch, checkpoint_path=checkpoint_path
    )

    test_loss = evaluate(model, test_loader, device=device, criterion=Config.criterion)
    print(f"Final Test Loss: {test_loss:.4f}")

    nll_test = test_nll(model, test_loader, dataset_type=Config.dataset)

    writer.close()

