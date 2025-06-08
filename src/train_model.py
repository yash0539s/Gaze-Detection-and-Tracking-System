import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from data_loader import MPIIGazeDataset
from tqdm import tqdm 
import os

class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MPIIGazeDataset(root_dir="../data/mpiigaze", split='train')
    val_dataset = MPIIGazeDataset(root_dir="../data/mpiigaze", split='val')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = GazeCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.5f} Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/gaze_regressor.pth")
            print(f"Model saved with val loss {best_val_loss:.5f}")

if __name__ == '__main__':
    train()
