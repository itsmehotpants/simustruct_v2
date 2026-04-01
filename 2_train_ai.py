import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

class PlateDataset(Dataset):
    def __init__(self, data_dir):
        files = glob.glob(f"{data_dir}/*.npz")
        self.inputs, self.targets = [], []
        for f in files:
            data = np.load(f)
            nodes = data["nodes"][:, :2]
            param_block = np.tile(data["params"], (nodes.shape[0], 1))
            self.inputs.append(np.hstack((nodes, param_block)))
            self.targets.append(data["stress"])
        self.X = torch.tensor(np.vstack(self.inputs), dtype=torch.float32)
        self.Y = torch.tensor(np.hstack(self.targets), dtype=torch.float32).unsqueeze(1) / 1e6

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    dataset = PlateDataset("dataset_ellipse")
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    model = SurrogateModel()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("Training AI (50 Epochs)...")
    for epoch in range(50):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    # ---> THE MISSING LINE! <---
    torch.save(model.state_dict(), "stress_model.pth")
    print(f"Final Loss: {loss.item():.4f}")
    print("✅ Model successfully saved as stress_model.pth!")
