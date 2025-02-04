# train_with_precomputed_full.py
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb
import datetime
from cnn import CNNNetwork  # Your CNN model definition

# -------------------------
# Custom Dataset for Precomputed Files
# -------------------------
class PrecomputedCochleogramDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: Directory containing the precomputed .npz files.
        """
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path)
        # Load the combined cochleogram, mask, and correctness score.
        cochleogram_combined = torch.tensor(data["cochleogram"], dtype=torch.float32)
        mask = torch.tensor(data["mask"], dtype=torch.float32)
        correctness = torch.tensor(data["correctness"], dtype=torch.float32)
        return {
            "cochleogram_combined": cochleogram_combined,
            "mask": mask,
            "correctness": correctness
        }

# -------------------------
# Training & Validation Functions
# -------------------------
def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    for batch_idx, batch in enumerate(data_loader):
        inputs = batch["cochleogram_combined"].to(device)  # shape: [B, 2, 64, T]
        targets = batch["correctness"].to(device).unsqueeze(1)  # shape: [B, 1]
        masks = batch["mask"].to(device).unsqueeze(1)  # adjust shape if needed

        predictions = model(inputs).float()
        # Compute masked MSE loss
        loss = (loss_fn(predictions, targets) * masks).sum() / masks.sum()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        print(f"Epoch {epoch} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

    return total_loss / num_batches

def validate_single_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["cochleogram_combined"].to(device)
            targets = batch["correctness"].to(device).unsqueeze(1)
            predictions = model(inputs).float()
            loss = loss_fn(predictions, targets).mean()
            total_loss += loss.item()
    return total_loss / num_batches

def train_model(model, data_loader, loss_fn, optimiser, device, epochs, model_filename):
    train_losses = []
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        loss_epoch = train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch)
        train_losses.append(loss_epoch)
        wandb.log({"epoch": epoch, "loss": loss_epoch})
        print(f"Epoch {epoch}: Loss = {loss_epoch:.4f}")
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved at {model_filename}")
    return train_losses

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Initialize wandb for experiment tracking
    wandb.init(project="speech-clarity-prediction-cochleogram", entity="codeexia0")
    wandb.config.update({
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "sample_rate": 16000,
        "max_length": 537,
    })

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "C:/Users/Codeexia/FinalSemester/Thesis/coch_data"  # Directory where NPZ files are saved

    # Create the precomputed dataset (using the entire dataset)
    dataset = PrecomputedCochleogramDataset(DATA_DIR)
    
    # Create a DataLoader for the full dataset
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Initialize your CNN model and move it to the device
    model = CNNNetwork().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss(reduction='none')

    # Define a filename for saving the model
    model_filename = f"cnn_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

    # Train the model on the full dataset
    train_losses = train_model(model, data_loader, loss_fn, optimiser, DEVICE, 30, model_filename)

    wandb.finish()
