# train_with_precomputed_full.py
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb
import datetime
from cnn import CNNNetwork  # Your CNN model definition
from ResidualBlock import ResidualCNN  # Your ResidualCNN (or ResidualBlock) model definition
from HCM import HCM  # Your HCM model definition



LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 200

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
# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "C:/Users/Codeexia/FinalSemester/Thesis/coch_data_train"

    # Create the precomputed dataset (using the entire dataset)
    dataset = PrecomputedCochleogramDataset(DATA_DIR)
    
    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Choose your model

    # model = HCM(freq_bins=64, rnn_hidden_size=128).to(DEVICE)  # Update if needed
    # MODEL_NAME = "HCM"  # Change to CNNNetwork or ResidualCNN if needed
    
    model = CNNNetwork().to(DEVICE)
    MODEL_NAME = "CNNNetwork"
    
    # model = ResidualCNN().to(DEVICE)
    # MODEL_NAME = "ResidualCNN"
    

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss(reduction='none')

    # Define model filename dynamically
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{MODEL_NAME}_model_{timestamp}.pth"

    # Initialize wandb for experiment tracking with model filename + model name as run name
    wandb.init(project="speech-clarity-prediction-cochleogram",
               entity="codeexia0",
               name=f"{model_filename} | {MODEL_NAME}")

    wandb.config.update({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "sample_rate": 16000,
        "max_length": 537,
        "model_name": MODEL_NAME,
        "model_filename": model_filename
    })

    # Train the model
    train_losses = train_model(model, data_loader, loss_fn, optimiser, DEVICE, EPOCHS, model_filename)

    wandb.finish()
