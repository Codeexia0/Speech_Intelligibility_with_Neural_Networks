import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
import numpy as np
from cnn import CNNNetwork  # Your CNN model
from CPC1_data_loader import CPC1  # Your CPC dataset implementation
import datetime
import os
import sys
from sklearn.model_selection import train_test_split

# ================================
# Hyperparameters & Dataset Params
# ================================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/metadata/CPC1.train.json"
SPIN_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/HA_outputs/train"
SCENES_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/scenes"

SAMPLE_RATE = 16000
MAX_LENGTH = 537  # Maximum length for the cochleograms

# Initialize wandb for experiment tracking
wandb.init(project="speech-clarity-prediction-cochleogram", entity="codeexia0")
wandb.config.update({
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "sample_rate": SAMPLE_RATE,
    "max_length": MAX_LENGTH,
})


# ================================
# DataLoader Helper Function
# ================================
def create_dataset_loader(X, y, masks, batch_size, shuffle=True):
    """
    Converts extracted NumPy arrays into a TensorDataset and DataLoader.
    
    Parameters:
        X (np.ndarray): Input features array.
        y (np.ndarray): Target values array.
        masks (np.ndarray): Mask values array.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
    
    Returns:
        DataLoader: A DataLoader for the TensorDataset.
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),   # Ensures target shape is [batch_size, 1]
        torch.tensor(masks, dtype=torch.float32).unsqueeze(1)  # Ensures mask shape matches model output
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ================================
# Extraction Function
# ================================
def extract_numpy_data(cpc_dataset):
    """
    Iterates through the CPC1 dataset and converts each sample into numpy arrays.
    Assumes each sample is a dict with keys: 'cochleogram_combined', 'mask', and 'correctness'.
    
    Returns:
        X (np.ndarray): Array of inputs (cochleogram_combined).
        y (np.ndarray): Array of target values (correctness).
        masks (np.ndarray): Array of masks.
    """
    X_list, y_list, masks_list = [], [], []
    for i in range(len(cpc_dataset)):
        sample = cpc_dataset[i]
        # Move tensors to CPU and convert to numpy arrays.
        X_list.append(sample["cochleogram_combined"].cpu().numpy())
        y_list.append(sample["correctness"].cpu().numpy())
        masks_list.append(sample["mask"].cpu().numpy())
    X = np.array(X_list)
    y = np.array(y_list)
    masks = np.array(masks_list)
    return X, y, masks


# ================================
# Training & Validation Functions
# ================================
def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    for batch_idx, (inputs, targets, masks) in enumerate(data_loader):
        # Move data to the correct device
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions = model(inputs).float()
        # Compute masked MSE loss
        loss = (loss_fn(predictions, targets) * masks).sum() / masks.sum()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        print(f"Epoch [{epoch}] | Batch [{batch_idx+1}/{num_batches}] | Loss: {loss.item():.4f}")

    return total_loss / num_batches

def validate_single_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    with torch.no_grad():
        for inputs, targets, masks in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs).float()
            # Compute average loss (mask not applied during validation)
            loss = loss_fn(predictions, targets).mean()
            total_loss += loss.item()
    return total_loss / num_batches

def train_model(model, train_loader, val_loader, loss_fn, optimiser, device, epochs, model_filename):
    train_losses, val_losses = [], []

    try:
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            train_loss = train_single_epoch(model, train_loader, loss_fn, optimiser, device, epoch)
            val_loss = validate_single_epoch(model, val_loader, loss_fn, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Log results to wandb
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user! Saving model before exiting...")

    finally:
        torch.save(model.state_dict(), model_filename)
        print(f"✅ Model saved at {model_filename}")

    return train_losses, val_losses


# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # Load the CPC1 dataset (with actual CPC data, not dummy numbers)
    cpc_dataset = CPC1(
        ANNOTATIONS_FILE,
        SPIN_FOLDER,
        SCENES_FOLDER,
        target_sample_rate=SAMPLE_RATE,
        num_samples=None,  # Use all available samples (or limit if desired)
        device=device,
        max_length=MAX_LENGTH
    )

    # Extract the real cochleogram data as NumPy arrays
    print("Extracting data from the CPC1 dataset...")
    X, y, masks = extract_numpy_data(cpc_dataset)
    print(f"Extracted {len(X)} samples.")

    # Split the data into training (80%) and validation (20%) sets
    X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(
        X, y, masks, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create DataLoaders using the helper function
    train_loader = create_dataset_loader(X_train, y_train, masks_train, BATCH_SIZE, shuffle=True)
    val_loader = create_dataset_loader(X_val, y_val, masks_val, BATCH_SIZE, shuffle=False)

    # Initialize your CNN model and move it to the chosen device
    model = CNNNetwork().to(device)

    # Define the optimizer and loss function (using a masked MSE loss)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss(reduction='none')

    # Define the filename for saving the model
    model_filename = f"cnn_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, loss_fn, optimiser, device, EPOCHS, model_filename)

    # End the wandb session
    wandb.finish()
