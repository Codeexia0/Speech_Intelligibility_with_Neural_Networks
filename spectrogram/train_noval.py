import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchaudio
import wandb
import numpy as np
from cnn import CNNNetwork
from CPC1_data_loader import CPC1
from ResidualBlock import ResidualCNN
import random
import datetime
import os
import time


# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Dataset Parameters
ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/metadata/CPC1.train.json"
SPIN_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/HA_outputs/train"
SCENES_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/scenes"

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 6

# Set random seeds for reproducibility
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


def plot_training_loss(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss over Epochs")
    plt.show()

def plot_validation_loss(val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses, label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Validation Loss over Epochs")
    plt.show()


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch):
    model.train()
    total_loss = 0
    for input, target, mask in data_loader:
        input = input.to(device).float()
        mask = mask.to(device).unsqueeze(1)  # Ensure mask matches prediction dimensions
        target = target.to(device).float().unsqueeze(1)
        prediction = model(input).float()
        loss = (loss_fn(prediction, target) * mask).sum() / mask.sum()  # Apply mask to loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    wandb.log({"epoch": epoch, "train_loss": total_loss / len(data_loader)})
    print(f"Epoch {epoch}: Loss: {total_loss / len(data_loader):.4f}")
    return total_loss / len(data_loader)

def validate_single_epoch(model, data_loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input, target, mask in data_loader:
            input = input.to(device).float()
            target = target.to(device).float().unsqueeze(1)
            prediction = model(input).float()
            loss = loss_fn(prediction, target).mean()  # Validation without masking
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    wandb.log({"epoch": epoch, "val_loss": avg_loss})
    print(f"Epoch {epoch}: Validation Loss: {avg_loss:.4f}")
    return avg_loss

def train_and_validate(model, train_loader, val_loader, loss_fn, optimiser, device, epochs):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_single_epoch(model, train_loader, loss_fn, optimiser, device, epoch + 1)
        val_loss = validate_single_epoch(model, val_loader, loss_fn, device, epoch + 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print("---------------------------")

    # Plot training loss and validation loss in separate figures
    plot_training_loss(train_losses)
    plot_validation_loss(val_losses)

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # Define transformations
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )

    dataset = CPC1(
        ANNOTATIONS_FILE,
        SPIN_FOLDER,
        SCENES_FOLDER,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        max_length=169,
    )
    

    # Convert dataset to features (X), labels (y), and masks
    X, y, masks = [], [], []
    for data in dataset:
        X.append(data['spin'].cpu().numpy())
        y.append(data['correctness'])  # Correctness is already a float
        masks.append(data['mask'].cpu().numpy())
    X = np.array(X)
    y = np.array(y)
    masks = np.array(masks)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(X, y, masks, test_size=0.1)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create DataLoaders for batch processing
    def create_data_loader(X, y, masks, batch_size):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, masks_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_data_loader(X_train, y_train, masks_train, BATCH_SIZE)
    val_loader = create_data_loader(X_val, y_val, masks_val, BATCH_SIZE)

    # Initialize model, loss function, and optimizer
    # model = CNNNetwork().to(device)
    # MODEL_NAME = "CNNNetwork"
    
    model = CNNNetwork().to(device)
    MODEL_NAME = "CNNNetwork_V2"
        
    # model = ResidualCNN().to(device)
    # MODEL_NAME = "ResidualCNN"
    loss_fn = nn.MSELoss(reduction='none')
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{MODEL_NAME}_model_{timestamp}_{EPOCHS}_{LEARNING_RATE}.pth"

    # Initialize wandb for experiment tracking with model filename + model name as run name
    wandb.init(project="speech-clarity-prediction_",
               entity="codeexia0",
               name=f"{model_filename}")
    wandb.config.update({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "sample_rate": 16000,
        "max_length": 169,
        "model_name": MODEL_NAME,
        "model_filename": model_filename
    })

    try:
        # Train and validate
        train_and_validate(model, train_loader, val_loader, loss_fn, optimiser, device, EPOCHS)
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        print("Ensure the padding length is consistent for all tensors.")
    finally:

        if not os.path.exists(model_filename):
            print("Saving model due to final exit.")
            torch.save(model.state_dict(), model_filename)
            print("Batch size: ", BATCH_SIZE, "Epochs: ", EPOCHS, "Learning rate: ", LEARNING_RATE)
            print(f"Final model saved at {model_filename}")

    wandb.finish()