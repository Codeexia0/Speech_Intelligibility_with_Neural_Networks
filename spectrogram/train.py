import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
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

# Initialize wandb for experiment tracking
wandb.init(project="speech-clarity-prediction", entity="codeexia0")
wandb.config.update({
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "sample_rate": SAMPLE_RATE,
})

def plot_combined_loss(train_losses, val_losses, fold_index, model_filename):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training and Validation Loss for {model_filename} (Fold {fold_index + 1})")
    plot_filename = f"{model_filename}_fold_{fold_index + 1}.svg"
    plt.savefig(plot_filename, format="svg")
    print(f"Saved loss plot for Fold {fold_index + 1} at {plot_filename}")
    plt.close()
    
def create_data_loader(X, y, masks, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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

    return total_loss / len(data_loader)


def validate_single_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input, target, mask in data_loader:
            input = input.to(device).float()
            target = target.to(device).float().unsqueeze(1)
            prediction = model(input).float()
            loss = loss_fn(prediction, target).mean()  # Validation without masking
            total_loss += loss.item()
    return total_loss / len(data_loader)


def cross_validate(model_class, dataset, loss_fn, optimiser_class, device, epochs, batch_size, k=5):
    n_samples = len(dataset)
    fold_size = n_samples // k

    # Extract features, labels, and masks from the dataset
    X, y, masks = [], [], []
    for data in dataset:
        X.append(data['spin'].cpu().numpy())
        y.append(data['correctness'])
        masks.append(data['mask'].cpu().numpy())
    X = np.array(X)
    y = np.array(y)
    masks = np.array(masks)

    fold_results = []  # To store final results for each fold

    for fold in range(k):
        print(f"Starting Fold {fold + 1}/{k}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_model_filename = f"{MODEL_NAME}_fold_{fold + 1}_{timestamp}.pth"
        # Re-initialize wandb for each fold
        wandb.init(
            project="speech-clarity-prediction-cochleogram",
            entity="codeexia0",
            name=f"{MODEL_NAME}_Fold_{fold + 1}_{timestamp} | {MODEL_NAME}",
            config={
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,  # Define manually
                "model_name": MODEL_NAME,
            }
        )

        # Define train and validation indices
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n_samples
        val_indices = list(range(val_start, val_end))
        train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))

        # Create train and validation DataLoaders
        X_train, y_train, masks_train = X[train_indices], y[train_indices], masks[train_indices]
        X_val, y_val, masks_val = X[val_indices], y[val_indices], masks[val_indices]

        train_loader = create_data_loader(X_train, y_train, masks_train, batch_size)
        val_loader = create_data_loader(X_val, y_val, masks_val, batch_size, shuffle=False)

        # Initialize model and optimizer for each fold
        model = model_class().to(device)
        optimiser = optimiser_class(model.parameters(), lr=LEARNING_RATE)

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} - Fold {fold + 1}/{k}")
            train_loss = train_single_epoch(model, train_loader, loss_fn, optimiser, device, epoch + 1)
            val_loss = validate_single_epoch(model, val_loader, loss_fn, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Log results to WandB
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            print("---------------------------")

        # Save model for this fold
        
        torch.save(model.state_dict(), fold_model_filename)
        print(f"Model for Fold {fold + 1} saved at {fold_model_filename}")

        # Save loss plot for this fold
        plot_combined_loss(train_losses, val_losses, fold, fold_model_filename)

        # Store final epoch results
        fold_results.append({
            "fold": fold + 1,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "model_name": fold_model_filename
        })

        # End WandB run for the fold
        wandb.finish()

    print("Cross-validation complete.")
    # Save summary of all folds to a text file
    summary_filename = f"cross_validation_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_filename, "w") as summary_file:
        summary_file.write("Final Results:\n")
        for result in fold_results:
            summary_file.write(f"Fold {result['fold']}: Final Train Loss = {result['final_train_loss']:.4f}, Final Validation Loss = {result['final_val_loss']:.4f}, Model Name = {result['model_name']}\n")
    print(f"Summary of results saved at {summary_filename}")

    # Print summary of all folds
    print("\nFinal Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Final Train Loss = {result['final_train_loss']:.4f}, Final Validation Loss = {result['final_val_loss']:.4f}, Model Name = {result['model_name']}")


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

    # Perform multiple cross-validation runs
    num_runs = 1  # Number of times to repeat k-fold cross-validation
    for run in range(num_runs):
        print(f"Starting Cross-Validation Run {run + 1}/{num_runs}")
        wandb.init(
            project="speech-clarity-prediction",
            entity="codeexia0",
            name=f"Cross-Validation_Run_{run + 1}",
            tags=[f"run-{run + 1}"],
        )
        
        
        # Choose your model

        # model = HCM(freq_bins=64, rnn_hidden_size=128).to(DEVICE)  # Update if needed
        # MODEL_NAME = "HCM"  # Change to CNNNetwork or ResidualCNN if needed
        
        # model = CNNNetwork().to(DEVICE)
        MODEL_NAME = "CNNNetwork"
        # 
        # model = ResidualCNN()
        # MODEL_NAME = "ResidualCNN"
        cross_validate(
            CNNNetwork,
            # model,
            dataset,
            nn.MSELoss(reduction='none'),
            torch.optim.Adam,
            device,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            k=5  # Number of folds
        )
        # Finish run for multiple cross-validation
        wandb.finish()
