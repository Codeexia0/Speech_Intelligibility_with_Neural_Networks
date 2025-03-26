import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import wandb
import datetime
import random

from cnn import CNNNetwork      # Your CNN model definition (if needed)
from HCM import HCM             # Your HCM model definition (if needed)
from ResidualBlock import ResidualCNN  # Your ResidualCNN (or ResidualBlock) model definition

# -------------------------
# Custom Dataset for Precomputed Files
# -------------------------
class PrecomputedCochleogramDataset(Dataset):
    def __init__(self, data_dir):
        """ data_dir: Directory containing the precomputed .npz files. """
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path)
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
        inputs = batch["cochleogram_combined"].to(device)
        targets = batch["correctness"].to(device).unsqueeze(1)
        masks = batch["mask"].to(device).unsqueeze(1)  # Adjust shape if needed

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
            loss = loss_fn(predictions, targets).mean()  # No masking during validation
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

            # Log results to wandb for this fold
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user! Saving model before exiting...")

    finally:
        torch.save(model.state_dict(), model_filename)
        print(f"✅ Model saved at {model_filename}")

    return train_losses, val_losses

# -------------------------
# k-Fold Cross Validation Function
# -------------------------
def cross_validate(dataset, model_class, loss_fn, optimiser_class, device, epochs, batch_size, k=5):
    n_samples = len(dataset)
    indices = list(range(n_samples))
    random.shuffle(indices)  # Optional: shuffle before splitting
    fold_size = n_samples // k
    fold_results = []

    for fold in range(k):
        print(f"\n========== Starting Fold {fold + 1}/{k} ==========")
        val_indices = indices[fold * fold_size: (fold + 1) * fold_size] if fold < k - 1 else indices[fold * fold_size:]
        train_indices = list(set(indices) - set(val_indices))

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_class().to(device)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_model_filename = f"{MODEL_NAME}_fold_{fold + 1}_{timestamp}.pth"

        # 🛠 **Fix: Move wandb.init() BEFORE accessing `wandb.config`**
        wandb.init(
            project="speech-clarity-prediction-cochleogram",
            entity="codeexia0",
            name=f"{MODEL_NAME}_Fold_{fold + 1}_{timestamp} | {MODEL_NAME}",
            config={
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,  # Define manually
            }
        )

        # ✅ **Now it's safe to use wandb.config**
        optimiser = optimiser_class(model.parameters(), lr=LEARNING_RATE)

        train_losses, val_losses = train_model(model, train_loader, val_loader, loss_fn, optimiser, device, epochs, fold_model_filename)

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Training & Validation Loss (Fold {fold + 1})")
            plot_filename = f"{fold_model_filename}_loss.svg"
            plt.savefig(plot_filename, format="svg")
            plt.close()
            print(f"Saved loss plot for Fold {fold + 1} at {plot_filename}")
        except ImportError:
            print("matplotlib not installed; skipping loss plot saving.")

        fold_results.append({
            "fold": fold + 1,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "model_filename": fold_model_filename
        })

        wandb.finish()

    summary_filename = f"cross_validation_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_filename, "w") as summary_file:
        summary_file.write("Cross-Validation Final Results:\n")
        for result in fold_results:
            summary_file.write(
                f"Fold {result['fold']}: Final Train Loss = {result['final_train_loss']:.4f}, "
                f"Final Validation Loss = {result['final_val_loss']:.4f}, Model = {result['model_filename']}\n"
            )
    print(f"\nSummary of cross-validation results saved at {summary_filename}")
    return fold_results


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {DEVICE}")
    BATCH_SIZE = 32  
    EPOCHS = 100  
    LEARNING_RATE = 0.001  

    # Set directories (adjust paths as needed)
    DATA_DIR_TRAIN = "C:/Users/Codeexia/FinalSemester/Thesis/coch_data_train"
    
    # For k-fold, we use one dataset directory (the training set)
    dataset = PrecomputedCochleogramDataset(DATA_DIR_TRAIN)
    
    NUM_FOLDS = 5  # Set the number of folds
    
    # Choose your model

    # model = HCM(freq_bins=64, rnn_hidden_size=128)  # Update if needed
    # MODEL_NAME = "HCM"
    
    MODEL_NAME = "CNNNetwork"
    
    # model = ResidualCNN()
    # MODEL_NAME = "ResidualCNN"
    fold_results = cross_validate(
        dataset=dataset,
        model_class=CNNNetwork,            # Use your desired model class
        loss_fn=nn.MSELoss(reduction='none'), # reduction='none' for masking during training because of variable lengths
        optimiser_class=torch.optim.Adam,
        device=DEVICE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        k=NUM_FOLDS
    ) 
    
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{MODEL_NAME}_model_{timestamp}.pth"

    # Initialize global wandb run (optional, for overall tracking)
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

    wandb.finish()  # Finish the global wandb run

    # (Optional) Print final summary to console
    print("\n=== Final Cross-Validation Results ===")
    for result in fold_results:
        print(f"Fold {result['fold']}: Final Train Loss = {result['final_train_loss']:.4f}, "
              f"Final Validation Loss = {result['final_val_loss']:.4f}, "
              f"Model File = {result['model_filename']}")
