# inference.py
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
import csv
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from cnn import CNNNetwork  # Your CNN model definition
from HCM import HCM  # Your HCM model definition
from ResidualBlock import ResidualCNN # Your ResidualBlock definition

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
# Prediction and Evaluation Functions
# -------------------------
def predict(model, input_tensor, target):
    """
    Make a prediction using the model.
    
    Parameters:
        model: Trained CNN model.
        input_tensor: Input tensor with shape [1, 2, 64, T].
        target: Ground truth correctness (scalar).
    
    Returns:
        predicted (float): Model's predicted value.
        expected (float): Ground truth value.
    """
    model.eval()
    with torch.no_grad():
        # Forward pass
        prediction = model(input_tensor)
        # For a regression output, assume prediction is a tensor with shape [1, 1]
        predicted = prediction[0].item()
        expected = target.item()
    return predicted, expected

def save_results_to_csv(predictions, expected_values, filename="results.csv"):
    """Save predictions and expected values to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sample", "Predicted", "Expected"])
        for idx, (pred, exp) in enumerate(zip(predictions, expected_values), 1):
            writer.writerow([f"Sample {idx}", pred, exp])
    print(f"Results saved to {filename}")

def evaluate_and_plot(predictions, expected_values):
    """Evaluate predictions using MSE, RMSE, MAE and plot metrics."""
    mse = mean_squared_error(expected_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(expected_values, predictions)

    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")

    # Scatter plot: True vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(expected_values, predictions, s=10, alpha=0.7, color="blue")
    plt.plot([min(expected_values), max(expected_values)], [min(expected_values), max(expected_values)],
             color='red', linestyle='--', label='Ideal: y = x')
    plt.xlabel("True Correctness")
    plt.ylabel("Predicted Correctness")
    plt.title("Predicted vs. True Correctness")
    plt.legend()
    plt.show()

# -------------------------
# Main Inference Execution
# -------------------------
if __name__ == "__main__":
    # Set device (for inference, CPU is often sufficient; change if desired)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {DEVICE}")
    
    # Path to precomputed NPZ files (adjust this path as needed)
    DATA_DIR = "C:/Users/Codeexia/FinalSemester/Thesis/coch_data_test"
    
    # Load the precomputed dataset
    dataset = PrecomputedCochleogramDataset(DATA_DIR)
    total_samples = len(dataset)
    print(f"Total samples for inference: {total_samples}")
    
    # Initialize the model and load the trained weights
    model = CNNNetwork().to(DEVICE)
    # model = HCM(freq_bins=64, rnn_hidden_size=128).to(DEVICE)
    # model = ResidualCNN().to(DEVICE)
    
    model_filename = "CNNNetwork_fold_5_20250206_214610.pth"  # Adjust to your trained model file
    model.load_state_dict(torch.load(model_filename, map_location=DEVICE))
    model.eval()
    
    predictions = []
    expected_values = []
    correct_count = 0
    total_correctness = 0
    
    # Loop over the dataset for inference
    for idx in range(total_samples):
        sample = dataset[idx]
        # Use the precomputed combined cochleogram
        # If you want to use only the spin channel, use sample["cochleogram_combined"][0:1] 
        # (note the extra dimension for batch), otherwise feed the entire combined tensor.
        # Here, we assume the model expects the full combined input.
        input_tensor = sample["cochleogram_combined"].unsqueeze(0).to(DEVICE)  # shape: [1, 2, 64, T]
        target = sample["correctness"]
        
        pred, exp = predict(model, input_tensor, target)
        predictions.append(pred)
        expected_values.append(exp)
        
        print(f"Sample {idx+1}: Predicted: {pred:.4f}, Expected: {exp:.4f}")
        if abs(pred - exp) < 0.05:
            correct_count += 1
        total_correctness += exp
    
    correctness_percentage = (correct_count / total_samples) * 100
    avg_correctness = total_correctness / total_samples

    print(f"\nTotal samples: {total_samples}")
    print(f"Correct predictions (within 0.05): {correct_count}")
    print(f"Correctness percentage: {correctness_percentage:.2f}%")
    print(f"Average expected correctness: {avg_correctness:.4f}")
    
    # Save results to CSV
    csv_filename = model_filename.replace(".pth", ".csv")
    save_results_to_csv(predictions, expected_values, filename=csv_filename)
    
    # Evaluate and plot
    evaluate_and_plot(predictions, expected_values)
