import torch
from cnn import CNNNetwork
import torchaudio
from CPC1_data_loader import CPC1
from train import NUM_SAMPLES, SPIN_FOLDER, SCENES_FOLDER, SAMPLE_RATE
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import csv
from ResidualBlock import ResidualCNN



ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/metadata/CPC1.test.json"
SPIN_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/HA_outputs/test"
SCENES_FOLDER = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/scenes"

def predict(model, input, target):
    """Make a prediction and return the predicted and expected values."""
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted = predictions[0].item()  # Single float value prediction
        expected = target  # Already a float value
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
    """Evaluate the predictions and plot results."""

    # Calculate metrics
    mse = mean_squared_error(expected_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(expected_values, predictions)

    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")

    # # 1) Histogram of Prediction Errors
    # errors = [pred - exp for pred, exp in zip(predictions, expected_values)]
    # plt.figure(figsize=(8, 4))
    # plt.hist(errors, bins=30, alpha=0.7, color='blue')
    # plt.xlabel('Prediction Error (predicted - true)')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Prediction Errors')
    # plt.show()

    # # 2) Histogram of Predicted Values
    # plt.figure(figsize=(8, 4))
    # plt.hist(predictions, bins=30, alpha=0.7, color='green')
    # plt.xlabel('Predicted Values')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Predicted Values')
    # plt.show()

    # # 3) Scatter plot: True Intelligibility (x) vs. Predicted Intelligibility (y)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(expected_values, predictions, s=10, alpha=0.7, color="blue")
    # # Optional: If your range is 0..100, set axis limits and draw the diagonal:
    # plt.plot([0,100], [0,100], color='red', linestyle='--', label='Ideal = y=x')
    # plt.xlim([0, 100])
    # plt.ylim([0, 100])
    
    # plt.xlabel("True Intelligibility")
    # plt.ylabel("Predicted Intelligibility")
    # plt.title("Predicted vs. True Intelligibility")
    # plt.legend()
    # plt.show()
    
    import matplotlib.pyplot as plt



    
if __name__ == "__main__":
    model = CNNNetwork()
    # model = ResidualCNN()
    model_filename = "CNNNetwork_fold_5_20250215_172806.pth"
    model.load_state_dict(torch.load(model_filename))
    model.eval()

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
        'cpu',
        max_length=169
    )

    predictions = []
    expected_values = []
    correct_count = 0
    total_samples = len(dataset)
    total_correctness = 0

    for idx in range(total_samples):
        sample = dataset[idx]
        input, target = sample["spin"], sample["correctness"]

        input = input.unsqueeze(0)
        predicted, expected = predict(model, input, target)

        predictions.append(predicted)
        expected_values.append(expected)

        print(f"Sample {idx + 1}: Predicted: {predicted:.4f}, Expected: {expected:.4f}")

        if abs(predicted - expected) < 0.05:
            correct_count += 1

        total_correctness += expected

    correctness_percentage = (correct_count / total_samples) * 100
    avg_correctness = total_correctness / total_samples

    print(f"\nTotal samples: {total_samples}")
    print(f"Correct predictions: {correct_count}")
    print(f"Correctness percentage: {correctness_percentage:.2f}%")
    print(f"Average expected correctness: {avg_correctness:.4f}")

    save_results_to_csv(predictions, expected_values, filename=model_filename.replace(".pth", ".csv"))
    evaluate_and_plot(predictions, expected_values)
