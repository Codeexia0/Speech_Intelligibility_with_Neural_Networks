# precompute_cpc.py
import os
import numpy as np
import torch
from CPC1_data_loader import CPC1  # Make sure this file contains your CPC1 class

def precompute_and_save(annotations_file, spin_folder, scenes_folder,
                        sample_rate, num_samples, device, max_length, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the CPC1 dataset instance
    dataset = CPC1(
        annotations_file,
        spin_folder,
        scenes_folder,
        target_sample_rate=sample_rate,
        num_samples=num_samples,
        device=device,
        max_length=max_length
    )
    
    print(f"Precomputing and saving {len(dataset)} samples to {save_dir} ...")
    for i in range(len(dataset)):
        sample = dataset[i]
        # Each sample is a dict with keys: "cochleogram_combined", "mask", "correctness"
        # cochleogram_combined is a tensor of shape [2, 64, T]
        cochleogram_combined = sample["cochleogram_combined"].cpu().numpy()
        mask = sample["mask"].cpu().numpy()
        correctness = sample["correctness"].cpu().numpy()  # a scalar
        
        # Save all data in one NPZ file
        file_path = os.path.join(save_dir, f"sample_{i:05d}.npz")
        np.savez(file_path, cochleogram=cochleogram_combined, mask=mask, correctness=correctness)
        
        if (i+1) % 100 == 0:
            print(f"Saved {i+1} samples...")
    
    print("Precomputation complete.")

if __name__ == "__main__":
    # Define paths and parameters (adjust these paths as needed)
    annotations_file = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/metadata/CPC1.train.json"
    spin_folder = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/HA_outputs/train"
    scenes_folder = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/scenes"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 2421
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 537  # This is your fixed length for padding/truncation
    SAVE_DIR = "C:/Users/Codeexia/FinalSemester/Thesis/coch_data_train"  # Folder where precomputed files will be saved

    precompute_and_save(annotations_file, spin_folder, scenes_folder,
                        SAMPLE_RATE, NUM_SAMPLES, DEVICE, MAX_LENGTH, SAVE_DIR)
