import json
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gammatone.gtgram import gtgram 

class CPC1(Dataset):
    def __init__(self,
                 annotations_file,
                 spin_folder,
                 scenes_folder,
                 target_sample_rate,
                 num_samples,
                 device,
                 max_length):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.spin_folder = Path(spin_folder)
        self.scenes_folder = Path(scenes_folder)
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        entry = self.annotations[index]
        spin_path, target_path = self._get_audio_sample_paths(entry)
        correctness = entry['correctness']

        # Load audio signals
        spin_signal, spin_sr = torchaudio.load(spin_path)
        target_signal, target_sr = torchaudio.load(target_path)

        spin_signal = spin_signal.to(self.device)
        target_signal = target_signal.to(self.device)

        # Process the signals
        spin_signal = self._resample_if_necessary(spin_signal, spin_sr)
        target_signal = self._resample_if_necessary(target_signal, target_sr)
        spin_signal = self._mix_down_if_necessary(spin_signal)
        target_signal = self._mix_down_if_necessary(target_signal)

        # Compute cochleograms
        spin_cochleogram = self._compute_cochleogram(spin_signal)
        target_cochleogram = self._compute_cochleogram(target_signal)
        
        plot_cochleograms(spin_cochleogram, target_cochleogram)


        # Normalize
        spin_cochleogram = self._normalize_spectrogram(spin_cochleogram)
        target_cochleogram = self._normalize_spectrogram(target_cochleogram)
        plot_cochleograms(spin_cochleogram, target_cochleogram, title="after normalization")



        # Pad or truncate
        spin_cochleogram, target_cochleogram = self._pad_or_truncate_to_length(
            spin_cochleogram, target_cochleogram, self.max_length
        )
        plot_cochleograms(spin_cochleogram, target_cochleogram, title="after padding/truncation")



        # Create mask
        mask = self._create_mask(spin_cochleogram, self.max_length)

        # Stack cochleograms
        cochleogram_combined = torch.stack((spin_cochleogram, target_cochleogram), dim=0)

        # Debugging print to confirm types
        # print(f"Debugging: cochleogram_combined type = {type(cochleogram_combined)}, shape = {cochleogram_combined.shape}")
        # print(f"Debugging: correctness type = {type(correctness)}, value = {correctness}")

        # Ensure correctness is a tensor
        correctness_tensor = torch.tensor(correctness / 100.0, dtype=torch.float32)

        # 🔹 Add an assertion to catch any string values
        assert isinstance(cochleogram_combined, torch.Tensor), f"cochleogram_combined is {type(cochleogram_combined)}"
        assert isinstance(mask, torch.Tensor), f"mask is {type(mask)}"
        assert isinstance(correctness_tensor, torch.Tensor), f"correctness is {type(correctness_tensor)}"

        return {
            "cochleogram_combined": cochleogram_combined,
            "mask": mask,
            "correctness": correctness_tensor
        }


        
    def _pad_or_truncate_to_length(self, signal1, signal2, target_length):
        # Truncate if the signal is longer than target_length.
        if signal1.shape[-1] > target_length:
            signal1 = signal1[..., :target_length]
        if signal2.shape[-1] > target_length:
            signal2 = signal2[..., :target_length]
        
        # Pad if the signal is shorter than target_length.
        if signal1.shape[-1] < target_length:
            pad1 = target_length - signal1.shape[-1]
            signal1 = F.pad(signal1, (0, pad1))
        if signal2.shape[-1] < target_length:
            pad2 = target_length - signal2.shape[-1]
            signal2 = F.pad(signal2, (0, pad2))
        
        return signal1, signal2

        
    def _compute_cochleogram(self, signal):
        """
        Convert a raw audio signal to a cochleogram using gammatone filtering.
        """
        signal = signal.cpu().numpy().squeeze()  # Convert to NumPy array
        cochleogram = gtgram(signal,
                             self.target_sample_rate,
                             window_time=0.025,  # 25ms window
                             hop_time=0.010,     # 10ms step
                             channels=64,        # 64 frequency channels
                             f_min=50)           # Minimum frequency 50 Hz
        cochleogram = torch.tensor(cochleogram, dtype=torch.float32).to(self.device)
        return cochleogram

    def _normalize_spectrogram(self, spectrogram):
        """
        Normalize the input (spectrogram or cochleogram) by applying log scaling and then 
        global normalization to have zero mean and unit variance.
        """
        spectrogram = torch.clamp(spectrogram, min=1e-10)
        log_spectrogram = torch.log10(spectrogram)
        mean = log_spectrogram.mean()
        std = log_spectrogram.std()
        normalized_spectrogram = (log_spectrogram - mean) / std
        return normalized_spectrogram

    def _resample_if_necessary(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_timings(self, spin_signal, target_signal, sample_rate):
        cut_start = int(2 * sample_rate)  # Remove first 2 seconds
        cut_end = int(1 * sample_rate)      # Remove last 1 second
        if spin_signal.shape[1] > cut_start + cut_end:
            spin_signal = spin_signal[:, cut_start:-cut_end]
        if target_signal.shape[1] > cut_start + cut_end:
            target_signal = target_signal[:, cut_start:-cut_end]
        return spin_signal, target_signal

    def _pad_to_same_length(self, signal1, signal2, max_length):
        time_dim1 = signal1.shape[-1]
        time_dim2 = signal2.shape[-1]
        max_time_dim = max(time_dim1, time_dim2, max_length)
        padding1 = (0, max_time_dim - time_dim1)
        padding2 = (0, max_time_dim - time_dim2)
        signal1 = F.pad(signal1, padding1)
        signal2 = F.pad(signal2, padding2)
        return signal1, signal2

    def _create_mask(self, spectrogram, max_length):
        time_dim = spectrogram.shape[-1]
        mask = torch.ones(time_dim, dtype=torch.float32, device=spectrogram.device)
        if time_dim < max_length:
            padding = max_length - time_dim
            mask = torch.cat([mask, torch.zeros(padding, dtype=torch.float32, device=spectrogram.device)])
        return mask

    def _get_audio_sample_paths(self, entry):
        spin_path = self.spin_folder / f"{entry['signal']}.wav"
        target_path = self.scenes_folder / f"{entry['scene']}_target_anechoic.wav"
        return spin_path, target_path
    
    def compute_raw_max_length(self):
        """
        Compute the maximum time dimension (T) for the raw cochleograms (after computing
        and normalizing, but before applying any padding/truncation).
        """
        max_length = 0
        for i in range(len(self.annotations)):
            entry = self.annotations[i]
            spin_path, target_path = self._get_audio_sample_paths(entry)
            
            # Load audio signals
            spin_signal, spin_sr = torchaudio.load(spin_path)
            target_signal, target_sr = torchaudio.load(target_path)
            
            spin_signal = spin_signal.to(self.device)
            target_signal = target_signal.to(self.device)
            
            # Resample if needed
            spin_signal = self._resample_if_necessary(spin_signal, spin_sr)
            target_signal = self._resample_if_necessary(target_signal, target_sr)
            
            # Mix down to mono if necessary
            spin_signal = self._mix_down_if_necessary(spin_signal)
            target_signal = self._mix_down_if_necessary(target_signal)
            
            # Cut off beginning and end portions
            spin_signal, target_signal = self._cut_timings(spin_signal, target_signal, self.target_sample_rate)
            
            # Compute cochleograms (and normalize, if desired)
            spin_cochleogram = self._compute_cochleogram(spin_signal)
            target_cochleogram = self._compute_cochleogram(target_signal)
            
            spin_cochleogram = self._normalize_spectrogram(spin_cochleogram)
            target_cochleogram = self._normalize_spectrogram(target_cochleogram)
            
            # Determine the raw time dimensions
            current_length = max(spin_cochleogram.shape[-1], target_cochleogram.shape[-1])
            if current_length > max_length:
                max_length = current_length
            print(f"Processed sample {i+1}/{len(self.annotations)}, Current Raw Max Length: {max_length}")
        return max_length
    
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_cochleograms(spin_cochleogram, target_cochleogram, title=None, spin_label="Spin", target_label="Target"):
    """
    Plots two cochleograms one above the other with an optional title.

    Parameters:
    - spin_cochleogram (torch.Tensor or np.ndarray): Cochleogram for the spin signal.
    - target_cochleogram (torch.Tensor or np.ndarray): Cochleogram for the target signal.
    - title (str, optional): Custom title prefix. If None, defaults to "Spin Cochleogram" and "Target Cochleogram".
    - spin_label (str): Label for the spin cochleogram.
    - target_label (str): Label for the target cochleogram.
    """

    # Convert tensors to NumPy if needed
    if isinstance(spin_cochleogram, torch.Tensor):
        spin_cochleogram = spin_cochleogram.cpu().numpy()
    if isinstance(target_cochleogram, torch.Tensor):
        target_cochleogram = target_cochleogram.cpu().numpy()

    # Determine titles dynamically
    spin_title = f"{spin_label} Cochleogram" if title is None else f"{spin_label} Cochleogram {title}"
    target_title = f"{target_label} Cochleogram" if title is None else f"{target_label} Cochleogram {title}"

    # Plot the cochleograms
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.imshow(spin_cochleogram, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label="Amplitude")
    plt.title(spin_title)
    plt.xlabel("Time Steps")
    plt.ylabel("Frequency Channels")

    plt.subplot(2, 1, 2)
    plt.imshow(target_cochleogram, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label="Amplitude")
    plt.title(target_title)
    plt.xlabel("Time Steps")
    plt.ylabel("Frequency Channels")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    annotations_file = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/metadata/CPC1.test.json"
    spin_folder = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/HA_outputs/test"
    scenes_folder = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.test.v1/clarity_CPC1_data/clarity_data/scenes"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 2421
    # For the temporary dataset, you can pass a dummy max_length (e.g., 0)
    DUMMY_MAX_LENGTH = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a temporary dataset for finding the maximum raw length
    # Create a temporary dataset.
    temp_dataset = CPC1(
        annotations_file,
        spin_folder,
        scenes_folder,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device,
        max_length=1  # The value here is irrelevant since we won't call __getitem__ directly.
    )

    # # Compute the maximum raw cochleogram length without padding.
    # max_coch_length = temp_dataset.compute_raw_max_length()
    # print("Maximum raw cochleogram length:", max_coch_length)

    # Now reinitialize your dataset with the computed maximum length.
    dataset = CPC1(
        annotations_file,
        spin_folder,
        scenes_folder,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device,
        max_length=537
    )


    # Verify one sample:
    print("Total samples:", len(dataset))
    sample = dataset[0]
    print("Cochleogram shape:", sample["cochleogram_combined"].shape)  # Should be [2, 64, max_coch_length]
    # Get a sample from the dataset

    # Extract the cochleograms
    spin_cochleogram = sample["cochleogram_combined"][0]  # First channel = Spin
    target_cochleogram = sample["cochleogram_combined"][1]  # Second channel = Target

    # # Plot the spin cochleogram
    # plot_cochleogram(spin_cochleogram, title="Spin Cochleogram")

    # # Plot the target cochleogram
    # plot_cochleogram(target_cochleogram, title="Target Cochleogram")

    
