import torch
import numpy as np
import h5py
import pickle
from torch.utils.data import Dataset
from collections import Counter
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch




class WRsmallepoch(Dataset):
    def __init__(self, data_file: str, annotation_file: str, epoch_size: float, single_channel_flag: bool =True, psd_flag: bool = True, epoch_id_restriction: int = None, sample_rate: int = 5000):
        self.data_file = data_file
        self.annotation_file = annotation_file
        self.annotations = self.load_annotations(epoch_id_restriction)
        self.epoch_size = epoch_size
        self.sample_rate = sample_rate
        self.epoch_num_samples = self.epoch_size * self.sample_rate
        self.frequencies = self.compute_frequency_vector()
        self.freq_weights = torch.Tensor(np.roll(np.unique(self.frequencies),1))
        self.single_channel_flag = single_channel_flag
        self.psd_flag = psd_flag


    def compute_frequency_vector(self):
        # Example vector
        epochs = self.annotations['epoch_id'].to_list()
        # Step 1: Count occurrences of each number
        counts = Counter(epochs)

        # Step 2: Calculate relative frequency
        total_count = len(epochs)
        relative_frequency = {num: count / total_count for num, count in counts.items()}

        # Step 3: Replace each number with its relative frequency
        result_vector = [relative_frequency[num] for num in epochs]
        return torch.Tensor(result_vector)
    
    def load_annotations(self, epoch_id_restriction):
        with open(self.annotation_file, 'rb') as f:
            annotations = pickle.load(f)
        
        if epoch_id_restriction is not None:
            annotations = annotations[annotations['epoch_id'] == epoch_id_restriction]
        
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        start_time = annotation['start_time']
        end_time = annotation['stop_time']
        label = annotation['epoch_id']

        start_index = int(start_time * self.sample_rate)
        
        if self.single_channel_flag:
            with h5py.File(self.data_file, 'r') as f:
                ch1_data = f['Ch.1'][start_index:int(start_index+self.epoch_num_samples)]
                ch1_mean = f['Ch.1'].attrs['mean']
                ch1_std = f['Ch.1'].attrs['std']
            
            ch1_data = self.downsample(ch1_data, original_fs=self.sample_rate, target_fs=100)
            ch1_data = self.filter_data(ch1_data, lowcut=5, highcut=30, fs=100.0, order=5)
            ch1_data = (ch1_data - ch1_mean) / (ch1_std + 1e-10)
            if self.psd_flag:
                _, ch1_data = self.power_spectrum(ch1_data, fs=100.0)
            data_tensor = torch.as_tensor(ch1_data.copy(), dtype=torch.get_default_dtype())
            label_tensor = torch.tensor(label, dtype=torch.long)
            return (data_tensor, label_tensor)
        else:
            with h5py.File(self.data_file, 'r') as f:
                ch1_data = f['Ch.1'][start_index:int(start_index+self.epoch_num_samples)]
                ch2_data = f['Ch.2'][start_index:int(start_index+self.epoch_num_samples)]
                
                ch1_mean = f['Ch.1'].attrs['mean']
                ch1_std = f['Ch.1'].attrs['std']
                ch2_mean = f['Ch.2'].attrs['mean']
                ch2_std = f['Ch.2'].attrs['std']
            
            ch1_data = self.downsample(ch1_data, original_fs=self.sample_rate, target_fs=100)
            ch2_data = self.downsample(ch2_data, original_fs=self.sample_rate, target_fs=100)

            ch1_data = self.filter_data(ch1_data, lowcut=5, highcut=30, fs=100.0, order=5)
            ch2_data = self.filter_data(ch2_data, lowcut=5, highcut=30, fs=100.0, order=5)

            if self.psd_flag:
                _, ch1_data = self.power_spectrum(ch1_data, fs=100.0)
                _, ch2_data = self.power_spectrum(ch2_data, fs=100.0)

            #normalize each channel separately to zero mean and unit variance
            ch1_data = (ch1_data - ch1_mean) / (ch1_std + 1e-10)  # add small value to avoid division by zero
            ch2_data = (ch2_data - ch2_mean) / (ch2_std + 1e-10)


            
            epoch_data = np.stack([ch1_data, ch2_data], axis=0)  # Shape: (2, num_samples)
            epoch_data = epoch_data.transpose(1, 0)  # Shape: (num_samples, 2)

            data_tensor = torch.as_tensor(epoch_data.copy(), dtype=torch.get_default_dtype())
            label_tensor = torch.tensor(label, dtype=torch.long)

            return (data_tensor, label_tensor)

    def downsample(self, data, original_fs=5000, target_fs=100):
        """
        Downsample the data to the target frequency using 1D interpolation.

        Parameters:
        - data: The original data array.
        - original_fs: The original sampling frequency (default is 5000 Hz).
        - target_fs: The target sampling frequency (default is 100 Hz).

        Returns:
        - downsampled_data: The data resampled to the target frequency.
        """
        duration = len(data) / original_fs
        time_original = np.linspace(0, duration, len(data))
        time_target = np.linspace(0, duration, int(duration * target_fs))

        interpolator = interp1d(time_original, data, kind='linear')
        downsampled_data = interpolator(time_target)

        return downsampled_data
    
    def filter_data(self, data, lowcut=5, highcut=30, fs=100.0, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def power_spectrum(self, data, fs=100.0):
        freqs, psd = welch(data, fs,nperseg=150)
        return freqs, np.log1p(psd) 
