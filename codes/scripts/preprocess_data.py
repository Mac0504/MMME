import os
import numpy as np
import cv2
import pickle
from scipy.signal import butter, lfilter, welch
from sklearn.preprocessing import StandardScaler

# Create necessary directories
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# **1. EEG Data Preprocessing**
class EEGPreprocessor:
    def __init__(self, sampling_rate=128, filter_bands=None):
        """
        Initialize the EEG preprocessor.
        
        Args:
            sampling_rate (int): Sampling rate of the EEG signal (Hz).
            filter_bands (list): List of filter bands, e.g., [(0.5, 4), (4, 8), (8, 14)].
        """
        self.sampling_rate = sampling_rate
        self.filter_bands = filter_bands or [(0.5, 4), (4, 8), (8, 14), (14, 30), (30, 47)]
        self.scaler = StandardScaler()

    def bandpass_filter(self, data, lowcut, highcut):
        """
        Apply a bandpass filter to the EEG data.
        
        Args:
            data (numpy.ndarray): Input EEG data with shape (channels, samples).
            lowcut (float): Low cutoff frequency.
            highcut (float): High cutoff frequency.
        
        Returns:
            numpy.ndarray: Filtered EEG data.
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return lfilter(b, a, data, axis=1)

    def compute_psd(self, data):
        """
        Compute the power spectral density (PSD) of the EEG data.
        
        Args:
            data (numpy.ndarray): Input EEG data with shape (channels, samples).
        
        Returns:
            numpy.ndarray: Frequency power spectral density with shape (channels, len(freqs)).
        """
        psd_features = []
        for channel_data in data:
            freqs, psd = welch(channel_data, fs=self.sampling_rate, nperseg=256)
            psd_features.append(psd)
        return np.array(psd_features)

    def preprocess(self, eeg_data):
        """
        Perform the complete preprocessing pipeline on the EEG data.
        
        Args:
            eeg_data (numpy.ndarray): Raw EEG data with shape (channels, samples).
        
        Returns:
            numpy.ndarray: Preprocessed EEG features.
        """
        all_band_features = []
        for low, high in self.filter_bands:
            filtered_data = self.bandpass_filter(eeg_data, low, high)
            psd_features = self.compute_psd(filtered_data)
            all_band_features.append(psd_features)
        
        # Combine features from all bands
        combined_features = np.concatenate(all_band_features, axis=1)
        
        # Standardize
        combined_features = self.scaler.fit_transform(combined_features)
        return combined_features


# **2. Micro-expression Data Preprocessing**
class MicroExpressionPreprocessor:
    def __init__(self, image_size=(112, 112)):
        """
        Initialize the micro-expression data preprocessor.

        Args:
            image_size (tuple): Target image size (height, width).
        """
        self.image_size = image_size

    def read_image(self, filepath):
        """
        Read and resize an image.
        
        Args:
            filepath (str): Path to the image file.
        
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"File not found: {filepath}")
        image = cv2.resize(image, self.image_size)
        return image

    def compute_optical_flow(self, frame1, frame2):
        """
        Compute optical flow between two frames.
        
        Args:
            frame1 (numpy.ndarray): First frame (grayscale).
            frame2 (numpy.ndarray): Second frame (grayscale).
        
        Returns:
            numpy.ndarray: Optical flow features.
        """
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return magnitude, angle

    def preprocess(self, onset_frame_path, apex_frame_path):
        """
        Perform the complete preprocessing pipeline on the micro-expression frames.
        
        Args:
            onset_frame_path (str): Path to the onset frame image.
            apex_frame_path (str): Path to the apex frame image.
        
        Returns:
            dict: Dictionary containing micro-expression images and optical flow features.
        """
        # Read onset and apex frames
        onset_frame = self.read_image(onset_frame_path)
        apex_frame = self.read_image(apex_frame_path)

        # Convert to grayscale
        onset_gray = cv2.cvtColor(onset_frame, cv2.COLOR_BGR2GRAY)
        apex_gray = cv2.cvtColor(apex_frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        magnitude, angle = self.compute_optical_flow(onset_gray, apex_gray)

        # Return results
        return {
            "onset_frame": onset_frame,
            "apex_frame": apex_frame,
            "optical_flow_magnitude": magnitude,
            "optical_flow_angle": angle,
        }


# **3. Data Saving and Loading**
def save_data(data, save_path):
    """
    Save preprocessed data.
    
    Args:
        data (dict): Preprocessed data.
        save_path (str): Path to save the data.
    """
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def load_data(load_path):
    """
    Load preprocessed data.
    
    Args:
        load_path (str): Path to load the data.
    
    Returns:
        dict: Loaded data.
    """
    with open(load_path, 'rb') as f:
        return pickle.load(f)


# **4. Main Function**
if __name__ == "__main__":
    # Example input paths
    eeg_data_path = "data/eeg_sample.npy"  # Path to EEG data
    onset_frame_path = "data/onset.jpg"  # Path to onset frame
    apex_frame_path = "data/apex.jpg"  # Path to apex frame
    processed_data_save_path = "processed_data.pkl"  # Path to save processed data

    # Create preprocessors
    eeg_processor = EEGPreprocessor()
    me_processor = MicroExpressionPreprocessor()

    # Load EEG data
    eeg_data = np.load(eeg_data_path)  # Assuming data shape is (channels, samples)
    eeg_features = eeg_processor.preprocess(eeg_data)

    # Process micro-expression data
    me_features = me_processor.preprocess(onset_frame_path, apex_frame_path)

    # Save preprocessed data
    processed_data = {
        "eeg_features": eeg_features,
        "me_features": me_features,
    }
    save_data(processed_data, processed_data_save_path)

    print(f"Preprocessed data saved to {processed_data_save_path}")