import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ASVspoofDataset(Dataset):
    def __init__(self, base_dir, protocol_file_path, transform=None, is_train=True, sample_rate=16000, max_time_steps=128):
        self.base_dir = base_dir
        self.protocol_file_path = protocol_file_path
        self.transform = transform
        self.is_train = is_train
        self.sample_rate = sample_rate
        self.max_time_steps = max_time_steps
        
        # Carga del protocolo
        self.protocol_df = pd.read_csv(protocol_file_path, sep=' ', header=None, names=['speaker', 'filename', 'system_id', 'null', 'label'])
        self.protocol_df['target'] = self.protocol_df['label'].apply(lambda x: 0 if x == 'bonafide' else 1)

    def __len__(self):
        return len(self.protocol_df)

    def __getitem__(self, idx):
        row = self.protocol_df.iloc[idx]
        filename = row['filename']
        target = row['target']
        file_path = os.path.join(self.base_dir, filename + '.flac')
        
        try:
            feature_tensor = self._process_audio(file_path)
            return feature_tensor, torch.tensor(target, dtype=torch.float32)
        except Exception as e:
            # Retorno de tensor vacío en caso de archivo corrupto para mantener flujo
            return torch.zeros((1, 128, self.max_time_steps)), torch.tensor(target, dtype=torch.float32)

    def _process_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        
        if self.transform and self.is_train:
            y = self.transform(y)
            
        # Normalización de amplitud
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
            
        # Extracción de Log-Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalización estándar (Centrado)
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-6
        log_mel = (log_mel - mean) / std
        
        # Ajuste temporal (Padding/Truncating)
        time_steps = log_mel.shape[1]
        if time_steps < self.max_time_steps:
            pad_width = self.max_time_steps - time_steps
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            if self.is_train:
                start = np.random.randint(0, time_steps - self.max_time_steps)
                log_mel = log_mel[:, start : start + self.max_time_steps]
            else:
                log_mel = log_mel[:, :self.max_time_steps]
                
        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)