import librosa,os
from Log4p.core import *
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioAugmentation:
    def __init__(self, max_shift=2000, noise_factor=0.1):
        self.max_shift = max_shift
        self.noise_factor = noise_factor

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        # 随机裁剪
        start = np.random.randint(0, self.max_shift)
        waveform = waveform[start:]

        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)  # 将一维数组转换为二维数组

        # 添加噪声
        noise = torch.randn_like(torch.from_numpy(waveform)) * self.noise_factor
        augmented_waveform = waveform + noise.numpy()

        # 数值范围控制
        augmented_waveform = np.clip(augmented_waveform, -1.0, 1.0)  # 限制数据范围在[-1, 1]之间

        return augmented_waveform

class AudioDataset(Dataset):
    def __init__(self, audio_files:list = None, labels:list = None, sr: int = 44100, duration: int = 3, transform=None):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.transform = transform
        self.first_shape = -1
        self.log = LogManager().GetLogger("AudioDatasets")

    def __len__(self):
        return len(self.audio_files)


    def __getitem__(self, idx):

        #断言防止条数不一致的报错
        assert len(self.audio_files) == len(self.labels), "音频文件数量和标签数量不匹配"
        try:
            audio_file = self.audio_files[idx]
            label = self.labels[idx]

            waveform, _ = librosa.load(audio_file, sr=self.sr, duration=self.duration, mono=True)  # 使用 librosa 加载音频文件

            mfccs = librosa.feature.mfcc(y=waveform, sr=self.sr, n_mfcc=13)  # 计算MFCCs特征

            if self.first_shape == -1:
                self.first_shape = mfccs.shape[1]
            
            if mfccs.shape[1] != self.first_shape:
                mfccs = np.pad(mfccs, ((0, 0), (0, self.first_shape - mfccs.shape[1])), mode='constant')

            if self.transform:
                waveform = self.transform(waveform)  # 应用额外的转换操作
            
            mfccs = torch.from_numpy(mfccs)
            label = torch.tensor(label)

            return mfccs, label
        except IndexError as e:
            self.log.error(f"音频文件数量和标签数量不匹配!")
            raise