import torchaudio
from torch import Tensor, nn
import numpy as np


class TimeStretch(nn.Module):
    def __init__(self, min_value, max_value, p, *args, **kwargs):
        super().__init__()
        
        self.complex_spec = torchaudio.transforms.Spectrogram(power=None)
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.p = p

    def __call__(self, data: Tensor):
        if np.random.random() > self.p:
            return data
        
        x = data.unsqueeze(1)
        random_value = np.random.uniform(self.min_value, self.max_value)
        return self._aug(self.complex_spec(x), random_value).squeeze(1)
