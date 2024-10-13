import torchaudio
from torch import Tensor, nn
import numpy as np


class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param, p, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.p = p

    def __call__(self, data: Tensor):
        if np.random.random() > self.p:
            return data
        # x = data.unsqueeze(1)
        # return self._aug(x).squeeze(1)
        return self._aug(data)
