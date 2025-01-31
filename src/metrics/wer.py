from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer
import numpy as np

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):

        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)

            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            # pred_text = self.text_encoder.decode(log_prob_vec[:length])

            wers.append(calc_wer(target_text, pred_text))

            
        print(f'argmax (wer): {sum(wers) / len(wers)}')
        return sum(wers) / len(wers)
    

class BeamWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):

        wers = []
        # predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)

            pred_inds = self.text_encoder.ctc_beam_search_ind(log_prob_vec[:length].exp().detach().cpu().numpy(), 3)[-1][0]
            pred_text = self.text_encoder.ind2text(list(pred_inds))
            wers.append(calc_wer(target_text, pred_text))

        print(f'beam (wer): {sum(wers) / len(wers)}')
        return sum(wers) / len(wers)
