import torch
from torch import nn
from torch.nn import Sequential
from hydra.utils import instantiate


class DeepSpeech2(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, rnn_type, dropout, n_feats, n_tokens, num_layers=1, fc_hidden=512, bidirectional=True):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )

        self.rnn = eval(rnn_type)(
            input_size=n_feats,
            hidden_size=fc_hidden // 2 if bidirectional else fc_hidden,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True, # TODO: 
        )

        self.rnn_type = rnn_type
        self.ln_linear = nn.Sequential(
            nn.LayerNorm(fc_hidden),
            nn.Linear(fc_hidden, n_tokens, bias=False),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        
        print(spectrogram.unsqueeze(1).shape)
        output = self.conv_layers(spectrogram.unsqueeze(1))
        output = output.squeeze(1)
        print(output.shape)

        output = nn.utils.rnn.pack_padded_sequence(spectrogram.transpose(1, 2), spectrogram_length.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(output)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=spectrogram.shape[2], batch_first=True)

        output = self.ln_linear(output)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
