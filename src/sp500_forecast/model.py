from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class AdditiveSelfAttention(nn.Module):
    """Additive self-attention layer following the paper's SAM equations.

    The paper scores pairwise hidden states as tanh(W_i h_i + W_j h_j + b),
    then applies softmax over j and forms H_i = sum_j a_ij h_j. A small
    projection vector turns the tanh feature into a scalar score.
    """

    def __init__(self, channels: int, attention_size: int | None = None):
        super().__init__()
        attention_size = attention_size or channels
        self.w_i = nn.Linear(channels, attention_size, bias=False)
        self.w_j = nn.Linear(channels, attention_size, bias=True)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.w_i(x).unsqueeze(2)
        right = self.w_j(x).unsqueeze(1)
        scores = self.v(torch.tanh(left + right)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.downsample(x))


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        *,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for dilation in dilations:
            layers.append(
                TemporalBlock(
                    current_channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_channels = channels
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BiLSTMSAMTCN(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        tcn_channels: int,
        tcn_kernel_size: int,
        tcn_dilations: list[int],
        dropout: float,
        variant: str = "proposed",
    ):
        super().__init__()
        self.variant = variant
        self.use_recurrent = variant != "no_bilstm"
        self.use_attention = variant != "no_attention"
        self.use_tcn = variant != "no_tcn"
        self.use_cnn = variant == "bilstm_sam_cnn"
        self.bidirectional = variant != "lstm_sam_tcn"

        if variant not in {
            "proposed",
            "no_attention",
            "no_tcn",
            "no_bilstm",
            "lstm_sam_tcn",
            "bilstm_sam_cnn",
        }:
            raise ValueError(f"Unsupported model variant: {variant}")

        if self.use_recurrent:
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            feature_channels = hidden_size * (2 if self.bidirectional else 1)
        else:
            feature_channels = hidden_size * 2
            self.input_projection = nn.Linear(input_size, feature_channels)

        self.attention = (
            AdditiveSelfAttention(feature_channels) if self.use_attention else nn.Identity()
        )

        if self.use_cnn:
            padding = tcn_kernel_size - 1
            self.sequence_model = nn.Sequential(
                nn.Conv1d(feature_channels, tcn_channels, tcn_kernel_size, padding=padding),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            output_channels = tcn_channels
        elif self.use_tcn:
            self.sequence_model = TemporalConvNet(
                feature_channels,
                tcn_channels,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                dropout=dropout,
            )
            output_channels = tcn_channels
        else:
            self.sequence_model = nn.Identity()
            output_channels = feature_channels

        self.head = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_recurrent:
            sequence, _ = self.recurrent(x)
        else:
            sequence = self.input_projection(x)
        attended = self.attention(sequence)
        if self.use_tcn or self.use_cnn:
            sequence_input = attended.transpose(1, 2)
            sequence_output = self.sequence_model(sequence_input).transpose(1, 2)
        else:
            sequence_output = self.sequence_model(attended)
        return self.head(sequence_output[:, -1, :])
