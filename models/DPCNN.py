import torch
import torch.nn as nn


class AvgPoolLSTM(nn.Module):
    def __init__(self, num_labels, vocab_size, embed_size, hidden_size, num_layers):
        super(AvgPoolLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=embed_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True)
        self.pooling = nn.AdaptiveAvgPool2d((1, None))
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        embeds = self.embed(x.long())
        output, _ = self.rnn(embeds)
        output = self.pooling(output)
        output = output[:, -1, :]
        output = self.fc(output)
        return output


class DPCNNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, downsample):
        super(DPCNNResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, padding=padding)
        )
        self.pool = nn.MaxPool1d(2)
        self.downsample = downsample

    def forward(self, x):
        output = self.residual(x)
        output = x + output
        if self.downsample:
            output = self.pool(output)
        return output


class DPCNN(nn.Module):
    def __init__(self, num_labels, vocab_size, embed_size, hidden_size, num_layers):
        super(DPCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.cnn = nn.Conv1d(
            in_channels=embed_size, out_channels=hidden_size,
            kernel_size=1, padding=0
        )
        self.residual_layer = self._make_layer(num_layers, hidden_size, kernel_size=3, padding=1, downsample=True)
        self.globalpool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Linear(hidden_size, num_labels)

    def _make_layer(self, num_layers, channels, kernel_size, padding, downsample):
        layers = []
        for i in range(num_layers - 1):
            layers.append(DPCNNResidualBlock(channels, kernel_size, padding, downsample))
        layers.append(DPCNNResidualBlock(channels, kernel_size, padding, downsample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        embeds = self.embed(x.long())
        embeds = embeds.permute(0, 2, 1)
        output = self.cnn(embeds)
        output = self.residual_layer(output)
        output = self.globalpool(output).squeeze()
        output = self.fc(output)
        return output