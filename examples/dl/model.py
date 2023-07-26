import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class CNNModelConfig:
    input_size: int


class CNNModel(nn.Module):
    def __init__(self, config: CNNModelConfig):
        super().__init__()
        self.config = config

        # input 2 leads with 4096 samples (= 20 seconds)
        self.block1 = ResidualBlock(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.block2 = ResidualBlock(in_channels=16, out_channels=16, kernel_size=7, padding=3)
        self.block3 = ResidualBlock(in_channels=16, out_channels=16, kernel_size=7, padding=3)

        self.dropout = nn.Dropout(p=0.2)

        self.block4 = ResidualBlock(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.block5 = ResidualBlock(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.block6 = ResidualBlock(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        self.block7 = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.block8 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.block9 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # invert dimension for convolution on leads (2)
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=8, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        size = int(config.input_size / 8)

        self.fc1 = nn.Linear(size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # block 1
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.dropout(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.dropout(x)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=2, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv_shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        shortcut = self.conv_shortcut(x_in)
        shortcut = self.maxpool(shortcut)

        x += shortcut

        return x
