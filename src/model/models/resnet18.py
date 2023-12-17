from torch import nn


class ResNet18Block(nn.Module):
    def __init__(self, c_in, c_out, stride=1, conv3=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv3 = (
            nn.Conv2d(c_in, c_out, 1, stride=stride) if conv3 else None
        )

    def forward(self, X):
        Y = self.bn1(self.conv1(X))
        Y = self.relu(Y)
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return self.relu(Y + X)


class ResNet18(nn.Module):
    def __init__(self, c_in=3, c_out=3, activation="linear") -> None:
        super().__init__()
        self.out = c_out
        self.seq = nn.Sequential(
            nn.Conv2d(c_in, 64, 7),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3),
            ResNet18Block(c_in=64, c_out=64),
            nn.ReLU(),
            ResNet18Block(c_in=64, c_out=64),
            nn.ReLU(),
            ResNet18Block(c_in=64, c_out=128, stride=2, conv3=True),
            nn.ReLU(),
            ResNet18Block(c_in=128, c_out=128),
            nn.ReLU(),
            ResNet18Block(c_in=128, c_out=256, stride=2, conv3=True),
            nn.ReLU(),
            ResNet18Block(c_in=256, c_out=256),
            nn.ReLU(),
            ResNet18Block(c_in=256, c_out=512, stride=2, conv3=True),
            nn.ReLU(),
            ResNet18Block(c_in=512, c_out=512),
            nn.ReLU(),
            nn.AvgPool2d(5),
            nn.Conv2d(512, c_out, 1),
        )

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax()
        else:
            self.activation = lambda x: x

    def forward(self, X):
        return self.activation(self.seq(X)).view(-1, self.out)

