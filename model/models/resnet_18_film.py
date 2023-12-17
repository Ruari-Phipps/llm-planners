from torch import nn


class FiLM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        return gamma * x + beta


class ResNet18BlockFiLM(nn.Module):
    def __init__(self, c_in, c_out, stride=1, conv3=False) -> None:
        super().__init__()
        self.c_out = c_out
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.film = FiLM()
        self.conv3 = (
            nn.Conv2d(c_in, c_out, 1, stride=stride) if conv3 else None
        )

    def forward(self, X, gamma, beta):
        Y = self.bn1(self.conv1(X))
        Y = self.relu(Y)
        Y = self.bn2(self.conv2(Y))
        Y = self.film(Y, gamma, beta)
        if self.conv3:
            X = self.conv3(X)
        return self.relu(Y + X)

    def get_channels(self):
        return self.c_out


class ResNet18FiLM(nn.Module):
    def __init__(
        self, c_in=3, c_out=3, film_size=3, activation="linear"
    ) -> None:
        super().__init__()
        self.c_out = c_out

        self.c1 = nn.Conv2d(c_in, 64, 7)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [
                ResNet18BlockFiLM(c_in=64, c_out=64),
                ResNet18BlockFiLM(c_in=64, c_out=64),
                ResNet18BlockFiLM(c_in=64, c_out=128, stride=2, conv3=True),
                ResNet18BlockFiLM(c_in=128, c_out=128),
                ResNet18BlockFiLM(c_in=128, c_out=256, stride=2, conv3=True),
                ResNet18BlockFiLM(c_in=256, c_out=256),
                ResNet18BlockFiLM(c_in=256, c_out=512, stride=2, conv3=True),
                ResNet18BlockFiLM(c_in=512, c_out=512),
            ]
        )
        self.avgpool = nn.AvgPool2d(5)
        self.conv_out_1 = nn.Conv2d(512, 256, 1)
        self.conv_out_2 = nn.Conv2d(256, c_out, 1)

        self.channels = [
            res_block.get_channels() for res_block in self.res_blocks
        ]

        film_count = 2 * sum(self.channels)
        # self.film_generator = nn.Linear(film_size, film_count)
        # self.film_generator2 = nn.Linear(film_count, film_count)
        self.film_generator = nn.Linear(film_size, film_size)
        self.film_generator2 = nn.Linear(film_size, film_size)
        self.film_generator3 = nn.Linear(film_size, film_count)
        self.film_generator4 = nn.Linear(film_count, film_count)

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax()
        else:
            self.activation = lambda x: x

    def forward(self, X, input):

        film = self.relu(self.film_generator(input))
        # film = self.film_generator2(film)
        film = self.relu(self.film_generator2(film))
        film = self.relu(self.film_generator3(film))
        film = self.film_generator4(film)

        X = self.c1(X)
        X = self.bn1(X)
        X = self.maxpool(X)

        tot = 0
        for i, res_block in enumerate(self.res_blocks):
            channel = self.channels[i]
            g_b = film[:, tot : tot + 2 * channel]
            gamma, beta = g_b[:, :channel], g_b[:, channel:]

            tot += channel

            X = res_block(X, gamma, beta)
            X = self.relu(X)

        X = self.avgpool(X)
        X = self.conv_out_1(X)
        X = self.conv_out_2(X)
        X = self.activation(X)

        return X.view(-1, self.c_out)

