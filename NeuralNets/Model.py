import torch.nn as nn


def makeBlock(in_features: int, out_features: int, dropout=None):
    if dropout is None:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout),
            nn.SiLU(),
        )


class Model(nn.Module):

    def __init__(
        self, input_size, output_size, num_blocks=200, width=512, dropout=None
    ):
        super().__init__()
        self.width = width
        if dropout is not None:
            self.input = nn.Sequential(
                nn.Linear(input_size, width),
                nn.BatchNorm1d(width),
                nn.Dropout(dropout),
                nn.SiLU(),
            )
        else:
            self.input = nn.Sequential(
                nn.Linear(input_size, width), nn.BatchNorm1d(width), nn.SiLU()
            )
        self.hiddenLayers = []

        for i in range(num_blocks):
            self.hiddenLayers.append(makeBlock(width, width, dropout=dropout))

        self.hidden = nn.Sequential(*self.hiddenLayers)
        self.output = nn.Sequential(nn.Linear(width, output_size))

    def forward(self, X):
        X = self.input(X)
        X = self.hidden(X)
        X = self.output(X)
        return X

    def reset_output(self):
        self.output = nn.Sequential(nn.Linear(self.width, 1), nn.Sigmoid())
