from torch import nn
from app.profiler.classifiers.Classifier import Classifier


class RaceClassifier(Classifier):

    def __init__(self, learning_rate, weights):
        super().__init__(learning_rate, weights=weights)
        self.layers = nn.Sequential(
            nn.Linear(97, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, len(weights))
        )

    def forward(self, x):
        return self.layers(x)
