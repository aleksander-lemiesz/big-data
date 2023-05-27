from torch import nn
from app.profiler.classifiers.Classifier import Classifier


class AgeClassifier(Classifier):

    def __init__(self, learning_rate, weights):
        super().__init__(learning_rate, weights)
        self.layers = nn.Sequential(
            nn.Linear(97, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(weights))
        )

    def forward(self, x):
        return self.layers(x)
