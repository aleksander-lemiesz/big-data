import os

import torch
from pathlib import PurePath

from app.profiler.classifiers.AgeClassifier import AgeClassifier
from app.profiler.classifiers.RaceClassifier import RaceClassifier
from app.profiler.classifiers.SexClassifier import SexClassifier
from app.profiler.dataloaders.SRADataloader import SRADataloader


class SRAClassifier:

    def __init__(self, load_patches, no_workers):
        self.models = {}

        for p in load_patches:
            checkpoint = torch.load(p)
            model = {
                's.ckpt': SexClassifier(1, [1] * 2),
                'r.ckpt': RaceClassifier(1, [1] * 4),
                'a.ckpt': AgeClassifier(1, [1] * 5)
            }.get(PurePath(p).name)
            model.load_state_dict(checkpoint['state_dict'])
            self.models[PurePath(p).name.split('.')[0]] = model
        self.models = {k: v.eval() for k, v in self.models.items()}

    def predict(self, x):
        predictions = {
            k: torch.max(model(x).data, 1)[1]
            for k, model in self.models.items()
        }
        return predictions
