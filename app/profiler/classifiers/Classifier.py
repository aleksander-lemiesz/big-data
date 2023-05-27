import time
import io

import numpy as np
import pandas as pd
import seaborn as sn
import pytorch_lightning as pl
import torchmetrics
from torch import optim, tensor
from torch.nn import functional as f
from torchvision import transforms as t
from PIL import Image
from matplotlib import pyplot as plt


def create_conf_matrix(conf_matrix, no_classes):
    df_cm = pd.DataFrame(
        conf_matrix,
        index=np.arange(no_classes),
        columns=np.arange(no_classes))
    plt.figure()
    sn.set(font_scale=1.4)
    s = sn.heatmap(df_cm, annot=True, annot_kws={'size': 14}, fmt='g')
    s.set(xlabel='Predicted', ylabel='GT')
    buf = io.BytesIO()

    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    return t.ToTensor()(im)


class Classifier(pl.LightningModule):

    def __init__(self, learning_rate, weights):
        super().__init__()
        self.learning_rate = learning_rate
        self.no_classes = len(weights)
        self.weights = weights

        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_confusion = torchmetrics.ConfusionMatrix(num_classes=self.no_classes)
        self.test_confusion = torchmetrics.ConfusionMatrix(num_classes=self.no_classes)

        self.epoch = 0.0
        self.start = None

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_start(self):
        self.start = time.time()

    def on_train_epoch_start(self) -> None:
        self.epoch += 1
        self.log('epoch', self.epoch)

    def on_train_epoch_end(self):
        self.log('train_time', time.time() - self.start)

        tb = self.logger.experiment
        conf_matrix = self.train_confusion.compute().detach().cpu().numpy().astype(np.int)
        im = create_conf_matrix(conf_matrix, self.no_classes)
        tb.add_image('Train_Confusion_Matrix', im, global_step=self.current_epoch)
        self.train_confusion.reset()

    def on_test_epoch_end(self):
        tb = self.logger.experiment
        conf_matrix = self.test_confusion.compute().detach().cpu().numpy().astype(np.int)
        im = create_conf_matrix(conf_matrix, self.no_classes)
        tb.add_image('Test_Confusion_Matrix', im, global_step=self.current_epoch)
        self.test_confusion.reset()

    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        predicted = self.forward(x)
        loss = f.cross_entropy(predicted, target, weight=tensor(self.weights))

        # log metrics
        self.train_accuracy(predicted, target)
        self.train_confusion.update(predicted, target)
        self.log('Accuracy', {'Train': self.train_accuracy}, on_epoch=True, on_step=False, prog_bar=True)
        self.log('Loss', {'Train': loss})
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        predicted = self.forward(x)
        loss = f.cross_entropy(predicted, target)

        # log metrics
        self.val_accuracy(predicted, target)
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False)
        self.log('Accuracy', {'Val': self.val_accuracy}, on_epoch=True, on_step=False, prog_bar=True)
        self.log('Loss', {'Val': loss}, on_epoch=True, on_step=False)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, target = test_batch
        predicted = self.forward(x)
        loss = f.cross_entropy(predicted, target)

        # log metrics
        self.test_accuracy(predicted, target)
        self.test_confusion.update(predicted, target)
        self.log('Accuracy', {'Test': self.test_accuracy}, on_epoch=True, on_step=False, prog_bar=True)
        self.log('Loss', {'Test': loss})
        return loss
