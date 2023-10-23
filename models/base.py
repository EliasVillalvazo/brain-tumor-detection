import torch
import random
import numpy as np
import lightning as L
import torch.nn as nn

from torch.nn import functional as F
from torchmetrics.functional import accuracy


class BaseModel(L.LightningModule):
    def __init__(self, learning_rate=0.001, seed=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        super().__init__()

        self.optimizers = None
        self.scheduler = None
        self.learning_rate = learning_rate
        self.val_outputs = []
        self.val_accs = []
        self.test_outputs = []
        self.test_accs = []
        self.loss_fn = nn.CrossEntropyLoss()

    def cross_entropy_loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        return F.nll_loss(logits, labels)
        # return self.loss_fn(logits, labels.data)

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        self.val_outputs.append(loss)
        self.val_accs.append(acc)

        return {"val_step_loss": loss, "val_step_acc": acc}

    def on_validation_epoch_end(self):
        """
        Computes average validation loss
        """
        avg_loss = torch.stack(self.val_outputs).mean()
        avg_acc = torch.stack(self.val_accs).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val_acc", avg_acc, sync_dist=True)
        self.val_outputs.clear()
        self.val_accs.clear()

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu(), task="multiclass", num_classes=10)
        self.test_outputs.append(test_acc)
        return {"test_acc": test_acc}

    def on_test_epoch_end(self):
        """
        Computes average test accuracy score
        """
        avg_test_acc = torch.stack(self.test_outputs).mean()
        self.log("avg_test_acc", avg_test_acc, sync_dist=True)
        self.test_outputs.clear()

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]
