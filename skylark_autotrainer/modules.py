import torch.nn.functional as F
from torch import optim
import torchmetrics
from .module import TrainerModule

class SoftmaxCategoricalClassificationModule(TrainerModule):
    def __init__(self, model):
        super().__init__(model)

        self._accuracy = torchmetrics.Accuracy(num_classes = model.hparams.num_classes, average = 'weighted')
        self._precision = torchmetrics.Precision(num_classes = model.hparams.num_classes, average = 'weighted')
        self._recall = torchmetrics.Recall(num_classes = model.hparams.num_classes, average = 'weighted')
        self._f1 = torchmetrics.F1(num_classes = model.hparams.num_classes, average = 'weighted')
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat = y_hat.argmax(1)

        self.log_metrics(
            {
                'train_loss': loss,
                'train_acc': self._accuracy(y_hat, y),
                'train_precision': self._precision(y_hat, y),
                'train_recall': self._recall(y_hat, y),
                'train_f1': self._f1(y_hat, y),
            },
        )

        if self._first_train_batch_uninit:
            self.sample_train_batch = batch
            self._first_train_batch_uninit = True

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat = y_hat.argmax(1)

        self.log_metrics(
            {
                'val_loss': loss,
                'val_acc': self._accuracy(y_hat, y),
                'val_precision': self._precision(y_hat, y),
                'val_recall': self._recall(y_hat, y),
                'val_f1': self._f1(y_hat, y),
            },
        )

        if self._first_val_batch_uninit:
            self.sample_val_batch = batch
            self._first_val_batch_uninit = True
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat = y_hat.argmax(1)

        self.log_metrics(
            {
                'test_loss': loss,
                'test_acc': self._accuracy(y_hat, y),
                'test_precision': self._precision(y_hat, y),
                'test_recall': self._recall(y_hat, y),
                'test_f1': self._f1(y_hat, y),
            },
        )

        if self._first_test_batch_uninit:
            self.sample_test_batch = batch
            self._first_test_batch_uninit = True
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), self.model.hparams.lr)

class SigmoidBinaryClassificationModule(TrainerModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'train_loss': loss,
                'train_acc': acc,
            },
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'val_loss': loss,
                'val_acc': acc,
            },
        )
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'test_loss': loss,
                'test_acc': acc,
            },
        )
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), self.model.hparams.lr)

class SigmoidCategoricalClassificationModule(TrainerModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'train_loss': loss,
                'train_acc': acc,
            },
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'val_loss': loss,
                'val_acc': acc,
            },
        )
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'test_loss': loss,
                'test_acc': acc,
            },
        )
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), self.model.hparams.lr)

class FaceRecognitionModule(TrainerModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'train_loss': loss,
                'train_acc': acc,
            },
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'val_loss': loss,
                'val_acc': acc,
            },
        )
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'test_loss': loss,
                'test_acc': acc,
            },
        )
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), self.model.hparams.lr)

class ImageTransformationModule(TrainerModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'train_loss': loss,
                'train_acc': acc,
            },
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'val_loss': loss,
                'val_acc': acc,
            },
        )
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'test_loss': loss,
                'test_acc': acc,
            },
        )
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), self.model.hparams.lr)

class ObjectDetectionModule(TrainerModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'train_loss': loss,
                'train_acc': acc,
            },
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'val_loss': loss,
                'val_acc': acc,
            },
        )
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log_metrics(
            {
                'test_loss': loss,
                'test_acc': acc,
            },
        )
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), self.model.hparams.lr)