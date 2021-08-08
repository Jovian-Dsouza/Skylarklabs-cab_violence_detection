import torch
from torch import nn, optim
import torch.nn.functional as F
import torchmetrics

from skylark_autotrainer import AutoTrainer, TrainerModule

class ViolenceDetectionModule(TrainerModule):
    def __init__(self, model):
        super().__init__(model)

        self._accuracy = torchmetrics.Accuracy(num_classes = model.hparams.num_classes, average = 'weighted')
        self._precision = torchmetrics.Precision(num_classes = model.hparams.num_classes, average = 'weighted')
        self._recall = torchmetrics.Recall(num_classes = model.hparams.num_classes, average = 'weighted')
        self._f1 = torchmetrics.F1(num_classes = model.hparams.num_classes, average = 'weighted')
    
    def training_step(self, batch, batch_idx):
        x = batch['video']
        y = batch['label']
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
        x = batch['video']
        y = batch['label']
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
        x = batch['video']
        y = batch['label']
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
    
    def on_train_start(self):
        # freezing params from pretrained_block
        for name, param in self.model.pretrained_block.named_parameters():
            param.requires_grad = False
            # print(f'Freezing {name}')
        print('Freezing layers')
        
    def on_epoch_start(self):
        # unfreezing params from pretrained_block at 5th epoch
        if self.trainer.current_epoch == 5:
            for name, param in self.model.pretrained_block.named_parameters():
                param.requires_grad_()
                # print(f'Unfreezing {name}')
            print('Unfreezing layers')

    def configure_optimizers(self):
        if self.model.hparams.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), self.model.hparams.lr, momentum = 0.9)
        
        elif self.model.hparams.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), self.model.hparams.lr)
        
        elif self.model.hparams.optimizer == 'adamax':
            return optim.Adamax(self.model.parameters(), self.model.hparams.lr)