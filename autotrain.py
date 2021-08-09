import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchmetrics

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import Callback

from skylark_autotrainer import AutoTrainer, TrainerModule
from skylark_autotrainer.modules import SoftmaxCategoricalClassificationModule

import wandb

from DataModule import VideoDataModule
import models.model1 as model1
import models.model2 as model2
from VideoTrainerModule import *
from callbacks import *


"""## Training configuration"""

class SplitDataModule():
    def __init__(self, data_splits, batch_size):
      self.n = [data_split//batch_size for data_split in data_splits]

    def cal(self, train, val, test):
      return int(self.n[0] * train), int(self.n[1] * val), int(self.n[2] * test)


datamodule = VideoDataModule(data_path='/content/CAR_VIOLENCE_DATASET_final',
                                 clip_duration=3.2, # 32 frames at 10 fps
                                 batch_size=16,
                                 num_workers=0,  
                                 pin_memory=True)
print(datamodule)
split_data = SplitDataModule((812, 197, 58), batch_size = 16)

autotrainer = AutoTrainer(
    project_name = 'cab_violence_detection-Test',
    trainer_module = ViolenceDetectionModule,
    datamodule = datamodule,
    models = [
        {
            'model': model1.VideoModel,
            'init': {'num_classes': 2, 'lr': 8e-3, 'optimizer': 'adamax'},
            'hyperparameters': {'method': 'grid', 'lr': [1e-3, 2e-3, 4e-3],
                                'optimizer': ['adam', 'sgd', 'adamax']},
            'description': 'Pretrained r2plus1d-18 with Conv2Plus1D',
        },
    ],
    checkpoint = {'filename': '{epoch}-{val_acc:.4f}', 'monitor': 'val_f1', 'mode': 'max'},
    evaluation_metric = {'monitor': 'test_f1', 'mode': 'max'},
    precision = 32,
    gpus = -1,
    max_epochs = 1,
    datasets_limits = split_data.cal(1.0, 1.0, 1.0),

    #overfit_batches = 1,
    #overfit_epochs = 1,
    stages = {
        'stage1': {
                    'precision': 16, 
                    'datasets_limits': split_data.cal(0.8, 1.0, 1.0),
                    'max_epochs': 5,
                    },
        'stage2': {
                    'precision': 16,
                    'datasets_limits': split_data.cal(0.5, 0.8, 1.0),
                    'max_epochs': 15,
                  },
        'stage3': {
                    'callbacks':[UnfreezingOnPlateau(monitor="train_loss", patience=2, mode="min")], 
                    'max_epochs': 80,
                },
    },
    restart = True, # restarting is supported now
)

autotrainer.start()
