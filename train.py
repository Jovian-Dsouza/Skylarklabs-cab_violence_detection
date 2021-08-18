from skylark_autotrainer import AutoTrainer

from DataModule import VideoDataModule
from VideoTrainerModule import *
from callbacks import *

from models.R2Plus1D_18_with_Conv2Plus1D import R2Plus1D_18_with_Conv2Plus1D
from models.R2Plus1D_18_with_Conv3DNoTemporal import R2Plus1D_18_with_Conv3DNoTemporal
from models.MC3_18_with_Conv2Plus1D import MC3_18_with_Conv2Plus1D
from models.MC3_18_with_Conv3DNoTemporal import MC3_18_with_Conv3DNoTemporal
from models.R3D_18_with_Conv2Plus1D import R3D_18_with_Conv2Plus1D
from models.R3D_18_with_Conv3DNoTemporal import R3D_18_with_Conv3DNoTemporal
from models.R2Plus1D_18_full import R2Plus1D_18_full

"""## Training configuration"""

datamodule = VideoDataModule(data_path='/content/CAR_VIOLENCE_DATASET_final',
                                 clip_duration=3.2, # 32 frames at 10 fps
                                 batch_size=8,
                                 pin_memory=True)
print(datamodule)

AutoTrainer(
    project_name = 'cab_violence_detection-Test5',
    trainer_module = ViolenceDetectionModule,
    datamodule = datamodule,
    models = [
        {
            'model': R2Plus1D_18_with_Conv2Plus1D,
            'init': {'num_classes': 2, 'lr': 8e-3, 'optimizer': 'adamax'},
            'hyperparameters': {'method': 'grid', 'lr': [8e-4],
                                'optimizer': ['adam']},
            'description': 'Pretrained r2plus1d-18 with Conv2Plus1D',
        },
    ],

    checkpoint = {'filename': '{epoch}-{val_acc:.4f}', 'monitor': 'val_f1', 'mode': 'max'},
    evaluation_metric = {'monitor': 'test_f1', 'mode': 'max'},
    gpus = -1,

    precision = 16,
    datasets_limits = (1, 1, 1),
    max_epochs = 1,
    # overfit_batches = 1,
    # overfit_epochs = 1,
    # wandb_logging = False

    stages = {
        'stage1': {
                    # 'precision': 16, 
                    # 'datasets_limits': datamodule.cal_batches(0.5, 1.0, 1.0),
                    # 'max_epochs': 10,
                    },
        'stage2': {
                    # 'precision': 16,
                    # 'datasets_limits': datamodule.cal_batches(0.5, 0.8, 1.0),
                    # 'max_epochs': 15,
                  },
        'stage3': {
                    # 'callbacks':[UnfreezingOnPlateau(monitor="train_loss", patience=2, mode="min")], 
                    'callbacks':[Unfreezing(epoch=5)],
                    'precision': 32,
                    'datasets_limits': datamodule.cal_batches(1.0, 1.0, 1.0),
                    'max_epochs': 80,
                },
    },
)

