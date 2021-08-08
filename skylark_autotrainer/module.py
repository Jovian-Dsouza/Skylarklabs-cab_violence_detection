from typing import Dict
from pytorch_lightning import LightningModule

class TrainerModule(LightningModule):
    def __init__(self, model):
        super().__init__()
    
        self.model = model
        self._first_train_batch_uninit = True
        self._first_val_batch_uninit = True
        self._first_test_batch_uninit = True

    def training_step(self, *args, **kwargs):
        raise NotImplementedError
    
    def validation_step(self, *args, **kwargs):
        raise NotImplementedError

    def test_step(self, *args, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self, *args, **kwargs):
        raise NotImplementedError

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        if 'loss' in tqdm_dict:
            del tqdm_dict['loss']
        return tqdm_dict
    
    def log_metrics(self, metric_dict):
        metric_name = list(metric_dict.keys())[0]
        assert 'train' in metric_name or 'val' in metric_name or 'test' in metric_name, f'\n\nOne of `train` or `valid` or `test` prefix must be present the metric name, but got "{metric_name}".\n'

        self.log_dict(
            metric_dict,
            on_epoch = True,
            prog_bar = True,
        )