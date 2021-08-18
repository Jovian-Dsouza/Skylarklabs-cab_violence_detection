from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.base import rank_zero_experiment
import wandb
import os

class CustomWandbLogger(WandbLogger):
    def __init__(
        self,
        project_name: str,
        model_name: str,
        model_stage: str,
        model_version: str,
        sweep: bool = False,
    ):
        run_id = str(hash(f'{project_name}{model_name}{model_stage}{model_version}')) if model_stage != 4 else None
        if run_id:
            run_id.replace('-', '')
            
        super().__init__(
            project = project_name if not sweep else None,
            group = 'model',
            name = f'{model_name}-stage({model_stage})' if model_stage in [1, 3] else f'{model_name}-stage({model_stage})-v({model_version})',
            id = run_id if not sweep else None,
            job_type = f'model-stage({model_stage})',
            log_model = False,
        )

        self.model_stage = model_stage
    
    def set_model(self, model: LightningModule):
        self.watch(model)
        # self.model = model

    @rank_zero_only
    def log_metrics(self, metrics, step = None) -> None:
        metrics.pop('epoch')

        for metric_name in metrics:
            if 'stage' not in metric_name:
                metric_value = metrics.pop(metric_name)
                if 'train_' in metric_name:
                    metric_name = metric_name.replace('train_', 'training/')
                elif 'val_' in metric_name:
                    metric_name = metric_name.replace('val_', 'validation/')
                elif 'test_' in metric_name:
                    metric_name = metric_name.replace('test_', 'testing/')

                metrics[f'stage({self.model_stage})/{metric_name}'] = metric_value

        self.experiment.log(metrics)
    
    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            if self._offline:
                os.environ['WANDB_MODE'] = 'dryrun'
            self._experiment = wandb.init(**self._wandb_init) if wandb.run is None else wandb.run
        
        # if getattr(self._experiment, "define_metric", None):
        #     self._experiment.define_metric("batch_step", hidden = True)
        #     self._experiment.define_metric("*", step_metric = "batch_step", step_sync = True)

        return self._experiment