import torch
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import _METRIC
from typing import Dict
import wandb
import os
from shutil import rmtree
from .utils import get_model_memory, get_model_parameters_count

class ArtifactModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        
        project_name: str,
        model: LightningModule,
        stage: int,

        filename,
        monitor,
        mode,
        save_top_k,

        best_stage2_version: int = None,
        exp_name: str = None,
        wandb_logging: bool = True,
        
        model_name: str = None,
        version: str = None,
        description: str = None,
        **kwargs,
    ) -> None:

        if version != 1:
            assert stage in [4, 2], f'\n\nModel versioning, (version > 0) is only allowed at stage 4.\n'

        CKPT_DIR = os.path.join(os.getcwd(), project_name, f'checkpoints-stage({stage})')
        
        if not os.path.isdir(CKPT_DIR):
            os.makedirs(CKPT_DIR)

        if stage in [4, 2]:
            CKPT_RUN_DIR = os.path.join(CKPT_DIR, model_name, str(version))
        else:
            CKPT_RUN_DIR = os.path.join(CKPT_DIR, model_name)

        if os.path.isdir(CKPT_RUN_DIR) and stage not in [3, 4]:
            # print(f'Deleting previous checkpoints in "{CKPT_RUN_DIR}"')
            rmtree(CKPT_RUN_DIR)

        super(ArtifactModelCheckpoint, self).__init__(
            dirpath = CKPT_RUN_DIR,
            filename = filename,
            monitor = monitor,
            mode = mode,
            save_top_k = save_top_k,
            verbose = True
        )
        self.wandb_logging = wandb_logging
        self.best_stage2_version = best_stage2_version
        
        if not os.path.isdir(CKPT_RUN_DIR):
            os.makedirs(CKPT_RUN_DIR)
            # print(f'Creating checkpoint directory: "{CKPT_RUN_DIR}"')

        self.exp_name = exp_name
        self.stage = stage
        self.model_name = model_name
        self.version = version
        self.description = description
        self.monitor = monitor
        self.mode = mode

        self.metadata = dict(model.hparams)
        self.metadata["parameters"] = get_model_parameters_count(model)
        self.metadata["memory"] = get_model_memory(model)
        for key, value in kwargs:
            self.metadata[key] = value
        
    def _save_model(self, trainer: Trainer, filepath: str) -> None:
        super()._save_model(trainer, filepath)
        
        if trainer.current_epoch == trainer.max_epochs - 1:
            self._save_artificat()
    
    def _save_artificat(self):
        if self.wandb_logging:
            if self.stage == 2:
                wandb.use_artifact(f'{self.model_name}-stage1:latest')
            elif self.stage == 3:
                wandb.use_artifact(f'{self.model_name}-stage2-v{self.best_stage2_version}:latest')
            elif self.version != 1:
                wandb.use_artifact(f'{self.model_name}-stage{self.stage}-v{self.version - 1}:latest')

            artifact = wandb.Artifact(
                name = f'{self.model_name}-stage{self.stage}' if self.stage in [1, 3] else f'{self.model_name}-stage{self.stage}-v{self.version}',
                type = "model",
                description = self.description,
                metadata = self.metadata
            )

            artifact.add_file(self.best_model_path, name = 'model.ckpt')
            wandb.log_artifact(artifact)
        
    def on_keyboard_interrupt(self, trainer, model):
        self._save_artificat()
    
    def on_train_end(self, trainer, model):
        self._save_artificat()
    
    def _update_best_and_save(
        self, current: torch.Tensor, trainer: Trainer, monitor_candidates: Dict[str, _METRIC]
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"))

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates.get("epoch")
            # step = monitor_candidates.get("step")
            # rank_zero_info(
            #     f"Epoch {epoch:d}, global step {step:d}: {self.monitor} reached {current:0.5f}"
            #     f' (best {self.best_model_score:0.5f}), saving model to "{filepath}" as top {k}'
            # )
            rank_zero_info(
                f"Epoch {epoch:d}: {self.monitor} reached {current:0.5f}"
                f' (best {self.best_model_score:0.5f}), saving best model...'
            )
        self._save_model(trainer, filepath)

        if del_filepath is not None and filepath != del_filepath:
            self._del_model(trainer, del_filepath)