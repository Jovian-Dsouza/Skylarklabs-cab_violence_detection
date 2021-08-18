from pytorch_lightning.callbacks.base import Callback
import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning import seed_everything

from .logger import CustomWandbLogger
from .callbacks import ArtifactModelCheckpoint

import os
import dill
import wandb
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from termcolor import colored

import argparse

parser = argparse.ArgumentParser(description = 'Skylark Autotraining Pipeline for auto training and producing production ready model.')
parser.add_argument('--stage', type = int, default = 0, help = 'Stage value')
parser.add_argument('--deploy', action = 'store_true', help = 'Deploy project to production')
parser.add_argument('--restart', action = 'store_true', help = 'Restart auto training')
parser.add_argument('--opset', type = int, default = 12, help = 'ONNX Opset Version')
args = parser.parse_args()

# assert not(args.stage == 0 and not args.deploy), f'\n\nYou must pass either --stage or --deploy, found both None.\n'

class AutoTrainer:
    def __init__(
        self,
        project_name: str,
        trainer_module: LightningModule,
        models: List,
        datamodule: LightningDataModule = None,
        checkpoint: Dict = None,
        max_epochs: int = None,
        evaluation_metric: str = None,
        callbacks: List[Callback] = None,
        gpus : Union[List, int] = -1,
        datasets_limits: Tuple[float] = (1.0, 1.0, 1.0),
        stages: Dict = None,
        precision: int = 32,
        seed: int = 0,
        min_epochs: int = 1,
        wandb_logging: bool = True,
        overfit_batches: int = 10,
        overfit_epochs: int = 10,
        **kwargs
    ) -> None:

        assert hasattr(datamodule, 'train_dataloader'), f'\n\nYou must define `train_dataloader` in your datamodule.\n'
        assert hasattr(datamodule, 'val_dataloader'), f'\n\nYou must define `val_dataloader` in your datamodule.\n'
        assert hasattr(datamodule, 'test_dataloader'), f'\n\nYou must define `test_dataloader` in your datamodule.\n'
            
        seed_everything(seed)

        os.makedirs(os.path.join(project_name), exist_ok = True)
        self.path = os.path.join(project_name, 'autotrainer.ckpt')
        self.trainer_module = trainer_module
        self.datamodule = datamodule
        self.project_name = project_name

        self.gpus = gpus
        self.datasets_limits = datasets_limits
        self.stages = stages
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.precision = precision
        self.callbacks = callbacks
        self.overfit_batches = overfit_batches
        self.overfit_epochs = overfit_epochs
        self.wandb_logging = wandb_logging
        self.kwargs = kwargs

        self.restart = args.restart
        
        for model in models:
            model['model_class'] = model.pop('model')
            model['model_class_name'] = model['model_class'].__name__

        self.models = models
        self.checkpoint = checkpoint
        self.evaluation_metric = evaluation_metric

        self._first_init = True

        self._start()
    
    def _read_buffer(self):
        if self._first_init:
            if not self.restart and os.path.isfile(self.path):
                self.buffer = torch.load(self.path)
            else:
                self.buffer = {}
                self.buffer['last_stage'] = 0
            self._first_init = False
        elif os.path.isfile(self.path):
            self.buffer = torch.load(self.path)
    
    def _get_stage_variables(self, stage):
        if self.stages:
            if stage in self.stages:
                datasets_limits = self.stages[stage]['datasets_limits'] if 'datasets_limits' in self.stages[stage] else self.datasets_limits
                max_epochs = self.stages[stage]['max_epochs'] if 'max_epochs' in self.stages[stage] else self.max_epochs
                min_epochs = self.stages[stage]['min_epochs'] if 'min_epochs' in self.stages[stage] else self.min_epochs
                precision = self.stages[stage]['precision'] if 'precision' in self.stages[stage] else self.precision
                gpus = self.stages[stage]['gpus'] if 'gpus' in self.stages[stage] else self.gpus
                callbacks = self.stages[stage]['callbacks'] if 'callbacks' in self.stages[stage] else self.callbacks
            else:
                datasets_limits = self.datasets_limits
                max_epochs = self.max_epochs
                min_epochs = self.min_epochs
                precision = self.precision
                gpus = self.gpus
                callbacks = self.callbacks
        else:
            datasets_limits = self.datasets_limits
            max_epochs = self.max_epochs
            min_epochs = self.min_epochs
            precision = self.precision
            gpus = self.gpus
            callbacks = self.callbacks
        
        if not callbacks:
            callbacks = []
        
        assert datasets_limits is not None, f'\n\nYou must pass `datasets_limits` during initialization or in the stage argument, but got `None`.\n'
        assert max_epochs is not None, f'\n\nYou must pass `max_epochs` during initialization or in the stage argument, but got `None`.\n'
        assert min_epochs is not None, f'\n\nYou must pass `min_epochs` during initialization or in the stage argument, but got `None`.\n'
        assert precision is not None, f'\n\nYou must pass `precision` during initialization or in the stage argument, but got `None`.\n'
        assert gpus is not None, f'\n\nYou must pass `gpus` during initialization or in the stage argument, but got `None`.\n'
        
        return datasets_limits, max_epochs, min_epochs, precision, gpus, callbacks
    
    def _print_heading(self, heading: str = None, idx: int = 0, heading_len: int = None, length: int = 100) -> None:
        if heading_len:
            print(colored('\n[' + '=' * (heading_len - 2) + ']\n', 'blue', attrs = ['blink', 'bold']))
        else:
            bar_len = (length - len(heading)) // 2
            TITLE = f'[{"=" * bar_len} {heading} {"=" * bar_len}{"=" if (bar_len * 2) != length else ""}]'
            if idx == 0:
                print(colored('\n[' + '=' * (len(TITLE) - 2) + ']', 'blue', attrs = ['blink', 'bold']))
                print(colored(TITLE, 'blue', attrs = ['blink', 'bold']))
                print(colored('[' + '=' * (len(TITLE) - 2) + ']', 'blue', attrs = ['blink', 'bold']))
            else:
                print(colored(f'\n{TITLE}\n', 'blue', attrs = ['bold']))
            
            return len(TITLE)
    
    def _save_buffer(self, last_stage: int) -> None:
        _trainer_module = self.trainer_module
        _buffer = self.buffer
        _datamodule = self.datamodule
        _callbacks = self.callbacks
        _models = self.models
        _stages = self.stages

        buffer = vars(self)
        buffer.pop('trainer_module')
        for key, value in buffer['buffer'].items():
            buffer[key] = value
        
        buffer.pop('buffer')
        buffer.pop('datamodule')
        buffer.pop('callbacks')
        buffer = deepcopy(buffer)
        buffer['last_stage'] = last_stage

        for model in buffer['models']:
            if 'model_class' in model:
                del model['model_class']
        for stage in buffer['stages'].values():
            if 'callbacks' in stage:
                del stage['callbacks']

        torch.save(buffer, self.path, pickle_module = dill)

        self.trainer_module = _trainer_module
        self.buffer = _buffer
        self.datamodule = _datamodule
        self.callbacks = _callbacks
        self.models = _models
        self.stages = _stages

    def _initiate_stage1(self) -> None:
        self._read_buffer()

        if self.buffer['last_stage'] < 1:
            datasets_limits, max_epochs, min_epochs, precision, gpus, callbacks = self._get_stage_variables('stage1')
            assert len(datasets_limits) == 3, f'\n\nYou must provide 3 dataset limits.\n'

            monitor = self.evaluation_metric['monitor']
            assert 'test' in monitor, f'\n\nStage evaluation must be done for metric obtained from testing dataloader, but got "{monitor}".\n'

            self.stage1_results = {}
            self.best_stage1_model = None
            self.best_stage1_score = 0.0

            for model in self.models:
                nn_model = model['model_class'](**model['init'])
                model_name = model['model_class_name']

                self._print_heading(f'Initializing Stage 1 for {model_name}')

                checkpoint_callback = ArtifactModelCheckpoint(
                    project_name = self.project_name,
                    model = nn_model,
                    stage = 1,

                    **self.checkpoint,
                    save_top_k = 1,

                    model_name = model_name,
                    version = 1,
                    description = model['description'],

                    wandb_logging = self.wandb_logging
                )

                self._print_heading('Performing Dev Run Test', idx = 1)
                dev_trainer = Trainer(
                    gpus = gpus,
                    logger = False,
                    fast_dev_run = True,
                    weights_summary = None,
                    precision = precision,
                )
                dev_trainer.fit(self.trainer_module(nn_model), self.datamodule)
                
                self._print_heading('Performing Overfitting Test', idx = 1)
                overfit_trainer = Trainer(
                    gpus = gpus,
                    max_epochs = self.overfit_epochs,
                    overfit_batches = self.overfit_batches,
                    logger = False,
                    precision = precision,
                    checkpoint_callback = False,
                    terminate_on_nan = True,
                    weights_summary = None,
                )
                overfit_trainer.fit(self.trainer_module(nn_model), self.datamodule)

                self._print_heading('Starting Training', idx = 1)
                wandb_logger = CustomWandbLogger(
                    project_name = self.project_name,
                    model_name = model_name,
                    model_stage = 1,
                    model_version = 1
                ) if self.wandb_logging else None
                
                if wandb_logger:
                    wandb_logger.set_model(nn_model)
                    
                lightning_trainer = Trainer(
                    gpus = gpus,
                    max_epochs = max_epochs,
                    min_epochs = min_epochs,
                    weights_summary = None,
                    precision = precision,

                    callbacks = [checkpoint_callback, *callbacks],
                    logger = wandb_logger,
                    terminate_on_nan = True,

                    limit_train_batches = datasets_limits[0],
                    limit_val_batches = datasets_limits[1],
                    limit_test_batches = datasets_limits[2],

                    **self.kwargs,
                )
                
                lightning_module = self.trainer_module(nn_model)
                lightning_trainer.fit(lightning_module, self.datamodule)

                reloaded_model = self.trainer_module.load_from_checkpoint(checkpoint_callback.best_model_path, model = nn_model)
                self.stage1_results[model_name] = lightning_trainer.test(reloaded_model, self.datamodule)

                if self.wandb_logging:
                    wandb_logger.experiment.finish()
            
                del dev_trainer
                del overfit_trainer
                del lightning_module
                del lightning_trainer
                del wandb_logger
                del reloaded_model

            self._finalize_stage1()
        else:
            heading_len = self._print_heading('Stage 1 Completed')
            print()
            for model_name, result in self.buffer['stage1_results'].items():
                print(colored(f'{model_name}  \t-> {result[0]}', 'green', attrs = ['bold']))
                
            print(colored(f'\nBest model \t-> {self.buffer["best_stage1_model"]}', 'green', attrs = ['bold']))
            self._print_heading(heading_len = heading_len)

    def _finalize_stage1(self):
        heading_len = self._print_heading('Stage 1 Results')
        print()
        for model_name, result in self.stage1_results.items():
            print(colored(f'{model_name}  \t-> {result[0]}', 'green', attrs = ['bold']))
            if self.evaluation_metric['mode'] == 'max':
                is_best = self.best_stage1_score < result[0][self.evaluation_metric['monitor']]
            else:
                is_best = self.best_stage1_score > result[0][self.evaluation_metric['monitor']]
            
            if is_best:
                self.best_stage1_score = result[0][self.evaluation_metric['monitor']]
                self.best_stage1_model = model_name
        
        print(colored(f'\nBest model \t-> {self.best_stage1_model}', 'green', attrs = ['bold']))
        self._print_heading(heading_len = heading_len)
        
        self._save_buffer(last_stage = 1)
    
    def _initiate_stage2(self):
        self._read_buffer()

        assert self.buffer['last_stage'] >= 1, f'\n\nFor stage 2 your model must have completed stage 1.\n'

        if self.buffer['last_stage'] < 2 or not self.buffer['best_stage2_config']:
            assert self.wandb_logging, f'\n\n`wandb_logging` must be True for performing stage 2.\n'

            datasets_limits, max_epochs, min_epochs, precision, gpus, callbacks = self._get_stage_variables('stage2')
            assert len(datasets_limits) == 3, f'\n\nYou must provide 3 dataset limits.\n'

            for model in self.models:
                if self.buffer['best_stage1_model'] == model['model_class_name']:
                    break

            self._print_heading(f'Initializing Stage 2 for {model["model_class_name"]}')
            
            assert 'hyperparameters' in model, f'\n\nYou must provide `hyperparameters` config values in your model key.\n'

            sweep_method = model['hyperparameters'].pop('method')
            init_params = deepcopy(model['init'])
            for key in model['hyperparameters']:
                init_params.pop(key)
            
            self.stage2_version = -1

            self.best_stage2_config = None
            self.best_stage2_score = 0.0
            self.best_stage2_model_path = None
            self.best_stage2_version = None
            self.stage2_results = []

            sweep_config = {
                'name': 'hyperparameter-optimization',
                'method': sweep_method,
                'metric': {'name': f"stage(2)/{self.evaluation_metric['monitor'].replace('test_', 'testing/')}", 'goal': self.evaluation_metric['mode'] + 'imize'}
            }
            parameters_dict = {}
            for key, value in model['hyperparameters'].items():
                parameters_dict[key] = {'values': value}

            sweep_config['parameters'] = parameters_dict

            sweep_id = wandb.sweep(sweep_config, project = self.project_name)

            def _sweep_train_function(*args, **kwargs):
                self.stage2_version += 1

                model_name = model['model_class_name']
                wandb_logger = CustomWandbLogger(
                    project_name = self.project_name,
                    model_name = model_name,
                    model_stage = 2,
                    model_version = self.stage2_version,
                    sweep = True,
                )
                wandb_logger.experiment

                print(colored(f'\n[================== Starting Config Training for {model_name} ==================]\n', 'blue', attrs = ['bold']))
                hyperparam_config = {key: wandb.config[key] for key in model['hyperparameters'].keys()}
                for param, value in hyperparam_config.items():
                    print(colored(f'{param}: {value}', 'green', attrs = ['bold']))
                print()
                
                nn_model = model['model_class'](**init_params, **hyperparam_config)

                if wandb_logger:
                    wandb_logger.set_model(nn_model)
                
                checkpoint_callback = ArtifactModelCheckpoint(
                    project_name = self.project_name,
                    model = nn_model,
                    stage = 2,

                    **self.checkpoint,
                    save_top_k = 1,

                    model_name = model_name,
                    version = self.stage2_version,
                    description = model['description'],

                    wandb_logging = self.wandb_logging
                )

                lightning_trainer = Trainer(
                    gpus = gpus,
                    max_epochs = max_epochs,
                    min_epochs = min_epochs,
                    weights_summary = None,
                    precision = precision,

                    callbacks = [checkpoint_callback, *callbacks],
                    logger = wandb_logger,
                    terminate_on_nan = True,

                    limit_train_batches = datasets_limits[0],
                    limit_val_batches = datasets_limits[1],
                    limit_test_batches = datasets_limits[2],
                    **self.kwargs,
                )
                
                lightning_module = self.trainer_module(nn_model)
                lightning_trainer.fit(lightning_module, self.datamodule)

                reloaded_model = self.trainer_module.load_from_checkpoint(checkpoint_callback.best_model_path, model = nn_model)
                self.stage2_results.append((hyperparam_config, lightning_trainer.test(reloaded_model, self.datamodule), checkpoint_callback.best_model_path))

                if self.wandb_logging:
                    wandb_logger.experiment.finish()
                
                del lightning_module
                del lightning_trainer
                del wandb_logger
                del reloaded_model
            
            wandb.agent(sweep_id, _sweep_train_function)
            wandb.finish()
            
            self._finalize_stage2()
        else:
            heading_len = self._print_heading('Stage 2 Completed')
            print()
            for config, result, _ in self.buffer['stage2_results']:
                print(colored(f'{config}  \t-> {result[0]}', 'green', attrs = ['bold']))
                
            print(colored(f'\nBest config \t-> {self.buffer["best_stage2_config"]}', 'green', attrs = ['bold']))
            self._print_heading(heading_len = heading_len)

    def _finalize_stage2(self):
        heading_len = self._print_heading('Stage 2 Results')
        print()
        for version, (config, result, ckpt_path) in enumerate(self.stage2_results):
            print(colored(f'{config}  \t-> {result[0]}', 'green', attrs = ['bold']))
            
            if self.evaluation_metric['mode'] == 'max':
                is_best = self.best_stage2_score < result[0][self.evaluation_metric['monitor']]
            else:
                is_best = self.best_stage2_score > result[0][self.evaluation_metric['monitor']]
            
            if is_best:
                self.best_stage2_score = result[0][self.evaluation_metric['monitor']]
                self.best_stage2_config = config
                self.best_stage2_model_path = ckpt_path
                self.best_stage2_version = version
        
        print(colored(f'\nBest config \t-> {self.best_stage2_config}', 'green', attrs = ['bold']))
        self._print_heading(heading_len = heading_len)
        
        self._save_buffer(last_stage = 2)
    
    def _initiate_stage3(self) -> None:
        self._read_buffer()

        assert self.buffer['last_stage'] >= 2, f'\n\nFor stage 3 your model must have completed stage 2 and stage 1.\n'

        if self.buffer['last_stage'] < 3:
            datasets_limits, max_epochs, min_epochs, precision, gpus, callbacks = self._get_stage_variables('stage3')
            assert len(datasets_limits) == 3, f'\n\nYou must provide 3 dataset limits.\n'

            monitor = self.evaluation_metric['monitor']
            assert 'test' in monitor, f'\n\nStage evaluation must be done for metric obtained from testing dataloader, but got "{monitor}".\n'

            self.best_stage3_test_score = 0.0
            self.best_stage3_val_score = 0.0
            self.best_stage3_path = None

            for model in self.models:
                if self.buffer['best_stage1_model'] == model['model_class_name']:
                    break
            
            init_params = deepcopy(model['init'])
            for key in self.buffer['best_stage2_config']:
                init_params.pop(key)
            
            nn_model = model['model_class'](**init_params, **self.buffer['best_stage2_config'])
            model_name = model['model_class_name']

            self._print_heading(f'Initializing Stage 3 for {model_name}')

            checkpoint_callback = ArtifactModelCheckpoint(
                project_name = self.project_name,
                model = nn_model,
                stage = 3,
                best_stage2_version = self.buffer['best_stage2_version'],

                **self.checkpoint,
                save_top_k = 1,

                model_name = model_name,
                version = 1,
                description = model['description'],

                wandb_logging = self.wandb_logging
            )

            wandb_logger = CustomWandbLogger(
                project_name = self.project_name,
                model_name = model_name,
                model_stage = 3,
                model_version = 1
            ) if self.wandb_logging else None
            
            if wandb_logger:
                wandb_logger.set_model(nn_model)

            self._print_heading('Starting Training', idx = 1)
                
            lightning_trainer = Trainer(
                gpus = gpus,
                max_epochs = max_epochs,
                min_epochs = min_epochs,
                weights_summary = None,
                precision = precision,

                callbacks = [checkpoint_callback, *callbacks],
                logger = wandb_logger,
                terminate_on_nan = True,

                limit_train_batches = datasets_limits[0],
                limit_val_batches = datasets_limits[1],
                limit_test_batches = datasets_limits[2],
                
                **self.kwargs,
            )

            lightning_module = self.trainer_module.load_from_checkpoint(self.buffer['best_stage2_model_path'], model = nn_model)
            lightning_trainer.fit(lightning_module, self.datamodule)

            self.sample_input_shape = lightning_module.sample_train_batch[0].shape

            self.best_stage3_path = checkpoint_callback.best_model_path
            self.best_stage3_val_score = {checkpoint_callback.monitor: checkpoint_callback.best_model_score.item()}

            reloaded_model = self.trainer_module.load_from_checkpoint(checkpoint_callback.best_model_path, model = nn_model)
            self.best_stage3_test_score = {self.evaluation_metric['monitor']: lightning_trainer.test(reloaded_model, self.datamodule)[0][self.evaluation_metric['monitor']]}

            if self.wandb_logging:
                wandb_logger.experiment.finish()
        
            del lightning_module
            del lightning_trainer
            del wandb_logger
            del reloaded_model

            self._finalize_stage3()
        else:
            heading_len = self._print_heading('Stage 3 Completed')
            print()
            print(colored(f'Best Validation Score  -> {self.buffer["best_stage3_val_score"]}', 'green', attrs = ['bold']))
            print(colored(f'Best Testing Score     -> {self.buffer["best_stage3_test_score"]}', 'green', attrs = ['bold']))

            self._print_heading(heading_len = heading_len)

    def _finalize_stage3(self):
        heading_len = self._print_heading('Stage 3 Results')
        print()
        print(colored(f'Best Validation Score  -> {self.best_stage3_val_score}', 'green', attrs = ['bold']))
        print(colored(f'Best Testing Score     -> {self.best_stage3_test_score}', 'green', attrs = ['bold']))
        
        self._print_heading(heading_len = heading_len)
        
        self._save_buffer(last_stage = 3)
    
    def _start(self):
        '''
        Starts the autotraining with the provided configurations

        Returns -> None
        '''
        
        if args.stage == 1:
            self._initiate_stage1()
        
        elif args.stage == 2:
            self._initiate_stage2()
        
        elif args.stage == 3:
            self._initiate_stage3()
        
        elif args.deploy:
            self._initiate_stage1()
            self._initiate_stage2()
            self._initiate_stage3()
            self._deploy(args.opset)
    
        # autotrainer._initiate_stage1()
        # autotrainer._initiate_stage2()
        # autotrainer._initiate_stage3()
    
    def _deploy(self, opset_version: int = 10):
        '''
        Deploys the model into production with ONNX runtime

        Returns -> None
        '''

        self._read_buffer()

        assert self.buffer['last_stage'] == 3, f'\n\nFor deployment your model must be in stage 3, but found stage {self.buffer["last_stage"]}.\n'

        for model in self.models:
            if self.buffer['best_stage1_model'] == model['model_class_name']:
                break
        
        init_params = deepcopy(model['init'])
        for key in self.buffer['best_stage2_config']:
            init_params.pop(key)
        
        nn_model = model['model_class'](**init_params, **self.buffer['best_stage2_config'])

        reloaded_model = self.trainer_module.load_from_checkpoint(self.buffer['best_stage3_path'], model = nn_model)
        reloaded_nn_model = reloaded_model.model
        reloaded_nn_model.eval()

        torch_path = os.path.join(self.project_name, f"{self.buffer['best_stage1_model']}.pth")
        torch.save(reloaded_nn_model.state_dict(), torch_path)

        sample_inputs = torch.ones(self.buffer['sample_input_shape'])
        onnx_path = os.path.join(self.project_name, f"{self.buffer['best_stage1_model']}.onnx")
        
        torch.onnx.export(
            reloaded_nn_model,
            sample_inputs,
            onnx_path,
            export_params = True,
            opset_version = opset_version,
            do_constant_folding = True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes = {
                'input': {0 : 'batch_size'},
                'output': {0 : 'batch_size'}
            },
        )

        wandb.finish()
        wandb.init(project = self.project_name, name = 'Deploy', resume = True, id = str(f'{self.project_name}-{self.buffer["best_stage1_model"]}-deploy'))
        wandb.save(torch_path)
        wandb.save(onnx_path)