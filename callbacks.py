from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class UnfreezingOnPlateau(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_unfreeze(trainer, pl_module)

    def _run_unfreeze(self, trainer, pl_module):
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to UNFREEZE pretrained layers
        """
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)

        should_stop, reason = self._evalute_stopping_criteria(current)

        if should_stop:
            for name, param in pl_module.model.pretrained_block.named_parameters():
                param.requires_grad_()
                # print(f'Unfreezing {name}')
            print('\nUnfreezing pretrained layers\n')

        if reason and self.verbose:
            self._log_info(trainer, reason)