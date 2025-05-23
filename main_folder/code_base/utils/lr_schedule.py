from torch.optim.lr_scheduler import LRScheduler


class WarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        plateau_lr,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_epochs = warmup_epochs
        self.plateau_lr = plateau_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # train all modules with same lr
        if self.last_epoch < self.warmup_epochs:
            return [
                (self.plateau_lr) * (self.last_epoch + 1) / self.warmup_epochs
                for _ in self.base_lrs
            ]
        else:
            return [self.plateau_lr for _ in self.base_lrs]
