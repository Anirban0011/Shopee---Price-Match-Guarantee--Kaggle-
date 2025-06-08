from torch.optim.lr_scheduler import LRScheduler


class WarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        plateau_lr_bb,
        plateau_lr_neck,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_epochs = warmup_epochs
        self.plateau_lr_backbone = plateau_lr_bb
        self.plateau_lr_neck = plateau_lr_neck
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # backbone warmup schedule
        if self.last_epoch < self.warmup_epochs:
            return [
                (self.plateau_lr_backbone) * (self.last_epoch + 1) / self.warmup_epochs
            ] + [
                (self.plateau_lr_neck) * (self.last_epoch + 1) / self.warmup_epochs
                for _ in self.base_lrs[1:]
            ]  # neck warmup schedule
        else:
            return [self.plateau_lr_backbone] + [
                self.plateau_lr_neck for _ in self.base_lrs[1:]
            ]
