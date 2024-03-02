import warnings

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> # xdoctest: +SKIP("undefined vars")
        >>> scheduler = PolynomialLR(self.opt, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False
    ):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = (
            (1.0 - self.last_epoch / self.total_iters)
            / (1.0 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                base_lr
                * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters)
                ** self.power
            )
            for base_lr in self.base_lrs
        ]
