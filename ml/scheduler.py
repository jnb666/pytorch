import logging as log
import weakref
from functools import wraps

import torch
from torch.optim import Optimizer


class StepLRandWeightDecay(object):
    """Decays the learning rate and weight decay of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('initial_weight_decay', group['weight_decay'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
                if 'initial_weight_decay' not in group:
                    raise KeyError("param 'initial_weight_decay' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.base_weight_decays = [group['initial_weight_decay'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.verbose = verbose

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a: class: `dict`."""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state."""
        self.__dict__.update(state_dict)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def get_weight_decay(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['weight_decay'] for group in self.optimizer.param_groups]
        return [group['weight_decay'] * self.gamma
                for group in self.optimizer.param_groups]

    def step(self):
        """Step to next epoch and update learning rate if needed"""
        self._step_count += 1
        self.last_epoch += 1
        lr_values = self.get_lr()
        wd_values = self.get_weight_decay()

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr_values[i]
            param_group['weight_decay'] = wd_values[i]
            if self._step_count > 1 and i == 0 and self.last_epoch % self.step_size == 0:
                log.info(f'Adjusting learning rate to  {lr_values[0]:.5} and weight decay to {wd_values[0]:.5}.')
