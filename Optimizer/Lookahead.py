from collections import defaultdict
#  The functionality of both dictionaries and defaultdict are almost same except for the fact that defaultdict
#  never raises a KeyError. It provides a default value for the key that does not exists.
import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):  # subclass of Optimizer class
    r"""
    Copyright (c) 2020 Michael Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.5, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        device (torch.device):
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha  # linear interpolation factor: 0 - keeps old slow weight, 1 - picks latest fast weight
        self._total_la_steps = la_steps  # steps taken at each iteration in the inner loop
        pullback_momentum = pullback_momentum.lower()  # returns lowercase string
        assert pullback_momentum in ["reset", "pullback", "none"]  # checks if correct input is given
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)  # creates an empty dict of dicts with default set to empty dict entry {}

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            # each param_group is a dictionary containing params (parameters) in form of tensors
            # and corresp. hyperparameters (eps,lr,etc)
            # one can add a new group of parameters and specify new hyperparameters
            # standard hyperparameters (e.g. lr, eps) are set to a standard value (lr=0.001) if not spec. otherwise
            for p in group['params']:
                param_state = self.state[p]  # creates an empty dict entry for p
                param_state['cached_params'] = torch.zeros_like(p.data)
                # creates a 0-tensor with same size as p.data and stores it in cached_params entry of dict entry of p
                param_state['cached_params'].copy_(p.data)  # copies values from p
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):  # inherits functions from Optimizer class
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)  # save current p
                p.data.copy_(param_state['cached_params'])  # load cached p

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])  # load backup in again
                del param_state['backup_params']  # and delete backup

    @property  # pythonic way to use getters and setters in object-oriented programming.
    def param_groups(self):
        # in this case a getter function is defined. this is to make sure that param_groups is not directly accessed
        # or modified but instead through this function
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss, required for some algorithms like BGFS
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0

            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]  # accesses dict entry of tensor p
                    """updates slow weight"""
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)
                    # param_state['cached_params'].to(device) to have all tensors on same device
                    # old slow weight is stored in cached_params entry in dict of p
                    # new slow weight= la_alpha * latest fast weight + (1-la_alpha) * old slow weight
                    param_state['cached_params'].copy_(p.data)
                    # stores new slow weight in cached_params
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss