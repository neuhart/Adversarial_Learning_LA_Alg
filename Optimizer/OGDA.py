import torch
from torch.optim import Optimizer


class OGDA(Optimizer):
    """Optimistic Gradient Descent Ascent Algorithm"""

    def __init__(self, params, lr=1e-3):
        self.prev_grad = []
        self.first = True
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        # compute new gradient and load old one
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            i=-1
            for p in group['params']:

                if p.grad is None:
                    continue
                grad = p.grad

                if self.first:
                    self.prev_grad.append(grad.clone())
                    old_grad = torch.zeros_like(p)

                else:
                    i += 1
                    old_grad = self.prev_grad[i]

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                state['step'] += 1

                p.data.add_(grad, alpha=-2*group['lr']).add_(old_grad, alpha=group['lr'])

                self.prev_grad[i] = grad.clone()
        self.first = False
        return loss


