import torch
from torch.optim import Optimizer


class OGDA(Optimizer):
    """Optimistic Gradient Descent Algorithm"""

    def __init__(self, params, lr=1e-3):
        self.prev_grad = []
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        # compute new gradient and load old one
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            i=-1-1
            for p in group['params']:
                i+=1

                print(self.prev_grad)
                print('param')

                if p.grad is None:
                    continue
                grad = p.grad

                if len(self.prev_grad) == 0:
                    self.prev_grad.append(grad.clone())
                    old_grad = torch.zeros_like(p)

                else:
                    old_grad = self.prev_grad[i]

                state = self.state[p]
                print('state', state)
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                state['step'] += 1

                print(p.data.shape, grad.shape, old_grad.shape)
                p.data.add_(grad, alpha=-2).add_(old_grad)

                self.prev_grad[i] = grad.clone()
                print('step for param finished')
        return loss


