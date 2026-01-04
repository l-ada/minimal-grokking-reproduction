from typing import Optional, Callable
import math
import torch


class AdamW_MSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            maximize: bool = False,
            rho = 0.05,
            ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rho < 0.0:
            raise ValueError('rho should be non-negative')
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            rho=rho,
        )
        super(AdamW_MSAM, self).__init__(params, defaults)
        for group in self.param_groups:
            group["norm_factor"] = [0.0]

    @torch.no_grad()
    def move_up(self):
        for group in self.param_groups:
            scale = group["norm_factor"][0]
            for p in group['params']:
                if "exp_avg" in self.state[p]:
                    p.sub_(self.state[p]["exp_avg"], alpha=scale)

    @torch.no_grad()
    def move_back(self):
        for group in self.param_groups:
            scale = group["norm_factor"][0]
            for p in group['params']:
                if "exp_avg" in self.state[p]:
                    p.add_(self.state[p]["exp_avg"], alpha=scale)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []

            # 1. Gather Loop
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                params_with_grad.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                state['step'] += 1
                state_steps.append(state['step'])

            # 2. Update Loop (Descent)
            beta1, beta2 = group['betas']
            for i, p in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]

                # Update momentum and variance
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Denominator and step size
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # Weight Update & Weight Decay
                p.addcdiv_(exp_avg, denom, value=-step_size)
                p.mul_(1 - group['lr'] * group['weight_decay'])

            # 3. Calculate Norm for next Jump (Ascent)
            if len(exp_avgs) > 0:
                ascent_norm = torch.norm(
                    torch.stack([buf.norm(p=2) for buf in exp_avgs]),
                    p=2
                )
                group["norm_factor"][0] = group['rho'] / (ascent_norm + 1e-12)

        return loss