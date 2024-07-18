import abc
import torch
import torch.nn.functional as F
from catsample import sample_with_strategy, sample_categorical


import abc


class Sampler(abc.ABC):
    def __init__(self, model, batch_dims, token_dim, strategy, strategy_para=None, device=torch.device('cuda')):
        super().__init__()
        self.model = model
        self.batch_dims = batch_dims
        self.device = device
        self.strategy = strategy
        self.strategy_para = strategy_para
        self.token_dim = token_dim

    @abc.abstractmethod
    def sample(self, steps):
        raise NotImplementedError


class DiffusionSampler(Sampler):
    def __init__(self, method, model, noise, batch_dims, token_dim, strategy, strategy_para=None, eps=1e-5, device=torch.device('cuda')):
        super().__init__(model, batch_dims, token_dim, strategy, strategy_para, device)
        self.noise = noise
        self.eps = eps
        self.method = method
        self.update_cnt = 0

    @torch.no_grad()
    def sample(self, steps, proj_fun=lambda x: x):
        if self.strategy == 'direct':
            return self.direct_sample(steps, proj_fun)
        else:
            return self.strateged_sample(steps, proj_fun)

    @torch.no_grad()
    def strateged_sample(self, steps, proj_fun=lambda x: x):
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)

        x = proj_fun(x)
        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        changed = torch.ones(self.batch_dims[0], dtype=torch.bool)
        logits = torch.zeros(*self.batch_dims, self.token_dim, dtype=self.model.dtype).to(self.device)

        for i in range(steps):
            t = timesteps[i]
            update_rate = self.get_update_rate(t, steps)
            if changed.any():
                logits[changed] = self.model.logits(x[changed])
                self.update_cnt += changed.sum().item()
            mask = x == self.token_dim - 1
            update_indices = (mask & (torch.rand(*self.batch_dims).to(self.device) < update_rate)) if i < steps - 1 else mask
            update_logits = logits[update_indices]
            update_x = sample_with_strategy(update_logits, self.strategy, self.strategy_para)
            x_old = x.clone()
            x[update_indices] = update_x
            changed = (x != x_old).any(dim=-1)
        return x

    @torch.no_grad()
    def direct_sample(self, steps, proj_fun=lambda x: x):
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)

        x = proj_fun(x)
        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        changed = torch.ones(self.batch_dims[0], dtype=torch.bool)
        p_condition = torch.zeros(*self.batch_dims, self.token_dim, dtype=torch.float16).to(self.device)
        for i in range(steps):
            t = timesteps[i]
            update_rate = self.get_update_rate(t, steps) if i < steps - 1 else 1 + 1e-3
            if changed.any():
                mask = x == self.token_dim - 1
                p_condition[changed] = self.model(x[changed]).exp()
                p_condition_mask = p_condition[mask]
            probs_mask = p_condition_mask * update_rate
            probs_mask[..., -1] = 1 - update_rate
            update_x_mask = sample_categorical(probs_mask.to(torch.float32))
            x_old = x.clone()
            x[mask] = update_x_mask
            changed = (x != x_old).any(dim=-1)
            self.update_cnt += changed.sum().item()
        return x

    def get_update_rate(self, t, steps):
        dt = (1 - self.eps) / steps
        curr_sigma, next_sigma = self.noise(t)[0], self.noise(t - dt)[0]
        d_curr_sigma = self.noise(t)[1]
        if self.method == 'tweedie':
            update_rate = ((-next_sigma).exp() - (-curr_sigma).exp()) / (1 - (-curr_sigma).exp())
        elif self.method == 'euler':
            update_rate = dt * d_curr_sigma * (-curr_sigma).exp() / (1 - (-curr_sigma).exp())
        return update_rate


class OrderedSampler(Sampler):
    def __init__(self, model, batch_dims, token_dim, strategy, strategy_para=None, order=None, device=torch.device('cuda')):
        super().__init__(model, batch_dims, token_dim, strategy, strategy_para, device)
        self.order = order

    @torch.no_grad()
    def sample(self, steps, proj_fun=lambda x: x):
        order = torch.randperm(1024) if self.order is None else self.order
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)
        x = proj_fun(x)

        for i in range(steps):
            logits = self.model.logits(x)
            update_logits = logits[:, order[i], :-1]
            x[:, order[i]] = sample_with_strategy(update_logits, self.strategy, self.strategy_para)
        return x
