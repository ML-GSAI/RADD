import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from noise_lib import add_noise_t, add_noise_lambda, add_noise_k



def get_loss_fn(noise, token_dim, train, sampling_eps=1e-3, loss_type='lambda_DCE',order = torch.arange(1024)):
    def t_DSE_loss(model, batch, cond = None):
        # sample t and add noise
        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        sigma, dsigma = noise(t)
        sigma, dsigma = sigma[:,None], dsigma[:,None]
        perturbed_batch = add_noise_t(batch, sigma, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        # compute c_theta and scaling factor
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),torch.exp(sigma) - 1 )
        # compute score (reuse log_condition to save memory)
        log_condition -=esigm1.log()[...,None]

        scaling_factor = 1 / esigm1.expand_as(perturbed_batch)
        
        # compute three terms
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        # add negative term
        loss[masked_index] = - torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss/= esigm1
        # add pos term
        loss[masked_index] += log_condition[masked_index][:, :-1].exp().sum(dim=-1)

        # add const term 
        loss[masked_index] += scaling_factor[masked_index] * (scaling_factor[masked_index].log() - 1)
        return (dsigma * loss).sum(dim=-1)

    def t_DCE_loss(model, batch, cond = None):
        # sample t and add noise
        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        sigma, dsigma = noise(t)
        sigma, dsigma = sigma[:,None], dsigma[:,None]
        perturbed_batch = add_noise_t(batch, sigma, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        # compute c_theta and scaling factor
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),torch.exp(sigma) - 1 )
        # compute score 
        log_condition -=esigm1.log()[...,None]

        # compute DCE loss
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = - torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss/= esigm1
        return (dsigma * loss).sum(dim=-1)

    def lambda_DCE_loss(model, batch, cond = None):
        # sample lambda and add noise
        Lambda = torch.rand(batch.shape[0], device=batch.device)
        perturbed_batch = add_noise_lambda(batch, Lambda, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]
        
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss = - loss.sum(dim = -1).to(torch.float64)/Lambda.to(torch.float64)
        return loss

    def k_DCE_loss(model, batch, cond = None): # any-order ar loss
        # sample k and add noise
        k = torch.randint(1, batch.shape[1] + 1 ,(batch.shape[0],),device=batch.device)
        perturbed_batch = add_noise_k(batch, k, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss = - loss.sum(dim = -1)/k * batch.shape[1]
        return loss.to(torch.float32)

    if loss_type =='ar_forward':
        order = torch.arange(0,1024)
    elif loss_type =='ar_backward':
        order = torch.arange(1023,-1,-1)
    else:
        order = torch.arange(1024)

    def ar_loss(model, batch):
        nonlocal order
        if loss_type == 'ar_random':
            order = torch.randperm(1024)
        if train:
            model.train()
        else:
            model.eval()
        loss = 0
        for i in range(batch.shape[1]):
            masked_batch = batch.clone()
            masked_batch[:,order[i:]] = token_dim - 1
            p_log_condition_i = model(masked_batch)[:,order[i]]
            loss += - p_log_condition_i[torch.arange(batch.shape[0]),batch[:,order[i]]].to(torch.float32)
        return loss
    
    if loss_type == 'ar_forward' or loss_type == 'ar_backward' or loss_type == 'ar_random': # ar loss for a fix order
        return ar_loss
    elif loss_type =='lambda_DCE':
        return lambda_DCE_loss
    elif loss_type =='t_DCE':
        return t_DCE_loss
    elif loss_type =='t_DSE':
        return t_DSE_loss
    elif loss_type =='k_DCE':  # any-order ar loss
        return k_DCE_loss
    else:
        raise NotImplementedError(f'Loss type {loss_type} not supported yet!')


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, token_dim,  train, optimize_fn, accum, loss_type):
    loss_fn = get_loss_fn(noise, token_dim, train, loss_type = loss_type)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn