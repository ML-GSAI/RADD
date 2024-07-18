import torch
import torch.nn.functional as F


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    
def direct_sampling(logits):
    probs = logits.softmax(dim=-1)
    index = sample_categorical(probs.to(torch.float32))
    return index


def top_p_sampling(logits, p=0.9):
    probs = logits.softmax(dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs.masked_fill_(indices_to_remove, 0)
    probs /= probs.sum(dim=-1).unsqueeze(-1)
    index = sample_categorical(probs.to(torch.float32))

    return index


def top_k_sampling(logits, k=400):
    top_k_values, top_k_indices = torch.topk(logits, int(k))
    top_k_probs = top_k_values.softmax(dim=-1)
    index = sample_categorical(top_k_probs.to(torch.float32))
    index = top_k_indices[torch.arange(index.size(0)), index]

    return index

def sample_with_strategy(update_logits, strategy, para = None):
    if strategy == "direct":
        return direct_sampling(update_logits)
    elif strategy == "top_p":
        return top_p_sampling(update_logits, para)
    elif strategy == "top_k":
        return top_k_sampling(update_logits, para)
    else:
        raise ValueError(f"Strategy {strategy} is not valid.")