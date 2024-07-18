import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNormWot(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.bias = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :] + self.bias[None, None, :]


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlockWot(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNormWot(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNormWot(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True), nn.GELU(approximate="tanh"), nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.use_checkpoint = use_checkpoint

    def forward(self, x, rotary_cos_sin, seqlens=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, rotary_cos_sin, seqlens)
        else:
            return self._forward(x, rotary_cos_sin, seqlens)

    def _forward(self, x, rotary_cos_sin, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        # attention operation
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, 'b s three h d -> three b h s d')
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        x = scaled_dot_product_attention(q, k, v)
        x = rearrange(x, 'b h s d-> b s (h d)', b=batch_size)

        x = x_skip + F.dropout(self.attn_out(x), p=self.dropout, training=self.training)

        # mlp operation
        x = torch.add(x, F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training))
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayerWot(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNormWot(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class RADD(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        vocab_size = config.tokens + 1

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList(
            [
                DDiTBlockWot(
                    config.model.hidden_size, config.model.n_heads, dropout=config.model.dropout, use_checkpoint=config.model.use_checkpoint
                )
                for _ in range(config.model.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayerWot(config.model.hidden_size, vocab_size)
        if config.model.dtype == 'float32':
            self.dtype = torch.float32
        elif config.model.dtype == 'float16':
            self.dtype = torch.float16
        elif config.model.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16

    def forward(self, indices):

        x = self.vocab_embed(indices)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)

            x = self.output_layer(x)

            x[:, :, :-1] = x[:, :, :-1].log_softmax(dim=-1)

        return x

    def logits(self, indices):

        x = self.vocab_embed(indices)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)

            x = self.output_layer(x)

        return x
