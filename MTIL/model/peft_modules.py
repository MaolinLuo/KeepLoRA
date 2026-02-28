import math
from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeepLoRA(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 8, lora_alpha: int = 8, use_rslora: bool = False, dtype=None):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = r
        self.scaling = lora_alpha / math.sqrt(r) if use_rslora else lora_alpha / r
        self.dtype = dtype

        self.reset_parameters()

    def reset_parameters(self, device=None):
        self.lora_A = torch.zeros(self.in_dim, self.rank, dtype=self.dtype, device=device)
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_dim, dtype=self.dtype, device=device))

        # Feature accumulator for SVD
        self.register_buffer('accumulated_feature_matrix', None)
        self.register_buffer('num_accumulated_samples', torch.tensor(0, dtype=torch.long, device=device))

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        module.lora_A = module.lora_A.to(*args, **kwargs)
        return module

    def release_buffer(self):
        """
        Releases the buffers used for accumulating feature statistics.
        """
        if self.accumulated_feature_matrix is not None:
            del self.accumulated_feature_matrix
        if self.num_accumulated_samples is not None:
            del self.num_accumulated_samples

    def accumulate_features(self, x: torch.Tensor):
        x = x.detach().clone()
        if x.dim() == 3:
            _emb_dim = x.size(-1)
            x = x.reshape(-1, _emb_dim)
            num_samples_in_batch = x.size(0)
        elif x.dim() == 2:
            _emb_dim = x.size(-1)
            num_samples_in_batch = x.size(0)
        else:
            raise ValueError(f"Input tensor must have 2 or 3 dimensions, but got shape {x.shape}")
        if _emb_dim != self.in_dim:
            raise ValueError(f"Input feature dimension {_emb_dim} does not match module in_dim {self.in_dim}")

        if self.accumulated_feature_matrix is None:
            self.accumulated_feature_matrix = torch.zeros(x.size(1), x.size(0), dtype=self.dtype, device=x.device)

        current_batch_sum_outer_prod = x.T

        self.accumulated_feature_matrix = (self.num_accumulated_samples / (self.num_accumulated_samples + num_samples_in_batch)) * self.accumulated_feature_matrix + (1 / (self.num_accumulated_samples + num_samples_in_batch)) * current_batch_sum_outer_prod
        self.accumulated_feature_matrix = self.accumulated_feature_matrix.to(dtype=self.dtype)
        self.num_accumulated_samples += num_samples_in_batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x

    def get_delta_weight(self) -> torch.Tensor:
        lora_A_param = self.lora_A.to(self.dtype)
        lora_B_param = self.lora_B.to(self.dtype)

        delta_W = lora_A_param @ lora_B_param

        return self.scaling * delta_W.T