import torch
from torch import nn
import einops
from einops import einsum

class Linear(nn.Module):
   def __init__(self, in_features, out_features, device=None, dtype=None):
      super().__init__()

      self.weight = nn.Parameter(
         torch.empty(
            out_features,
            in_features,
            device=device,
            dtype=dtype
         )
      )

      std = (2 / (in_features + out_features)) ** 0.5

      nn.init.trunc_normal_(
         self.weight,
         mean=0,
         std=std,
         a=-3 * std,
         b=3 * std
      )
      
   def forward(self, x):

      return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features ")
         

      