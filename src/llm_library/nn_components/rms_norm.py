import torch
from torch import nn
import einops
from einops import einsum, reduce


class RMSNorm(nn.Module):
   def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, impl = "default"):
      super().__init__()
      self.eps = eps
      self.gain = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
      self.impl = impl

   def forward(self, x):
      in_dtype = x.dtype
      x = x.to(torch.float32)
   
      # pytorch version
      if self.impl == "default":
         rms = (reduce(x**2, "... c -> ... 1","mean")+ self.eps) ** 0.5
         x_normalized = x / (rms)
         result = einsum(x_normalized, self.gain, "... features, features -> ... features")
         
      else:
         rms = ((x ** 2).mean(dim= -1, keepdim = True)+ self.eps) ** 0.5 # (B, T, 1)
         x_normalized = x / (rms ) # (B, T, C) / (B,T,1) -> Broadcast
         result = (x_normalized * self.gain) # B,T,C * C -> Broadcast

      
      return result.to(in_dtype)