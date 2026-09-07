import torch
from torch import nn

from llm_library.nn_components.multi_head_attention import CausalMultiHeadSelfAttention
from llm_library.nn_components.rms_norm import RMSNorm
from llm_library.nn_components.ffn import FeedForwardNetwork

class TransformerBlock(nn.Module):
   def __init__(self, d_model,  num_heads, d_ff, max_seq_len , theta = None, device = None, dtype = None ):
      super().__init__()

      self.multi_head_attention = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
      self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
      self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)
      self.ffn = FeedForwardNetwork(d_model, d_ff, device = device, dtype = dtype)

   def forward(self, x):

      #First sub-block with residual stream untouched
      y = x + self.multi_head_attention(self.rms_norm1(x))

      #Second sub-block with residual stream
      res = y + self.ffn(self.rms_norm2(y))

      return res
      
      