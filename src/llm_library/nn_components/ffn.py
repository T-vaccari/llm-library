import torch
from torch import nn


from llm_library.nn_components.linear import Linear


def silu(x):
   return x * torch.sigmoid(x)

"""
SwiGLU feed-forward network, composed of a SiLU activation
function and a GLU(Gated Linear Unit).

SiLU activation (Swish) is similar to the relu except that it's more smooth at zero
SiLU(x) = x / (1 + e^-x)

The idea behind the Gated Linear Unit is to maintain a non linear path that acts as a gate
for the informational path.


set 𝑑_ff to approximately 8/3 × 𝑑model, while ensuring that the
dimensionality of the inner feed-forward layer is a multiple of 64 to 
make good use of your hardware.

This comes from trying to keep the FFN roughly at the same original FLOPs costs.

# FFN(x) = SwiGLU(x, W1, W2, W3)  = W2 @ (SiLU(W1 @ x) * W3 @ x)

"""

# Hand - Derived Version without abstractions
# class FeedForwardNetwork(nn.Module):
#    def __init__(self, d_model, d_ff = None):
#       super().__init__()
#       # I need to find the nearest multiple of 64 of ceil(8/3 * d_model)
#       if d_ff is None:
#          d_ff = 64 * round(((8 / 3 )* d_model) / 64)

#       self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
#       nn.init.trunc_normal_(self.w1, mean = 0, std = (2 / (d_model + d_ff))** 0.5, a = - 3 * (2 / (d_model + d_ff))** 0.5, b =  3* (2 / (d_model + d_ff))** 0.5)

#       self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
#       nn.init.trunc_normal_(self.w2, mean = 0, std = (2 / (d_model + d_ff))** 0.5, a = - 3 * (2 / (d_model + d_ff))** 0.5, b =  3* (2 / (d_model + d_ff))** 0.5)
      
#       self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
#       nn.init.trunc_normal_(self.w3, mean = 0, std = (2 / (d_model + d_ff))** 0.5, a = - 3 * (2 / (d_model + d_ff))** 0.5, b =  3* (2 / (d_model + d_ff))** 0.5)

#    def forward(self, x):

#       #First Operation : h = W1 @ x
#       # x.shape (d_model, T, B)
   
#       h = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff") # (B, T, d_ff)

#       #Second op : SiLU(h) = h * sigmoid(h)
#       act =  h * torch.sigmoid(h)

#       # Third Operation : z = W3 @ x
      
#       z = einsum(x, self.w3 , "... d_model , d_ff d_model -> ... d_ff") #(B, T, d_ff)

#       # Fourth Op : elementwise k = act * z
      
#       k = act * z

#       # Fifth Operation result = w2 @ k
      
#       result = einsum(k, self.w2 , "... d_ff, d_model d_ff -> ... d_model")

#       return result
      
#---------------------

# Version using the already implemented Linear Layer


class FeedForwardNetwork(nn.Module):
   def __init__(self, d_model, d_ff=None, device=None, dtype=None):
      super().__init__()

      if d_ff is None:
         d_ff = 64 * round(((8 / 3) * d_model) / 64)

      self.d_model = d_model
      self.d_ff = d_ff

      self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)

      self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

      self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

   def forward(self, x):

      h = self.w1(x)

      act = silu(h)

      z = self.w3(x)

      k = act * z

      return self.w2(k)
      
      
