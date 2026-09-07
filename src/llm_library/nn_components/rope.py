import torch
from torch import nn
import einops
from einops import einsum, rearrange


from llm_library.nn_components.linear import Linear

"""
To inject positional information into the model, 
we will use Rotary Position Embeddings

"""

class RotaryPositionalEmbedding(nn.Module):
   def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
      super().__init__()
      self.theta = theta
      self.d_k = d_k
      self.max_seq_len = max_seq_len
      self.device = device

      # Pre process lookup tables
      angle = lambda token_i, couple_j: token_i / (self.theta ** ((2 * couple_j - 2) / self.d_k))
      cos_vec_table = torch.stack([torch.stack([torch.cos(torch.tensor(angle(i, k))) for k in range(1, int(d_k//2) + 1) ]) for i in range(max_seq_len)])
      sin_vec_table = torch.stack([torch.stack([torch.sin(torch.tensor(angle(i, k))) for k in range(1, int(d_k//2) + 1) ]) for i in range(max_seq_len)])
      self.register_buffer("cos_vec_table", cos_vec_table, persistent=False)
      self.register_buffer("sin_vec_table", sin_vec_table, persistent=False)
      
      
   def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor :
      """
      Process
      an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
      that you should tolerate 𝑥 with an arbitrary number of batch dimensions. You should assume
      that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
      𝑥 along the sequence dimension
      """


      # I receive (..., T, dk) where T is an arbitrary sequence length and it's not fixed, the position of each token
      # is specified in the token_position tensor (..., T) so for each embedding vector of dimension C i have also 
      # the corresponding i position
      
      # I assume that x is already a query or a key vector and I have to rotate it. I have to assert that it's d_k dimension
      # it's a multiple of 2, given that I am going to rotate in couples. That means that i am going to create couples of 
      # coordinate that are going to be rotate at different speeds, like take for example x = (x1, x2, x3, x4)
      # The couple (x1, x2) is going to be rotated using a certain coefficient, and instead (x3, x4) is going to be rotated 
      # at another speed. Why this difference in speed rotation? to increase entropy.
      
      # I need to build the rotation matrix, the rotation applied to a given couple k of coordinate of a given token ith
      #  is dependent on the ith index and on the couple k. 
      
  

      # I need to find a way in which I can pack the rotation into a matrix - vector mul
      x = rearrange(x, "... (couple_index len) -> ... couple_index len", len = 2)

      # x.shape -> (..., couple_index, 2)
      

      a, b = x[..., 0], x [..., 1]

      cos_vec = self.cos_vec_table[token_positions]
      sin_vec = self.sin_vec_table[token_positions]

      a_rotated = a * cos_vec - b * sin_vec
      b_rotated = a * sin_vec + b * cos_vec

      # Merge back them a = (..., T, couple_index) +  -> (..., T, 2)
      #                 b = (..., T, couple_index) ^
      merged = torch.stack((a_rotated, b_rotated), dim = -1)

      # Now i can rearrange merged (..., T, couple_Index, 2) ---> (..., T, couple_index * 2)

      result = rearrange(merged, "... T couple_index couple-> ... T (couple_index couple)")


      return result
