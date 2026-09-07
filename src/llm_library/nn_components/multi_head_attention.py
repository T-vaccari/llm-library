import torch
from einops import einsum, rearrange
from torch import nn
from llm_library.nn_components.linear import Linear
from llm_library.nn_components.rope import RotaryPositionalEmbedding


class CausalMultiHeadSelfAttention(nn.Module):

   def __init__(self, d_model, num_heads,  max_seq_len, theta = None, device = None, dtype = None):
      super().__init__()

      self.d_model = d_model
      self.num_heads = num_heads
      self.d_head = self.d_model // self.num_heads
      self.max_seq_len = max_seq_len
      assert self.d_model % self.num_heads == 0
      if theta is not None:
         self.rope = RotaryPositionalEmbedding(theta=theta, d_k = self.d_head, max_seq_len=max_seq_len, device=device)
         
      self.theta = theta
      #combining the key, query, and value projections into a single weight matrix so you only need a
      #single matrix multiply. x enters as (B, T, d_model) then i need to project with three matrices x down to q, k and v.
      # Instead of using three different matrices I can use a linear layer with 3 * d_model
      
      self.packed_qkv_matrix = Linear(self.d_model, 3 * self.d_model, device=device, dtype=dtype)
      self.register_buffer("causal_mask",torch.tril(torch.ones((max_seq_len, max_seq_len),dtype=torch.bool,device=device)),persistent=False)
      
      self.proj = Linear(d_model, d_model, device=device, dtype= dtype)
      

   def forward(self, x, token_positions = None):
      T = x.shape[-2]
      packed_qkv_projection = self.packed_qkv_matrix(x) # (B, T, 3 * d_model)
      q, k, v = torch.split(packed_qkv_projection,self.d_model,dim=-1) #Each one now has shape (B, T, d_model)

      # Reshape q, k, v to treat num_heads as a batch dimension (B, T, d_model) -> (B, num_head, T, d_head)
      q = rearrange(q, "B T (num_head d_head) -> B num_head T d_head",num_head =  self.num_heads , d_head = self.d_head)
      k = rearrange(k, "B T (num_head d_head) -> B num_head T d_head",num_head =  self.num_heads , d_head = self.d_head)
      v = rearrange(v, "B T (num_head d_head) -> B num_head T d_head",num_head =  self.num_heads , d_head = self.d_head)

      # Before applying attention I need to apply rope to q and k
      if self.theta is not None:
            
         if token_positions is None:
            token_positions = torch.arange(T, device=x.device)
         q = self.rope(q, token_positions)
         k = self.rope(k, token_positions)
      

      # I need to prepare the mask for the attention, it's better to have a mask precomputed and slice only for the 
      # needed len
      # I need a boolean mask that has false on the triangular upper part
      assert isinstance(self.causal_mask, torch.Tensor)
      from llm_library.nn_components.scaled_dot_product_attention import scaled_dot_product_attention
      multi_head = scaled_dot_product_attention(q, k, v, self.causal_mask[:T, :T]) #(B, num_heads, T, d_head)

      # I need to reshape res into multihead res -> (B, T, head1|head2 etc)
      
      multi_head = rearrange(multi_head, "B num_heads T d_head -> B T (num_heads d_head)")

      # Then I need the final projection that mixes up the feature result of concatenated heads

      res = self.proj(multi_head)

      return res
      
      



      
      