import torch
from torch import nn

class Embedding(nn.Module):

   def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
      super().__init__()
      self.embedding_table = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
      nn.init.trunc_normal_(self.embedding_table, mean= 0, std=1, a = -3, b = 3)
      
   def forward(self, token_ids):
      # Tokend ids has shape (batch_size, sequence_length) and we return (batch_size, sequence_length, embedding_dim)
      
      return self.embedding_table[token_ids]
      