from einops import einsum
from llm_library.nn_components.softmax import softmax


def scaled_dot_product_attention(query, key, value, mask = None):

   #First step: query key^T
   d_k = key.shape[-1]
   pre_softmax = einsum(query, key, "... seq_len1 d_k, ... seq_len2 d_k -> ... seq_len1 seq_len2") * (d_k ** -0.5)

   #If there is a mask i must apply the mask
   if mask is not None:
      pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))

   weights = softmax(pre_softmax, dimension_i = -1)

   attn = einsum(weights, value, "... seq_len1 seq_len2, ... seq_len2 d_v -> ... seq_len1 d_v")

   return attn


      