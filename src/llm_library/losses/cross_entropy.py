import torch
from einops import rearrange


"""
Here we are calculating the cross entropy loss on the whole batch that enters in.
Expected x: (...,T, vocab_size) expected y: (...,T) where in each step in x I have the probs distribution after
the softmax and instead in y the index of the correct token.

Starting with the definition of an encoding for a given vocab we can define entropy. Now take the 
encoding provided by our model, we have a set of probs over the vocab distribution for a given prediction.
That encoding schema is optimzed for what the model has seen during training. What we want to do is to 
see how the encoding schema that the model has optimized, performs on the real data distribution, for doing that
we measure the entropy of the other distribution q, but using the encoding schema of the former distribution p.
This is what is defined as cross entropy loss. Given that over a prediction the distribution q has one hot for the prediction
we obtain probs_token_on_q(100%) * log_probs_of_that_token(p_x_i).
"""

def cross_entropy_loss(logits, targets):

   # x : (..., T, vocab_size) -> (..., T) where vocab size is reduced as follows, fixed j in 0:T-1 :
   # -x[y[j]] + log(sum(x[...,J, :]))

   #Apply numerical stability trick, removing the max from the last dimension

   maxes = torch.max(logits, dim = -1, keepdim=True).values
   logits -= maxes

   targets = rearrange(targets, "... -> ... 1")

   result = torch.gather(logits, dim = -1, index = targets)
   result = - result + torch.log(torch.sum(torch.exp(logits), dim = -1, keepdim=True))

   # Then i could return the grand mean of the resulting matrix
   result = torch.mean(result)

   return result


   



