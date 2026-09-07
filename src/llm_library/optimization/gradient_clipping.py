import torch
import math
def grad_clipping(params, max_norm , eps = 1e-6):
   squared_grad_sum = 0
   for p in params:
      if p is None or p.grad is None:
         continue
      squared_grad_sum += torch.sum(p.grad ** 2)

   norm = math.sqrt(squared_grad_sum)
   if norm < max_norm:
      return norm

   factor =  max_norm/(norm + eps) 
   for p in params:
      if p is None or p.grad is None:
         continue
      p.grad *= factor

   return norm


