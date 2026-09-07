from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
   def __init__(self, params, lr=1e-3, betas = (0.9, 0.95), weight_decay = 0.1 , eps = 1e-5):
      if lr < 0:
         raise ValueError(f"Invalid learning rate: {lr}")
      
      defaults = {"alpha": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
      super().__init__(params, defaults)

   def step(self, closure: Optional[Callable] = None):
      loss = None if closure is None else closure()
      for group in self.param_groups:
         alpha = group["alpha"] # Get the learning rate.
         b1 =  group["betas"][0]
         b2 = group["betas"][1]
         eps = group["eps"]
         weight_decay = group["weight_decay"]

         for p in group["params"]:
            if p.grad is None:
               continue
            state = self.state[p] # Get state associated with p.
            t = state.get("t", 1) # Get iteration number from the state, or 0.
            m = state.get("m", 0)
            v = state.get("v", 0)
            grad = p.grad.data # Get the gradient of loss with respect to p.

            #Update alpha
            alpha_t =  alpha * ((math.sqrt(1 - b2**t)) / (1 - b1**t))

            #Update applying weight decays
            p.data = p.data - alpha * weight_decay * p.data # We want to squeeze gradients toward zero

            #Update first moment estimate
            m = b1 * m + (1 - b1) * grad 

            #Update second moment

            v = b2 * v + (1 - b2) * grad**2

            #Update applying running m and v
            p.data = p.data - alpha_t * ((m)/(torch.sqrt(v) + eps))

           
            state["t"] = t + 1 # Increment iteration number.
            state["m"] = m
            state["v"] = v

      
      return loss