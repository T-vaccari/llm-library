import torch

def softmax(x, dimension_i):
   """
   Apply the classic softmax over a tensor on the ith dimension_i, for numerical stability
   it's necessary to subtract the max on that dimension before applying softmax.
   
   """
   
   x = x - torch.max(x, dim = dimension_i, keepdim=True).values
   
   return torch.exp(x) / torch.sum(torch.exp(x), dim = -1, keepdim=True)