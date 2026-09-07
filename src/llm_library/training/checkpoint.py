import torch

def save_checkpoint(model, optimizer, iteration, out):
   """
   should dump all the state from the
   model, optimizer and iteration into the file-like object out. You can use the state_dict
   method of both the model and the optimizer to get their relevant states and use
   torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like
   object here). A typical choice is to have obj be a dictionary, but you can use whatever format
   you want as long as you can load your checkpoint later.
   
   """

   # Obj is going to be a dictionary

   obj = dict()
   obj["model"] = model.state_dict()
   obj["optimizer"] = optimizer.state_dict()
   obj["iteration"] = iteration

   torch.save(obj, out)
   
   return out

def load_checkpoint(src, model, optimizer):
   obj = torch.load(src)
   model.load_state_dict(obj["model"])
   optimizer.load_state_dict(obj["optimizer"])

   return obj["iteration"]

