import torch

def data_loader(x, batch_size, context_length, device = 'mps'):

   #I need to sample batch_size starting indexes for the sequence and retrieve the shifted context
   # The valid range for a starting index is [0, x.size() - context_length-1], given that
   # I must have room also for the shifted of 1 token prediction
   
   idx = torch.randint( size = (batch_size,1), high = x.shape[-1]- context_length , low =0 )
   offsets = torch.arange(context_length )

   tokens_idx = idx + offsets
   target_idx = tokens_idx + 1

   tokens = torch.tensor(x[tokens_idx], dtype=torch.long, device=device)
   target = torch.tensor(x[target_idx], dtype=torch.long, device=device)

   return (tokens , target)

   
