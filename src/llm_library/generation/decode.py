import argparse

import torch

from llm_library.bpe.tokenizer import Tokenizer
from llm_library.models.transformer_lm import TransformerLM


def sample_next_token(logits, temperature=1.0, top_p=1.0):
   if temperature <= 0:
      raise ValueError("temperature must be greater than 0")
   if not 0 < top_p <= 1:
      raise ValueError("top_p must be between 0 and 1")

   probs = torch.softmax(logits / temperature, dim=-1)

   if top_p < 1.0:
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      mask = cumulative_probs > top_p
      mask[1:] = mask[:-1].clone()
      mask[0] = False
      sorted_probs[mask] = 0
      sorted_probs = sorted_probs / sorted_probs.sum()
      next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
   else:
      next_token = torch.multinomial(probs, 1)

   return next_token.item()


def load_model_checkpoint(model, checkpoint):
   state = torch.load(
      checkpoint,
      map_location=next(model.parameters()).device,
      weights_only=True,
   )
   model_state = {
      key.removeprefix("_orig_mod."): value
      for key, value in state["model"].items()
   }
   model.load_state_dict(model_state)
   model.eval()

   return model


@torch.no_grad()
def generate(model, tokenizer, prompt, eos_token_id, context_length, max_new_tokens=100, temperature=1.0, top_p=1.0):
   token_ids = tokenizer.encode(prompt)

   if not token_ids:
      raise ValueError("prompt must contain at least one token")

   tokens = torch.tensor(
      token_ids,
      dtype=torch.long,
      device=next(model.parameters()).device,
   ).unsqueeze(0)

   for _ in range(max_new_tokens):
      model_input = tokens[:, -context_length:]
      logits = model(model_input)
      next_token = sample_next_token(logits[0, -1, :], temperature, top_p)
      next_token_tensor = torch.tensor([[next_token]], device=tokens.device)
      tokens = torch.cat([tokens, next_token_tensor], dim=1)

      if next_token == eos_token_id:
         break

   return tokenizer.decode(tokens[0].tolist())


def repl(model, tokenizer, eos_token_id, context_length, max_new_tokens=100, temperature=1.0, top_p=1.0):
   while True:
      prompt = input("> ")

      if prompt.lower() in ["quit", "exit"]:
         break
      if not prompt:
         continue

      text = generate(
         model,
         tokenizer,
         prompt,
         eos_token_id,
         context_length,
         max_new_tokens,
         temperature,
         top_p,
      )
      print(text)


def parse_args():
   parser = argparse.ArgumentParser(description="Generate text from a trained language model")

   parser.add_argument("--checkpoint", type=str, required=True)
   parser.add_argument("--vocab-path", type=str, default="data/tinystories_vocab.pkl")
   parser.add_argument("--merges-path", type=str, default="data/tinystories_merges.pkl")
   parser.add_argument("--vocab-size", type=int, default=10000)
   parser.add_argument("--context-length", type=int, default=256)
   parser.add_argument("--num-layers", type=int, default=4)
   parser.add_argument("--d-model", type=int, default=512)
   parser.add_argument("--num-heads", type=int, default=16)
   parser.add_argument("--d-ff", type=int, default=1344)
   parser.add_argument("--rope-theta", type=float, default=10000.0)
   parser.add_argument("--max-new-tokens", type=int, default=100)
   parser.add_argument("--temperature", type=float, default=0.8)
   parser.add_argument("--top-p", type=float, default=0.9)
   parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])

   return parser.parse_args()


def main(args):
   tokenizer = Tokenizer.from_files(
      args.vocab_path,
      args.merges_path,
      special_tokens=["<|endoftext|>"],
   )
   eos_token_id = tokenizer.reversed_vocab[b"<|endoftext|>"]

   model = TransformerLM(
      vocab_size=args.vocab_size,
      context_length=args.context_length,
      num_layers=args.num_layers,
      d_model=args.d_model,
      num_heads=args.num_heads,
      d_ff=args.d_ff,
      rope_theta=args.rope_theta,
   ).to(args.device)

   model = load_model_checkpoint(model, args.checkpoint)
   print("loaded", args.checkpoint)

   repl(
      model,
      tokenizer,
      eos_token_id,
      args.context_length,
      args.max_new_tokens,
      args.temperature,
      args.top_p,
   )


if __name__ == "__main__":
   main(parse_args())
