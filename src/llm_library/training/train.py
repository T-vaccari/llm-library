import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from llm_library.models.transformer_lm import TransformerLM
from llm_library.optimization.cosine_learning_rate_schedule import cosine_lr_scheduler
from llm_library.training.data_loader import data_loader
from llm_library.losses.cross_entropy import cross_entropy_loss
from llm_library.optimization.gradient_clipping import grad_clipping
from llm_library.training.checkpoint import load_checkpoint, save_checkpoint
from llm_library.optimization.adamw import AdamW

"""
knobs:
data
   train_path
   val_path

model
   vocab_size
   context_length
   num_layers
   d_model
   num_heads
   d_ff
   rope_theta

optimizer
   lr_max
   lr_min
   warmup_steps
   cosine_steps
   betas
   eps
   weight_decay
   max_grad_norm

training
   run_name
   experiments_dir
   batch_size
   num_steps
   eval_interval
   eval_batches
   checkpoint_interval
   resume_checkpoint
   device
"""


def create_experiment(args):
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   run_dir = os.path.join(args.experiments_dir, f"{args.run_name}_{timestamp}")
   checkpoint_dir = os.path.join(run_dir, "checkpoints")
   metrics_path = os.path.join(run_dir, "metrics.csv")

   os.makedirs(checkpoint_dir, exist_ok=False)

   config = {
      "data": {
         "train_path": args.train_path,
         "val_path": args.val_path,
      },
      "model": {
         "vocab_size": args.vocab_size,
         "context_length": args.context_length,
         "num_layers": args.num_layers,
         "d_model": args.d_model,
         "num_heads": args.num_heads,
         "d_ff": args.d_ff,
         "rope_theta": args.rope_theta,
      },
      "optimizer": {
         "lr_max": args.lr_max,
         "lr_min": args.lr_min,
         "warmup_steps": args.warmup_steps,
         "cosine_steps": args.cosine_steps,
         "betas": args.betas,
         "eps": args.eps,
         "weight_decay": args.weight_decay,
         "max_grad_norm": args.max_grad_norm,
      },
      "training": {
         "run_name": args.run_name,
         "experiments_dir": args.experiments_dir,
         "batch_size": args.batch_size,
         "num_steps": args.num_steps,
         "eval_interval": args.eval_interval,
         "eval_batches": args.eval_batches,
         "checkpoint_interval": args.checkpoint_interval,
         "resume_checkpoint": args.resume_checkpoint,
         "device": args.device,
      },
   }

   config_text = json.dumps(config, indent=3)

   print("Run directory:", run_dir)

   with open(os.path.join(run_dir, "config.json"), "w") as f:
      f.write(config_text + "\n")

   with open(metrics_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
         "step",
         "wall_time_s",
         "train_loss",
         "val_loss",
         "lr",
         "tokens_per_sec",
         "gradient_rms",
         "weight_rms",
      ])

   return checkpoint_dir, metrics_path


def write_metrics(
   metrics_path,
   step,
   wall_time_s,
   train_loss,
   val_loss,
   lr,
   tokens_per_second,
   gradient_rms,
   weight_rms,
):
   with open(metrics_path, "a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
         step,
         wall_time_s,
         train_loss,
         val_loss,
         lr,
         tokens_per_second,
         gradient_rms,
         weight_rms,
      ])


def parameter_rms(parameters, use_grad=False):
   squared_sum = 0.0
   num_elements = 0

   for parameter in parameters:
      tensor = parameter.grad if use_grad else parameter

      if tensor is not None:
         tensor = tensor.detach().float()
         squared_sum += tensor.square().sum()
         num_elements += tensor.numel()

   return torch.sqrt(squared_sum / num_elements).item()


def evaluate(model, val_data, args):
   model.eval()

   losses = []

   with torch.no_grad():
      for _ in range(args.eval_batches):
         
         x, y = data_loader(val_data, args.batch_size, args.context_length, device = args.device)

         logits = model(x)
         loss = cross_entropy_loss(logits, y)
         losses.append(loss.item())

   model.train()

   return sum(losses) / len(losses)



def main(args):
   checkpoint_dir, metrics_path = create_experiment(args)
   run_start_time = time.perf_counter()

   device = args.device

   # Retrieve data from disk, with mmap, so they are not loaded in bulk into memory
   train_data = np.load(args.train_path, mmap_mode="r")
   val_data = np.load(args.val_path, mmap_mode="r")

   print(f"Training tokens: {len(train_data):,}")
   print(f"Validation tokens: {len(val_data):,}")

   # Instantiate the model
   model = TransformerLM(
      vocab_size=args.vocab_size,
      context_length=args.context_length,
      num_layers=args.num_layers,
      d_model=args.d_model,
      num_heads=args.num_heads,
      d_ff=args.d_ff,
      rope_theta=args.rope_theta,
   )

   model = model.to(device)
   
   # Instantiate the optimizer
   optimizer = AdamW(
      model.parameters(),
      lr=args.lr_max,
      betas=tuple(args.betas),
      eps=args.eps,
      weight_decay=args.weight_decay,
   )

   start_step = 0

   # Resume training from checkpoint
   if args.resume_checkpoint is not None:
      checkpoint_step = load_checkpoint(args.resume_checkpoint, model, optimizer)
      start_step = checkpoint_step + 1

      print(f"Resuming training from step {start_step}")



   model.train()
   if args.device ==  "mps":
      model = torch.compile(model, backend="aot_eager")
   if args.device == "cuda":
      model = torch.compile(model)
      # torch.set_float32_matmul_precision('high')

   for step in range(start_step, args.num_steps):
      
      lr = cosine_lr_scheduler(step, args.lr_max, args.lr_min, args.warmup_steps, args.cosine_steps)


      for param_group in optimizer.param_groups:
         param_group["lr"] = lr

      
      x, y = data_loader(train_data, args.batch_size, args.context_length, device = device)

      optimizer.zero_grad()

      start = time.perf_counter()
      logits = model(x)
      loss = cross_entropy_loss(logits, y)

      
      loss.backward()

      should_evaluate = step % args.eval_interval == 0

      if should_evaluate:
         gradient_rms = parameter_rms(model.parameters(), use_grad=True)

      
      norm_pre_clip = grad_clipping(
         model.parameters(),
         args.max_grad_norm,
      )

      
      optimizer.step()

      if should_evaluate:
         weight_rms = parameter_rms(model.parameters())

      elapsed = time.perf_counter() - start
      tokens_this_step = args.batch_size * args.context_length
      tokens_per_second = tokens_this_step / elapsed
      

      if should_evaluate:
         
         val_loss = evaluate(model, val_data, args)

         print(
            f"step {step:6d} | "
            f"lr {lr:.6e} | "
            f"loss {loss: .6e} | "
            f"val_loss {val_loss: .6e} | "
            f"grad_norm {norm_pre_clip: .6e} | "
            f"grad_rms {gradient_rms: .6e} | "
            f"weight_rms {weight_rms: .6e} | "
            f"wall_clock_time {elapsed: .6e} | "
            f"Tok/s:{tokens_per_second}"
         )
         

         write_metrics(
            metrics_path,
            step,
            time.perf_counter() - run_start_time,
            loss.item(),
            val_loss,
            lr,
            tokens_per_second,
            gradient_rms,
            weight_rms,
         )

      if step > 0 and step % args.checkpoint_interval == 0:
         checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
         save_checkpoint(model, optimizer, step, checkpoint_path)

   checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
   save_checkpoint(model, optimizer, step, checkpoint_path)


def parse_args():
   parser = argparse.ArgumentParser(description="Train language model")

   # -------------------------------------------------------------------------
   # Data
   # -------------------------------------------------------------------------
   data = parser.add_argument_group("data")

   data.add_argument(
      "--train-path",
      type=str,
      required=True,
      help="Path to the training dataset",
   )

   data.add_argument(
      "--val-path",
      type=str,
      required=True,
      help="Path to the validation dataset",
   )

   # -------------------------------------------------------------------------
   # Model
   # -------------------------------------------------------------------------
   model = parser.add_argument_group("model")

   model.add_argument("--vocab-size", type=int, required=True)
   model.add_argument("--context-length", type=int, required=True)
   model.add_argument("--num-layers", type=int, required=True)
   model.add_argument("--d-model", type=int, required=True)
   model.add_argument("--num-heads", type=int, required=True)
   model.add_argument("--d-ff", type=int, required=True)

   model.add_argument(
      "--rope-theta",
      type=float,
      default=10000.0,
   )

   # -------------------------------------------------------------------------
   # Optimizer
   # -------------------------------------------------------------------------
   optimizer = parser.add_argument_group("optimizer")

   optimizer.add_argument("--lr-max", type=float, required=True)
   optimizer.add_argument("--lr-min", type=float, required=True)

   optimizer.add_argument(
      "--warmup-steps",
      type=int,
      required=True,
   )

   optimizer.add_argument(
      "--cosine-steps",
      type=int,
      default=None,
   )

   optimizer.add_argument(
      "--betas",
      type=float,
      nargs=2,
      metavar=("BETA1", "BETA2"),
      default=(0.9, 0.95),
   )

   optimizer.add_argument(
      "--eps",
      type=float,
      default=1e-8,
   )

   optimizer.add_argument(
      "--weight-decay",
      type=float,
      default=0.1,
   )

   optimizer.add_argument(
      "--max-grad-norm",
      type=float,
      default=1.0,
   )

   # -------------------------------------------------------------------------
   # Training
   # -------------------------------------------------------------------------
   training = parser.add_argument_group("training")

   training.add_argument(
      "--run-name",
      type=str,
      default="tinystories",
   )

   training.add_argument(
      "--experiments-dir",
      type=str,
      default="experiments",
   )

   training.add_argument(
      "--batch-size",
      type=int,
      required=True,
   )

   training.add_argument(
      "--num-steps",
      type=int,
      required=True,
   )

   training.add_argument(
      "--eval-interval",
      type=int,
      default=100,
   )

   training.add_argument(
      "--eval-batches",
      type=int,
      default=10,
   )

   training.add_argument(
      "--checkpoint-interval",
      type=int,
      default=1000,
   )

   training.add_argument(
      "--resume-checkpoint",
      type=str,
      default=None,
      help="Path to checkpoint to resume training from",
   )

   training.add_argument(
      "--device",
      type=str,
      default="cuda",
      choices=["cpu", "cuda", "mps"],
   )

   return parser.parse_args()


if __name__ == "__main__":
   args = parse_args()
   if args.cosine_steps is None:
      args.cosine_steps = args.num_steps
   main(args)
