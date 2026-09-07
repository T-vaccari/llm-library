import argparse
import pickle

import numpy as np

from llm_library.bpe.bpe_training import train_bpe
from llm_library.bpe.tokenizer import Tokenizer


def train(args):
    special_tokens = args.special_token or ["<|endoftext|>"]
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)

    with open(args.vocab_output, "wb") as f:
        pickle.dump(vocab, f)
    with open(args.merges_output, "wb") as f:
        pickle.dump(merges, f)


def tokenize(args):
    special_tokens = args.special_token or ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens)

    with open(args.input, encoding="utf-8") as f:
        token_ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

    np.save(args.output, token_ids)
    print(f"Saved {len(token_ids):,} tokens to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train and use the BPE tokenizer")
    commands = parser.add_subparsers(required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--vocab-output", required=True)
    train_parser.add_argument("--merges-output", required=True)
    train_parser.add_argument("--vocab-size", type=int, default=10_000)
    train_parser.add_argument("--special-token", action="append")
    train_parser.set_defaults(run=train)

    tokenize_parser = commands.add_parser("tokenize")
    tokenize_parser.add_argument("--input", required=True)
    tokenize_parser.add_argument("--output", required=True)
    tokenize_parser.add_argument("--vocab", required=True)
    tokenize_parser.add_argument("--merges", required=True)
    tokenize_parser.add_argument("--special-token", action="append")
    tokenize_parser.set_defaults(run=tokenize)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
