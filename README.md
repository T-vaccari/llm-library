# LLM Library

[![CI](https://github.com/T-vaccari/llm-library/actions/workflows/ci.yml/badge.svg)](https://github.com/T-vaccari/llm-library/actions/workflows/ci.yml)

![llm-library architecture](assets/llm-library-architecture.svg)

## What the Library Is

The idea behind this project was to have, in a unique place, implementations of the core ideas that power modern LLMs, unpacked from unnecessary complexity and written in clear Python without other abstraction layers.

The main focus is clarity and, when possible, also efficiency. The goal is to implement the core pieces behind modern LLMs in a way that is understandable to anybody who knows a bit of Python and PyTorch, without relying on other libraries or abstractions when they are not necessary.

I deeply belive in the fact that If I am not able to build something from scratch, then I have not understood deeply that thing. I know that this lead to reinventing the wheel everytime, but mastering the fundamentals is still something that makes the difference
in my opinion ;)

## Why I Built It

The core goal is to engage with the implementation and deeply understand it. What you will find here is handcrafted implementation code, written with the goal of being clear, explicit, and useful for understanding what is happening behind the abstractions. I am working to make it as correct and understandable as I can.

For sure, an AI agent could write much of this repository faster, and probably better, than me. But that is not the point. I hope you will take the same approach if trying to do this on your own.

## Implemented Components

Here is a list of what has been implemented (paper citations are still a work in progress):

- BPE training, encoding, and decoding
- Linear and embedding layers
- Softmax and cross-entropy loss
- RMS normalization
- Rotary positional embeddings
- Scaled dot-product and multi-head attention
- SwiGLU feed-forward network
- Transformer block and language model
- AdamW optimizer
- Learning-rate scheduling and gradient clipping
- Training, checkpointing, and text generation

## Installation and Quick Start

The project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and all the dependencies. After cloning the repository, everything can be installed with:

```bash
uv sync
```

Every Python file can be run inside the managed environment by using:

```bash
uv run python path/to/file.py
```

## Testing

The complete test suite can be run with:

```bash
uv run pytest
```

To run only a specific test:

```bash
uv run pytest tests/path/to/test_file.py::test_name
```

## Training a Small Model

This repository can be used for real to train and play around with a small language model.

I encourage you to play with this repo. Firstly, you need some data, so as a starting point you can use the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). The training and validation files can be downloaded with:

```bash
mkdir -p data
curl -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -o data/TinyStoriesV2-GPT4-train.txt
curl -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt -o data/TinyStoriesV2-GPT4-valid.txt
```

The first step is to train the BPE tokenizer on the training text, obtaining the vocabulary and the list of merges. The trained tokenizer can then be used to encode both the training and validation text into token IDs, which have to be saved as NumPy arrays.

Under `examples/tinystories` you can already find the vocabulary and merges that I obtained after training the BPE tokenizer on TinyStories. If you want to train them from scratch, you can run:

```bash
uv run python -m llm_library.bpe.cli train \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-output examples/tinystories/vocab.pkl \
  --merges-output examples/tinystories/merges.pkl \
  --vocab-size 10000
```

Then, the training and validation files can be tokenized with:

```bash
uv run python -m llm_library.bpe.cli tokenize \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories_train_tokens.npy \
  --vocab examples/tinystories/vocab.pkl \
  --merges examples/tinystories/merges.pkl

uv run python -m llm_library.bpe.cli tokenize \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/tinystories_valid_tokens.npy \
  --vocab examples/tinystories/vocab.pkl \
  --merges examples/tinystories/merges.pkl
```

These arrays, together with the size of the trained vocabulary, can then be passed to the training script to train the language model. The implementation supports CPU, CUDA, and Apple MPS devices thanks to PyTorch.

The configuration below is the one that I used for the provided TinyStories example. It trains for 5,000 steps, corresponding to about 40 million tokens:

```bash
uv run python -m llm_library.training.train \
  --train-path data/tinystories_train_tokens.npy \
  --val-path data/tinystories_valid_tokens.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --num-layers 4 \
  --d-model 512 \
  --num-heads 16 \
  --d-ff 1344 \
  --lr-max 3e-4 \
  --lr-min 3e-5 \
  --warmup-steps 100 \
  --batch-size 32 \
  --num-steps 5000 \
  --eval-interval 100 \
  --eval-batches 10 \
  --checkpoint-interval 1000 \
  --device cuda
```

Under `examples/tinystories` you can also find `16m_tokens.pth` and `80m_tokens.pth`, two checkpoints obtained after training on about 16 and 80 million tokens. You can resume from the first one using the same configuration and adding:

```bash
--resume-checkpoint examples/tinystories/16m_tokens.pth
```

Once you have a checkpoint, you can play with the model and generate text with:

```bash
uv run python -m llm_library.generation.decode \
  --checkpoint examples/tinystories/80m_tokens.pth \
  --vocab-path examples/tinystories/vocab.pkl \
  --merges-path examples/tinystories/merges.pkl \
  --device mps \
  --temperature 0.8 \
  --top-p 0.9 \
  --max-new-tokens 100
```

The generation script opens an interactive prompt, so you can write the beginning of a story and let the model continue it.
Give it a spin, even if it's still a mere english blabber !

### Example Generations

To show how the model changes during training, I used the same prompt with checkpoints trained on approximately 16, 40, and 80 million tokens. All generations used a temperature of `0.5`, top-p of `0.9`, and a maximum of 150 new tokens.

The prompt was:

```text
Once upon a time, there was a little girl named Lily.
```

After 16 million tokens:

```text
Once upon a time, there was a little girl named Lily. Lily loved to play outside in the sun. One day, she saw a big tree with a small hole in it. She wanted to touch it, but it was too high.

Lily asked her mom, "Can I touch the hole?" Her mom said, "No, Lily. It is not safe." Lily did not listen to her mom. She kept trying to make a mess, but she did not want to play with her.

Lily's mom had to clean up the mess. She said, "Lily, you should not play with your toys. You should not play with the ball. You should not play with the ball." Lily felt sad and angry. She said, "I'm sorry, mom. I was wrong.
```

After 40 million tokens:

```text
Once upon a time, there was a little girl named Lily. She loved to play with her toy bunny. One day, Lily saw a big, round ball in her yard. She wanted to play with it, but she was scared.

Lily asked her friend Tom, "Can I play with the ball?" Tom said, "No, it is my ball." Lily was sad, but she did not give up. She kept playing with the big, round ball.

Later, Lily's mom came into the yard. She saw the big, round, and said, "Lily, that's not a ball. It's a big, round, and shiny ball. It's a special gift for your friend, Tim." Lily was so happy, and she hugged her friend. They played with the big
```

After 80 million tokens:

```text
Once upon a time, there was a little girl named Lily. She loved to play with her toy train. One day, she was playing with her train when it started to rain. Lily was sad and didn't know what to do.

Lily's mommy saw her sad face and asked her what was wrong. Lily told her mommy that she had to go home. Her mommy said, "Don't worry, Lily. We can go back to the store and get your train."

So, Lily and her mommy went to the store. Lily saw a big box of toys and thought it would be a good idea to play with them. They played with the toys and had lots of fun. Lily was happy that she could play with her train, even if it was not wet.

<|endoftext|>
```

These are still small models trained on a limited dataset, but the comparison shows how the generations become more coherent as the training progresses.

## Limitations

For now, the implementation relies on the tensors offered by PyTorch and its built-in autograd. One day, I hope to substitute this dependency with an autograd system implemented here using custom tensors.

In some parts of the implementation I rely on `einops` to describe tensor operations at a higher level and make them clearer. This is done only for readability and it is not an architectural dependency: under the hood, these operations still call the corresponding PyTorch kernels.

## Roadmap

I would like to implement an autograd engine and to keep up with the most recent developments in LLM architecture.

As follow-up work, I would also like to implement the performance-critical pieces efficiently in C++.

## Acknowledgements

The implementations are supported by correctness tests adapted from the Stanford CS336 course, originally provided as part of the assignment for implementing these components.

If you find an error—or your agents find one—please open an issue or a pull request.

## Related Blog Post

Still a work in progress. I am preparing a deep dive for my blog that explains every single piece implemented here, the reasoning behind it, and what I learned while building it.
