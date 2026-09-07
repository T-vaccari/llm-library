# About the Project: LLM Library

The idea behind this project was to have, in a unique place, implementations of the core ideas that power modern LLMs unpacked by the unnecessary complexity and written in clear python without other abstractions layers.

The main focus is clarity and, when possible, also efficiency. The goal is to implement the core pieces behind modern LLMs in a way that is understandable to anybody who knows a bit of Python and PyTorch, without relying on other libraries/abstraction when not necessary. For istance as now the only abstraction that we are using are the tensors offered by pytorch and the built-in autograd(one day I hope to substitute this dependency with an here implemented autograd systems on custom tensors)

The implementations are supported by correctness tests adapted from the Stanford CS336 course, originally provided as part of the assignment for implementing these components.

From now on, I want to keep track of new ideas appearing in the LLM architecture landscape, study them properly, and continuously integrate them here together with their tests.

## AI Disclaimer

For sure, an AI agent could write much of this repository faster, and probably better, than me. But that is not the point.

What you will find here is handcrafted implementation code, written with the goal of being clear, explicit, and useful for understanding what is happening behind the abstractions. I am working to make it as correct and understandable as I can.

The core goal is to engage with the implementation and deeply understand it, so using an agents or any AI assistance would make no sense. I hope you will do the same if trying to do this on your own.

If you find an error—or your agents find one—please open an issue or a pull request.

## Contents and Related Papers

Here is a list of what has been implemented (papers citation are still a wip):

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
- Autograd engine( still a work in progress)

## Further Development

I would like to implement an autograd engine, and to keep up with the most recent developments in LLMs architecture.
As follow-up work, I would also like to implement the performance-critical pieces efficiently in C++.

## Related Blog Post

Still a work in progress. I am preparing a deep dive for my blog that explains every single piece implemented here, the reasoning behind it, and what I learned while building it.

## How to use it

The project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and all the dependencies. After cloning the repository, everything can be installed with:

```bash
uv sync
```

Every Python file can be run inside the managed environment by using:

```bash
uv run python path/to/file.py
```

The complete test suite can be run with:

```bash
uv run pytest
```

To run only a specific test:

```bash
uv run pytest tests/path/to/test_file.py::test_name
```

This repository can look like as a clunky mess of python, but I swear to you that you can try something that blabbel out something fromthis, it can be used for real to train and play around with a small language model. 

I encorauge you to play with this repo. Firstly you need some data, so as a starting point you can use the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), which can be downloaded and divided into training and validation data.

The first step is to train the BPE tokenizer on the training text, obtaining the vocabulary and the list of merges. The trained tokenizer can then be used to encode both the training and validation text into token IDs, which have to be saved as NumPy arrays.

These arrays, together with the size of the trained vocabulary, can then be passed to the training script to train the language model. The implementation supports CPU, CUDA and Apple MPS devices thanks to PyTorch.

I hope to also add a separate branch or release with an already trained version of the model, so that it will be possible to directly play with it without having to run the entire training process.

In some parts of the implementation I rely on `einops` to describe tensor operations at a higher level and make them clearer. This is done only for readability and it is not an architectural dependency: under the hood, these operations still call the corresponding PyTorch kernels.