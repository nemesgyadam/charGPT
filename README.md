# GPT Word-Level Implementation

Welcome to this repository which is home to a Python-based implementation of the GPT model, specifically optimized at the word-level. The architectural design and coding approach of the model have been drawn from the [stellar tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) provided by Andrej Karpathy. We extend our heartfelt gratitude towards him for the conceptual understanding and inspiration.

## Prerequisites

The codebase here necessitates the following Python libraries:

* PyTorch (torch)

If these libraries are not already installed in your environment, you can easily do so with pip:

```sh
pip install torch
```

## Usage
This repository features two distinct models - a basic bigram model and a more advanced, GPT2 model replica.

### Training
To train one of these models, all you need to do is **run the train.ipynb** notebook.

### Predicting
For generating text using these models:

```sh
python predict.py
```



## License and Acknowledgment

The original concepts and tutorial that this repository is based on were crafted by Andrej Karpathy. All associated rights are his. The intention of this repository and its contents is purely educational, aimed at spreading knowledge.