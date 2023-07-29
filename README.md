# GPT Word-Level Implementation

Welcome to this repository which is home to a Python-based implementation of the GPT model, specifically optimized at the word-level. The architectural design and coding approach of the model have been drawn from the [stellar tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) provided by Andrej Karpathy. We extend our heartfelt gratitude towards him for the conceptual understanding and inspiration.

## Example result:
12M parameter version of the word GPT model trained on a A10 for ~20minutes generated the following Shakespeare aftertaste:
```text
How now, how long-fortune's death, and true not
Could be yield quarrell'd: and some part of every moe
From that triumpe upon thy households from
the harmless apply mock, laid with
your givings of less your hands in their hands and time
In grial oyly counting enough, as societ
To unfinish the simple points
Lechange the business on't.

TRANIO:
Well say you, you must be so,
And such more born so betimate,
Is your pleasure. What's your worship?

LUCIO:
I am a one nothing of this?

LUCIO:
Kise o' the heaven and less fear that I can seest
So may put I a full spiciote. I were some of worse
to end anded then; but, I know there come, but thy
stab, though find, us and dance is founting and clap the
limony is pericherit in thy servant courst
it yields it in servahing the wealth on the city; you
which is wipe and as men servitors will approve
hence the capt of sour and such necessary.
Three every man of thy bustiness
disconjurate law to waith every any confidence
```
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