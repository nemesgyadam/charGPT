import torch

chars = [*open("charset.txt", "r").read()]
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


model = torch.load('model.pth')
print('Model loaded.')
start_text = ' '
print(decode(model.predict(encode(start_text), max_new_tokens=1000).tolist()))