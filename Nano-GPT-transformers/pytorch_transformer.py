import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 64
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

with open("bnc_spoken.train", "r", encoding="utf-8") as f:
    text = f.read()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tokenizer.train_from_iterator([text], trainer)
data = tokenizer.encode(text).ids
data = np.array(data, dtype=np.int32)
vocab = list(tokenizer.get_vocab().keys())  
val_start = int(len(data) * 0.9)
train_data = data[:val_start]
val_data = data[val_start:]
batches_per_epoch = len(train_data) // (BATCH_SIZE * CONTEXT_LENGTH) 


def get_batch(split, batch_size, context_length):
    
    dataset = train_data if split == 'train' else val_data

    ix = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))
    x = np.stack([dataset[i:i+context_length] for i in ix])
    y = np.stack([dataset[i+1:i+context_length+1] for i in ix])
    x = torch.from_numpy(x).long()   # (B, T) token ids
    y = torch.from_numpy(y).long()
    return x, y

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, head_size=32, d_ff=512): 
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key   = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)))
        self.fc1    = nn.Linear(d_model, d_ff)
        self.fc2    = nn.Linear(d_ff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = d_model // head_size

    def forward(self, x):  # (B, T, d_model)
        B,T,d_model = x.shape
        xn = self.ln1(x)
        q = self.query(xn)
        k = self.key(xn)
        v = self.value(xn)
        q = q.view(B, T, self.n_heads, d_model // self.n_heads).transpose(1, 2)   # (B, n_heads, T, head_size)
        k = k.view(B, T, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, d_model // self.n_heads).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)   # (B, n_heads, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        out = wei @ v                                           # (B, n_heads, T, head_size)
        out = out.transpose(1, 2).reshape(B, T, d_model)   # (B, T, d_model)
        out = self.proj(out)                          # W_O mixes across heads
        out = out + x
        a1 = torch.relu(self.fc1(self.ln2(out)))
        z2 = self.fc2(a1)
        z2 = z2 + out
        return z2

class Transformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, d_ff=512, head_size=32, n_layers=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)   # your self.embeddings
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, head_size, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
        self.head_ln = nn.LayerNorm(d_model)
    def forward(self, idx):            # idx: (B, T) ints
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device) 
        x = self.tok_emb(idx) + self.pos_emb(pos) # (B, T, d_model)
        z1 = self.blocks(x)
        logits = self.head(self.head_ln(z1))
        
        return logits


model = Transformer().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):

    for step in range(batches_per_epoch):

        X_batch, Y_batch = get_batch('train', BATCH_SIZE, CONTEXT_LENGTH)
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        logits = model(X_batch)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), Y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(epoch, step, loss.item())