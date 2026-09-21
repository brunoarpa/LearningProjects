import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 64
BATCH_SIZE = 32
LR = 3e-4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("bnc_spoken.train", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2,special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tokenizer.train_from_iterator([text], trainer)
data = np.array(tokenizer.encode(text).ids, dtype=np.int32)
val_start = int(len(data) * 0.9)
train_data, val_data = data[:val_start], data[val_start:]
batches_per_epoch = len(train_data) // (BATCH_SIZE * CONTEXT_LENGTH)

def get_batch(split, batch_size, context_length):
    dataset = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(dataset) - context_length - 1, size=(batch_size,))
    x = np.stack([dataset[i:i+context_length] for i in ix])
    y = np.stack([dataset[i+1:i+context_length+1] for i in ix])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, n_layers=2, context=CONTEXT_LENGTH):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            activation="relu", dropout=0.0, norm_first=True, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)  # stacks N of them

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):                       # idx: (B, T)
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)                       # (B, T, d_model)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)  # causal mask
        x = self.blocks(x, mask=mask)             # all the attention/FFN/residual/LN happens here
        x = self.ln_f(x)
        return self.head(x)                        # (B, T, vocab)


model = GPT(VOCAB_SIZE).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(100):
    for step in range(batches_per_epoch):
        xb, yb = get_batch('train', BATCH_SIZE, CONTEXT_LENGTH)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(epoch, step, loss.item())
