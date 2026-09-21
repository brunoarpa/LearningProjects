import torch
import torch.nn as nn
import numpy as np
import pandas as pd

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train = pd.read_csv("train.csv")

X_train = train.drop(columns="label")/255 # Normalize from 0-1
Y_train = train["label"]
X = torch.tensor(X_train.values, dtype=torch.float32)   # shape (N, 784)
Y = torch.tensor(Y_train.values, dtype=torch.long)      # shape (N,)
X = X.to(device)
Y = Y.to(device)
n_val = 1000
X_train_, Y_train_ = X[:-n_val], Y[:-n_val]   
X_val,    Y_val    = X[-n_val:], Y[-n_val:]   

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        a1 = torch.relu(self.fc1(x))
        a2 = torch.relu(self.fc2(a1))
        z3 = self.fc3(a2)
        return z3

EPOCHS = 10
LR = 0.1

model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for start in range(0,len(X_train_),64):
        xb = X_train_[start:start+64]
        yb = Y_train_[start:start+64]
        logits = model.forward(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(epoch, loss.item())

with torch.no_grad():
    val_logits = model(X_val)
    preds = val_logits.argmax(dim=1)        
    accuracy = (preds == Y_val).float().mean()
    print("val accuracy:", accuracy.item())
