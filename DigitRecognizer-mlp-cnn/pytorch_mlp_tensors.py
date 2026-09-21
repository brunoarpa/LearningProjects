import torch
import numpy as np
import pandas as pd
train = pd.read_csv("train.csv")

X_train = train.drop(columns="label")/255 # Normalize from 0-1
Y_train = train["label"]
X = torch.tensor(X_train.values, dtype=torch.float32)   # shape (N, 784)
Y = torch.tensor(Y_train.values, dtype=torch.long)      # shape (N,)

class MLP():
    def __init__(self):
        self.W0 = (torch.randn(784,128) * 0.01).requires_grad_(True)
        self.B0 = torch.zeros(128,requires_grad=True)
        self.W1 = (torch.randn(128,64) * 0.01).requires_grad_(True)
        self.B1 = torch.zeros(64,requires_grad=True)
        self.W2 = (torch.randn(64,10) * 0.01).requires_grad_(True)
        self.B2 = torch.zeros(10,requires_grad=True)

    def forward(self, x): # x is (1,784)
        z0 = torch.matmul(x, self.W0) + self.B0
        a0 = torch.relu(z0)
        z1 = torch.matmul(a0, self.W1) + self.B1
        a1 = torch.relu(z1)
        z2 = torch.matmul(a1, self.W2) + self.B2
        return z2

model = MLP()
loss_fn = torch.nn.CrossEntropyLoss()
params = [model.W0, model.B0, model.W1, model.B1, model.W2, model.B2]
EPOCHS = 100
lr = 0.1

for epoch in range(EPOCHS):
    for start in range(0, len(X), 64):        # walk through data in chunks of 64
        xb = X[start : start+64]              # a minibatch of inputs
        yb = Y[start : start+64]              # its labels
        logits = model.forward(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
                p.grad.zero_()
    print(loss.item())