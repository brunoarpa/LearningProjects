import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn

# a "transform" is a pipeline applied to each image as it's loaded
transform = T.Compose([
    T.ToTensor(),                          # PIL image -> tensor, scales pixels to [0,1]
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
val_set   = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)

# grab ONE batch and inspect it
xb, yb = next(iter(train_loader))
print("images:", xb.shape, xb.dtype)
print("labels:", yb.shape, yb.dtype)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*16*8,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        c1 = torch.relu(self.conv1(x))
        p1 = self.pool1(c1)
        f1 = torch.flatten(p1, start_dim=1)
        a1 = torch.relu(self.fc1(f1))
        a2 = torch.relu(self.fc2(a1))
        z3 = self.fc3(a2)
        return z3

LR = 1e-3
EPOCHS = 10
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    running = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running += loss.item()

    print(epoch, running/len(train_loader))

correct = 0
total = 0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)          # predicted class per image
        correct += (preds == yb).sum().item() # how many right in THIS batch
        total   += yb.size(0)                 # how many images in THIS batch
print("val accuracy:", correct / total)