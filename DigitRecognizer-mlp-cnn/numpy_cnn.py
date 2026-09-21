import random
import numpy as np
import pandas as pd


train = pd.read_csv("train.csv")

X_train = train.drop(columns="label")
Y_train = train["label"]


def ReLU(z):
    #z vector
    a = np.maximum(0,z)
    return a

def softmax(z):
    z = z - np.max(z)  
    return np.exp(z) / np.sum(np.exp(z))


def MSE(a, y):
    return 0.5*((a-y)**2)

def softmax_grad(a):
    J = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            if i == j:
                J[i,j] = a[i]*(1-a[i])
            else:
                J[i,j] = -a[i]*a[j]

    return J

def ReLU_grad(a):
    
    return (a > 0).astype(int)


class CNN:

    def __init__(self):
        self.f0 = np.random.randn(3, 3, 1, 8) * 0.01
        self.w1 = np.random.randn(64, 8*14*14) * 0.01
        self.w2 = np.random.randn(10, 64) * 0.01

        self.b0 = np.zeros((1, 1, 8))
        self.b1 = np.zeros((64, 1))
        self.b2 = np.zeros((10, 1))

        self.z0 = []
        self.a0 = []
        self.z1 = []
        self.a1 = []
        self.z2 = []
        self.a2 = []

        self.lr = 0.001

    def forward(self, image): 
        self.z0 = np.zeros((28, 28, 8))
        padded = np.pad(image, ((1,1),(1,1),(0,0)), 'constant')
        for k in range(self.f0.shape[3]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    self.z0[i,j, k] = np.sum(padded[i:i+3, j:j+3, :]*self.f0[:,:,:,k])
        self.z0 += self.b0 
        self.a0 = ReLU(self.z0)
        H, W, C = self.a0.shape
        self.p0 = np.zeros((H//2, W//2, C))

        for c in range(C):
            for i in range(H//2):
                for j in range(W//2):
                    patch = self.a0[2*i:2*i+2, 2*j:2*j+2, c]
                    self.p0[i, j, c] = np.max(patch)


        flat = self.p0.flatten().reshape(-1, 1)
        self.z1 = np.matmul(self.w1,flat) + self.b1
        self.a1 = ReLU(self.z1)
        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = softmax(self.z2)
        

    def backprop(self, image, y):

        dz2 = self.a2 - y.reshape(-1,1)
        dw2 = dz2 @ self.a1.T

        db2 = dz2
        da1 = self.w2.T @ dz2

        dz1 = da1 * ReLU_grad(self.z1)
        flat = self.p0.flatten().reshape(-1, 1)
        dw1 = dz1 @ flat.T

        db1 = dz1

        dflat = self.w1.T @ dz1
        
        dp0 = dflat.reshape(image.shape[0]//2, image.shape[0]//2, self.f0.shape[3])

        H_pool, W_pool, C = dp0.shape
        da0 = np.zeros_like(self.a0)

        for c in range(C):
            for i in range(H_pool):
                for j in range(W_pool):
                    patch = self.a0[2*i:2*i+2, 2*j:2*j+2, c]
                    mask = (patch == np.max(patch)) 
                    da0[2*i:2*i+2, 2*j:2*j+2, c] = mask * dp0[i,j,c]

        dz0 = da0 * ReLU_grad(self.z0)
        db0 = np.sum(dz0, axis=(0,1), keepdims=True)

        df0 = np.zeros_like(self.f0)
        padded = np.pad(image, ((1,1),(1,1),(0,0)), 'constant')
        for k in range(self.f0.shape[3]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    df0[:, :, :, k] += padded[i:i+3, j:j+3, :] * dz0[i, j, k]

        self.f0 -= self.lr * df0
        self.w1 -= self.lr * dw1
        self.w2 -= self.lr * dw2

        self.b0 -= self.lr * db0
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

def train_mlp():
    
    cnn = CNN()
    correct = 0
    total = 0

    for i, row in X_train.iterrows():
        x = row.values 
        image = x.reshape(28, 28, 1)
        cnn.forward(image)
        y = np.zeros(10) 
        y[Y_train[i]] = 1 #OHE the label number to 1 true
        loss = np.mean(MSE(cnn.a2, y))
        cnn.backprop(image, y)

        pred_label = np.argmax(cnn.a2)
        if pred_label == Y_train[i]:
            correct += 1
        total += 1

        if (i+1) % 1000 == 0:
            accuracy = correct / total * 100
            print(f"Step {i+1}: Loss = {loss:.10f}, Accuracy = {accuracy:.2f}%")

train_mlp()
