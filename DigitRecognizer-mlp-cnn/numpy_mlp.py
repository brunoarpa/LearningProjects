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
    #z vector
    a = (np.exp(z)) / (np.sum(np.exp(z)))
    return a 

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


class MLP:

    def __init__(self):
        self.w0 = np.random.randn(784, 128) * 0.01
        self.w1 = np.random.randn(128, 64) * 0.01
        self.w2 = np.random.randn(64, 10) * 0.01

        self.b0 = np.zeros(128)
        self.b1 = np.zeros(64)
        self.b2 = np.zeros(10)


        self.z0 = []
        self.a0 = []
        self.z1 = []
        self.a1 = []
        self.z2 = []
        self.a2 = []

        self.lr = 0.001

    def forward(self, x): 
        #input image mnist np vector size 784
        self.z0 = np.matmul(x,self.w0) + self.b0
        self.a0 = ReLU(self.z0)
        self.z1 = np.matmul(self.a0,self.w1) + self.b1
        self.a1 = ReLU(self.z1)
        self.z2 = np.matmul(self.a1,self.w2) + self.b2
        self.a2 = softmax(self.z2)

    def backprop(self, x, y):

        da2 = self.a2 - y 

        J2 = softmax_grad(self.a2)
        
        dz2 = da2 @ J2

        dw2 = np.outer(self.a1, dz2)

        db2 = dz2

        da1 = dz2 @ ((self.w2).T)

        dz1 = da1 * ReLU_grad(self.z1)

        dw1 = np.outer(self.a0, dz1)

        db1 = dz1

        da0 = dz1 @ ((self.w1).T)

        dz0 = da0 * ReLU_grad(self.z0)

        dw0 = np.outer(x, dz0)

        db0 = dz0

        self.w0 = self.w0 - (self.lr * dw0)
        self.w1 = self.w1 - (self.lr * dw1)
        self.w2 = self.w2 - (self.lr * dw2)

        self.b0 = self.b0 - (self.lr * db0)
        self.b1 = self.b1 - (self.lr * db1)
        self.b2 = self.b2 - (self.lr * db2)



def train_mlp():
    
    mlp = MLP()
    correct = 0
    total = 0

    for i, row in X_train.iterrows():
        x = row.values
        mlp.forward(x)
        y = np.zeros(10) 
        y[Y_train[i]] = 1 #OHE the label number to 1 true
        loss = np.mean(MSE(mlp.a2, y))
        mlp.backprop(x, y)

        pred_label = np.argmax(mlp.a2)
        if pred_label == Y_train[i]:
            correct += 1
        total += 1

        if (i+1) % 1000 == 0:
            accuracy = correct / total * 100
            print(f"Step {i+1}: Loss = {loss:.10f}, Accuracy = {accuracy:.2f}%")

train_mlp()
