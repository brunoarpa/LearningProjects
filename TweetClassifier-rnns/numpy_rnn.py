import numpy as np
import pandas as pd
from collections import Counter
train = pd.read_csv("train.csv")
texts = train["text"].astype(str).tolist()
labels = train["sentiment"].astype(str).tolist()

labels_to_idx = {"negative": 0, "neutral": 1, "positive": 2}

label_to_vec = {
    "negative": np.array([[1,0,0]]).T, 
    "neutral": np.array([[0,1,0]]).T, 
    "positive": np.array([[0,0,1]]).T 
}

y = np.array([labels_to_idx[label] for label in labels])
Y = np.array([label_to_vec[label] for label in labels])

all_words = []
for text in texts:
    all_words.extend(text.lower().split())

vocab_size = 2000
counts = Counter(all_words)
vocab = {word: id for id, (word, count) in enumerate(counts.most_common(vocab_size))}

def encode_sentence(sentence):
    words = sentence.lower().split()
    return [vocab[w] for w in words if w in vocab]

encoded_texts = [encode_sentence(t) for t in texts]

# def one_hot(idx):
#     v = np.zeros((vocab_size, 1))
#     v[idx] = 1
#     return v

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0, keepdims=True)

class RNN():
    def __init__(self):
        self.input_size = vocab_size
        self.hidden_size = 128
        self.output_size = Y.shape[1]
        self.lr = 0.005

        self.w_o = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b_o = np.zeros((self.output_size, 1))
        self.w_x = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.w_h = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.b_h = np.zeros((self.hidden_size, 1))

    def forward(self, idx):
        self.h = {}
        self.h[-1] = np.zeros((self.hidden_size, 1))
        for t, id in enumerate(idx):
            # x_t = one_hot(id)
            # h_t = np.tanh((self.w_x @ x_t) + (self.w_h @ self.h[t-1]) + self.b_h)
            h_t = np.tanh(self.w_x[:, id:id+1] + (self.w_h @ self.h[t-1]) + self.b_h)
            self.h[t] = h_t
        T = len(idx) - 1
        o = softmax((self.w_o @ self.h[T]) + self.b_o) 
        return o
    
    def backprop(self, y, o, idx):
        do = o - y
        #J = softmax_grad(o)
        dz = do #J @ do #o-y also works
        dbo = dz
        dwo = dz @ self.h[len(idx)-1].T
        dht = (self.w_o.T @ dz)

        dbh = np.zeros_like(self.b_h)
        dwh = np.zeros_like(self.w_h)
        dwx = np.zeros_like(self.w_x)
        for t in reversed(range(len(idx))):
            dtanh = (1-(self.h[t]**2)) * dht
            dbh += dtanh
            dwh += dtanh @ self.h[t-1].T
            # dwx += dtanh @ one_hot(idx[t]).T
            dwx[:, idx[t]] += dtanh[:, 0] 
            dht = self.w_h.T @ dtanh

        self.w_o -= self.lr * dwo
        self.b_o -= self.lr * dbo
        self.w_x -= self.lr * dwx
        self.w_h -= self.lr * dwh
        self.b_h -= self.lr * dbh

rnn = RNN()

epochs = 14

for epoch in range(epochs):
    total_loss = 0
    correct = 0

    for i, idx in enumerate(encoded_texts):
        if len(idx) == 0:
            continue
        target = Y[i]
        y_hat = rnn.forward(idx)
        loss = -np.sum(target * np.log(y_hat + 1e-8))

        total_loss += loss

        if np.argmax(y_hat) == np.argmax(target):
            correct += 1

        rnn.backprop(target, y_hat, idx)

        # if (i + 1) % 10000 == 0:
        #     print(f"  sample {i+1}/{len(encoded_texts)}")

    acc = correct / len(encoded_texts)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.3f} | Accuracy: {acc*100:.2f}%")

text = ""
while text != "quit":
    text = input("Say something: ")
    encoded_text = encode_sentence(text)
    y_hat = rnn.forward(encoded_text)

    idx_to_label = {v: k for k, v in labels_to_idx.items()}
    pred_label = idx_to_label[np.argmax(y_hat)]

    print("Predicted sentiment:", pred_label)

