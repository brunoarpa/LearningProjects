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


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LSTM():
    def __init__(self):
        self.input_size = vocab_size
        self.hidden_size = 128
        self.output_size = Y.shape[1]
        self.lr = 0.001

        self.w_y = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b_y = np.zeros((self.output_size, 1))
        self.w_f = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.u_f = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.b_f = np.ones((self.hidden_size, 1))
        self.w_i = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.u_i = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.b_i = np.zeros((self.hidden_size, 1))
        self.w_o = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.u_o = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.b_o = np.zeros((self.hidden_size, 1))
        self.w_cc = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.u_cc = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.b_cc = np.zeros((self.hidden_size, 1))

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t_adam = 0
        self.params = {
            'w_y': self.w_y, 'b_y': self.b_y, 'w_f': self.w_f, 'u_f': self.u_f, 'b_f': self.b_f,
            'w_i': self.w_i, 'u_i': self.u_i, 'b_i': self.b_i, 'w_o': self.w_o, 'u_o': self.u_o,
            'b_o': self.b_o, 'w_cc': self.w_cc, 'u_cc': self.u_cc, 'b_cc': self.b_cc
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, idx):
        self.h = {}
        self.c = {}
        self.o = {}
        self.f = {}
        self.i = {}
        self.cc = {}
        self.h[-1] = np.zeros((self.hidden_size, 1))
        self.c[-1] = np.zeros((self.hidden_size, 1))
        for t, id in enumerate(idx):
            self.f[t] = sigmoid((self.w_f @ self.h[t-1]) + (self.u_f[:, id:id+1]) + self.b_f)
            self.i[t] = sigmoid((self.w_i @ self.h[t-1]) + (self.u_i[:, id:id+1]) + self.b_i)
            self.o[t] = sigmoid((self.w_o @ self.h[t-1]) + (self.u_o[:, id:id+1]) + self.b_o)
            self.cc[t] = np.tanh((self.w_cc @ self.h[t-1]) + (self.u_cc[:, id:id+1]) + self.b_cc)
            self.c[t] = (self.f[t] * self.c[t-1]) + (self.i[t] * self.cc[t])
            self.h[t] = self.o[t] * np.tanh(self.c[t])

        T = len(idx) - 1
        y_hat = softmax((self.w_y @ self.h[T]) + self.b_y)
        return y_hat
    
    def backprop(self, y, o, idx):
        dy = o - y
        dz = dy 
        dw_y = dz @ self.h[len(idx)-1].T
        db_y = dz
        dht = (self.w_y.T @ dz)

        dbo = np.zeros_like(self.b_o)
        dwo = np.zeros_like(self.w_o)
        duo = np.zeros_like(self.u_o)
        dbcc = np.zeros_like(self.b_cc)
        dwcc = np.zeros_like(self.w_cc)
        ducc = np.zeros_like(self.u_cc)
        dbf = np.zeros_like(self.b_f)
        dwf = np.zeros_like(self.w_f)
        duf = np.zeros_like(self.u_f)
        dbi = np.zeros_like(self.b_i)
        dwi = np.zeros_like(self.w_i)
        dui = np.zeros_like(self.u_i)
        dctnext = np.zeros_like(self.c[-1])

        for t in reversed(range(len(idx))):

            dot = dht * np.tanh(self.c[t]) 
            dzo = (dot * self.o[t] * (1 - self.o[t]))
            dbo += dzo
            dwo += dzo @ self.h[t-1].T
            duo[:, idx[t]] += dzo[:, 0]

            dct = (dht * self.o[t] * (1 - np.tanh(self.c[t])**2)) + dctnext

            dcct = dct * self.i[t]
            dzcc = dcct * (1 - (self.cc[t]**2))
            dbcc += dzcc
            dwcc += dzcc @ self.h[t-1].T
            ducc[:, idx[t]] += dzcc[:, 0]

            dft = dct * self.c[t-1]
            dzf = (dft * self.f[t] * (1- self.f[t]))
            dbf += dzf
            dwf += dzf @ self.h[t-1].T
            duf[:, idx[t]] += dzf[:, 0]

            dit = dct * self.cc[t]
            dzi = (dit * self.i[t] * (1- self.i[t]))
            dbi += dzi
            dwi += dzi @ self.h[t-1].T
            dui[:, idx[t]] += dzi[:, 0]

            dht = (self.w_f.T @ dzf) + (self.w_i.T @ dzi) + (self.w_o.T @ dzo) + (self.w_cc.T @ dzcc)
            dctnext= (dct * self.f[t])

        grads = {
            'w_y': dw_y, 'b_y': db_y, 'w_f': dwf, 'u_f': duf, 'b_f': dbf,
            'w_i': dwi, 'u_i': dui, 'b_i': dbi, 'w_o': dwo, 'u_o': duo,
            'b_o': dbo, 'w_cc': dwcc, 'u_cc': ducc, 'b_cc': dbcc
        }

        clip_value = 5.0
        for g in grads.values():
            np.clip(g, -clip_value, clip_value, out=g)

        self.t_adam += 1
        for p_name, param in self.params.items():
            grad = grads[p_name]
            
            self.m[p_name] = self.beta1 * self.m[p_name] + (1 - self.beta1) * grad
            self.v[p_name] = self.beta2 * self.v[p_name] + (1 - self.beta2) * (grad**2)
            
            m_hat = self.m[p_name] / (1 - self.beta1**self.t_adam)
            v_hat = self.v[p_name] / (1 - self.beta2**self.t_adam)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

rnn = LSTM()
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    for i, idx in enumerate(encoded_texts):
        if len(idx) == 0:
            continue
        target = Y[i]
        y_hat = rnn.forward(idx)
        loss = -np.sum(target * np.log(y_hat + 1e-9))

        total_loss += loss

        if np.argmax(y_hat) == np.argmax(target):
            correct += 1

        rnn.backprop(target, y_hat, idx)

    acc = correct / len(encoded_texts)
    avg_loss = total_loss / len(encoded_texts)
    print(f"Epoch {epoch+1}/{epochs} -> Avg Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

text = ""
while text != "quit":
    text = input("Say something: ")
    encoded_text = encode_sentence(text)
    y_hat = rnn.forward(encoded_text)

    idx_to_label = {v: k for k, v in labels_to_idx.items()}
    pred_label = idx_to_label[np.argmax(y_hat)]

    print(f"Predicted sentiment: {pred_label} (Confidence: {np.max(y_hat)*100:.1f}%)")