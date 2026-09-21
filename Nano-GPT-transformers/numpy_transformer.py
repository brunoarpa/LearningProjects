import os
import pickle
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 64
BATCH_SIZE = 128
EPOCHS = 100

with open("bnc_spoken.train", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tokenizer.train_from_iterator([text], trainer)
data = tokenizer.encode(text).ids
data = np.array(data, dtype=np.int32)
vocab = list(tokenizer.get_vocab().keys())  

stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for i,ch in enumerate(vocab)}

def get_batch(split, batch_size, context_length):
    
    dataset = train_data if split == 'train' else val_data

    ix = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))
    x = np.stack([dataset[i:i+context_length] for i in ix])
    y = np.stack([dataset[i+1:i+context_length+1] for i in ix])
    
    return x, y

def ReLU(z):
    return np.maximum(0,z)

def softmax(z, axis):
    e_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e_z / e_z.sum(axis=axis, keepdims=True) 

def softmax_backward(grad_output, softmax_output):
    return softmax_output * (grad_output - np.sum(grad_output * softmax_output, axis=-1, keepdims=True))

def LayerNorm(X, gamma, beta):
    mu = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    std = np.sqrt(var + 1e-6)
    X_hat = (X - mu)/(std) * gamma + beta
    return X_hat

def LayerNormBackward(dY, X, gamma, eps=1e-6):
    mu = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    X_hat = (X - mu) / std
    N = X.shape[-1]
    
    dX_hat = dY * gamma
    dvar = np.sum(dX_hat * (X - mu) * -0.5 * std**(-3), axis=-1, keepdims=True)
    dmu = np.sum(-dX_hat / std, axis=-1, keepdims=True) + dvar * np.mean(-2*(X - mu), axis=-1, keepdims=True)
    dX = dX_hat / std + dvar * 2*(X - mu)/N + dmu / N
    dgamma = np.sum(dY * X_hat, axis=(0,1))
    dbeta = np.sum(dY, axis=(0,1))
    return dX, dgamma, dbeta

def cross_entropy(pred, Y):
    N, T, V = pred.shape 
    log_preds = np.log(pred + 1e-9)
    one_hot = np.eye(V)[Y]  
    return -np.sum(one_hot * log_preds) / (N * T)

class Transformer():
    def __init__(self):

        self.lr = 1e-3 

        self.dmodel = 128
        self.nheads = 4
        self.dheads = self.dmodel // self.nheads
        self.layers = 2
        self.dff = 512

        self.embeddings = np.random.randn(VOCAB_SIZE, self.dmodel) * 0.01
        self.pe = np.random.randn(CONTEXT_LENGTH, self.dmodel) * 0.01

        self.beta_out = np.random.randn(self.dmodel) * 0.01
        self.gamma_out = np.random.randn( self.dmodel) * 0.01
        self.w_out = np.random.randn(self.dmodel, VOCAB_SIZE) * 0.01
        self.b_out = np.random.randn(VOCAB_SIZE) * 0.01 

        self.params = {}

        for l in range(self.layers):
            self.params[l] = { 
                'W_Q': np.random.randn(self.dmodel, self.dmodel) * 0.01,
                'W_K': np.random.randn(self.dmodel, self.dmodel) * 0.01,
                'W_V': np.random.randn(self.dmodel, self.dmodel) * 0.01,
                'W_O': np.random.randn(self.dmodel, self.dmodel) * 0.01,
                'B_O': np.random.randn(self.dmodel) * 0.01,
                'W_1': np.random.randn(self.dmodel, self.dff) * 0.01,
                'B_1': np.random.randn(self.dff) * 0.01,
                'W_2': np.random.randn(self.dff, self.dmodel) * 0.01,
                'B_2': np.random.randn(self.dmodel) * 0.01,
                'Gamma1': np.random.randn(self.dmodel) * 0.01,
                'Beta1': np.random.randn(self.dmodel) * 0.01,
                'Gamma2': np.random.randn(self.dmodel) * 0.01,
                'Beta2': np.random.randn(self.dmodel) * 0.01,
            }

        self.beta1adam = 0.9
        self.beta2adam = 0.999
        self.epsilonadam = 1e-8
        self.t_adam = 0

        self.all_params = {
            'embeddings': self.embeddings,
            'pe': self.pe,
            'beta_out': self.beta_out,
            'gamma_out': self.gamma_out,
            'w_out': self.w_out,
            'b_out': self.b_out,
        }

        for l in range(self.layers):
            for k, v in self.params[l].items():
                self.all_params[f'{l}_{k}'] = v

        self.m = {k: np.zeros_like(v) for k, v in self.all_params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.all_params.items()}

    def save_checkpoint(self, filename="checkpoint.pkl"):
        checkpoint = {
            'all_params': self.all_params,
            'm': self.m,
            'v': self.v,
            't_adam': self.t_adam
        }
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"Saved checkpoint to {filename}")

    def load_checkpoint(self, filename="checkpoint.pkl"):
        with open(filename, "rb") as f:
            checkpoint = pickle.load(f)
        self.all_params = checkpoint['all_params']
        self.m = checkpoint['m']
        self.v = checkpoint['v']
        self.t_adam = checkpoint['t_adam']

        for key, val in self.all_params.items():
            if key in ['embeddings', 'pe', 'beta_out', 'gamma_out', 'w_out', 'b_out']:
                setattr(self, key, val)
            else:
                layer, pname = key.split('_', 1)
                self.params[int(layer)][pname] = val

        print(f"Loaded checkpoint from {filename}")

    def forward_backprop(self, tokens, batch_size, Y, inference_only):
        B, T = tokens.shape
        X = {}
        X_hat = {}
        Q = {}
        K = {}
        V = {}
        A = {}
        S = {}
        H = {}
        O_hat = {}
        O = {}
        F = {}
        Z = {}
        
        X[0] = self.embeddings[tokens] + self.pe[:T, :]

        for l in range(self.layers):

            X_hat[l] = LayerNorm(X[l], self.params[l]['Gamma1'], self.params[l]['Beta1'])

            Q[l] = X_hat[l] @ self.params[l]['W_Q']
            K[l] = X_hat[l] @ self.params[l]['W_K']
            V[l] = X_hat[l] @ self.params[l]['W_V']

            Q[l] = np.transpose((Q[l].reshape(B, T, self.nheads, self.dheads)), (0,2,1,3))
            K[l] = np.transpose((K[l].reshape(B, T, self.nheads, self.dheads)), (0,2,1,3))
            V[l] = np.transpose((V[l].reshape(B, T, self.nheads, self.dheads)), (0,2,1,3))

            A[l] = (Q[l] @ np.transpose(K[l], (0, 1, 3, 2)))/(np.sqrt(self.dheads))
            mask = np.triu(np.ones((T, T)), k=1).astype(bool)
            A[l] = np.where(mask[None, None, :, :], -1e9, A[l])
            S[l] = softmax(A[l], 3)
            H[l] = S[l] @ V[l]

            H[l] = np.transpose(H[l], (0, 2, 1, 3)).reshape(B, T, self.nheads * self.dheads)
            O[l] = (H[l] @ self.params[l]['W_O']) + self.params[l]['B_O'] + X[l]

            O_hat[l] = LayerNorm(O[l], self.params[l]['Gamma2'], self.params[l]['Beta2'])

            F[l] = (O_hat[l] @ self.params[l]['W_1']) + self.params[l]['B_1']
            Z[l] = ((ReLU(F[l])) @ self.params[l]['W_2']) + self.params[l]['B_2']
            X[l+1] = O[l] + Z[l]

        X_hat[self.layers] = LayerNorm(X[self.layers], self.gamma_out, self.beta_out)
        logits = (X_hat[self.layers] @ self.w_out) + self.b_out
        pred = softmax(logits, 2)

        if inference_only:
            return pred

        dL_dlogits = (pred - np.eye(VOCAB_SIZE)[Y]) / (B * T)
        dL_db_out = np.sum(dL_dlogits, axis=(0,1))
        dL_w_out = X_hat[self.layers].reshape(-1, self.dmodel).T @ dL_dlogits.reshape(-1, VOCAB_SIZE)

        dL_B2 = {}
        dL_W2 = {}
        dL_B1 = {}
        dL_W1 = {}
        dL_Gamma2 = {}
        dL_Beta2 = {}
        dL_BO = {}
        dL_WO = {}
        dL_W_Q = {}
        dL_W_K = {}
        dL_W_V = {}
        dL_Gamma1 = {}
        dL_Beta1 = {}

        dL_X_hat = dL_dlogits @ self.w_out.T
        dL_X, dL_gamma_out, dL_beta_out =  LayerNormBackward(dL_X_hat, X[self.layers], self.gamma_out)

        for l in reversed(range(self.layers)):
            dL_Z = dL_X
            dL_B2[l] = np.sum(dL_Z, axis=(0,1)) 
            dL_W2[l] = ReLU((F[l])).reshape(-1, self.dff).T @ dL_Z.reshape(-1, self.dmodel)
            dL_F = ((F[l] > 0).astype(int)) * (dL_Z @ self.params[l]['W_2'].T)
            dL_B1[l] = np.sum(dL_F, axis=(0,1))
            dL_W1[l] = O_hat[l].reshape(-1, self.dmodel).T @ dL_F.reshape(-1, self.dff)
            dL_O_hat = dL_F @ self.params[l]['W_1'].T
            dL_O, dL_Gamma2[l], dL_Beta2[l] = LayerNormBackward(dL_O_hat, O[l], self.params[l]['Gamma2'])
            dL_O += dL_X
            dL_BO[l] = np.sum(dL_O, axis=(0,1))
            dL_WO[l] = H[l].reshape(-1, self.dmodel).T @ dL_O.reshape(-1, self.dmodel)
            dL_H = dL_O @ self.params[l]['W_O'].T
            dL_H = np.transpose(dL_H.reshape(B, T, self.nheads, self.dheads), (0,2,1,3))
            dL_S = dL_H @ np.transpose(V[l], (0,1,3,2))
            dL_A = softmax_backward(dL_S, S[l])
            dL_A = np.where(mask[None, None, :, :], 0, dL_A)
            dL_Q = ((dL_A @ K[l])/(np.sqrt(self.dheads)))
            dL_K = ((np.transpose(dL_A, (0,1,3,2)) @ Q[l])/(np.sqrt(self.dheads)))
            dL_V = (np.transpose(S[l], (0,1,3,2)) @ dL_H)
            dL_Q = ((np.transpose(dL_Q, (0,2,1,3))).reshape(B, T, self.nheads * self.dheads))
            dL_K = ((np.transpose(dL_K, (0,2,1,3))).reshape(B, T, self.nheads * self.dheads))
            dL_V = ((np.transpose(dL_V, (0,2,1,3))).reshape(B, T, self.nheads * self.dheads))
            dL_W_Q[l] = X_hat[l].reshape(-1, self.dmodel).T @ dL_Q.reshape(-1, self.dmodel)
            dL_W_K[l] = X_hat[l].reshape(-1, self.dmodel).T @ dL_K.reshape(-1, self.dmodel)
            dL_W_V[l] = X_hat[l].reshape(-1, self.dmodel).T @ dL_V.reshape(-1, self.dmodel)
            dL_X_hat = (dL_Q @ self.params[l]['W_Q'].T) + (dL_K @ self.params[l]['W_K'].T) + (dL_V @ self.params[l]['W_V'].T)
            dL_X, dL_Gamma1[l], dL_Beta1[l] = LayerNormBackward(dL_X_hat, X[l], self.params[l]['Gamma1'])
            dL_X += dL_O

        dL_params = {}
        for l in range(self.layers):
            dL_params[l] = {
                'W_Q': dL_W_Q[l],
                'W_K': dL_W_K[l],
                'W_V': dL_W_V[l],
                'W_O': dL_WO[l],
                'B_O': dL_BO[l],
                'W_1': dL_W1[l],
                'B_1': dL_B1[l],
                'W_2': dL_W2[l],
                'B_2': dL_B2[l],
                'Gamma1': dL_Gamma1[l],
                'Beta1': dL_Beta1[l],
                'Gamma2': dL_Gamma2[l],
                'Beta2': dL_Beta2[l],
            }

        dL_embeddings = np.zeros_like(self.embeddings)
        dL_pe = np.zeros_like(self.pe)

        np.add.at(dL_embeddings, tokens, dL_X)  

        dL_pe = np.sum(dL_X, axis=0) 

        grads = {
            'embeddings': dL_embeddings, 
            'pe': dL_pe,
            'beta_out': dL_beta_out, 
            'gamma_out': dL_gamma_out,
            'w_out': dL_w_out, 
            'b_out': dL_db_out
        }


        for l in range(self.layers):
            for pname in ['W_Q', 'W_K', 'W_V', 'W_O', 'B_O', 'W_1', 'B_1', 'W_2', 'B_2', 'Gamma1', 'Beta1', 'Gamma2', 'Beta2']:
                grads[f"{l}_{pname}"] = dL_params[l][pname]


        clip_value = 1.0
        for g in grads.values():
            np.clip(g, -clip_value, clip_value, out=g)

        self.t_adam += 1
        for name, grad in grads.items():
            if name not in self.all_params:
                continue

            m = self.m[name]
            v = self.v[name]
            param = self.all_params[name]

            m = self.beta1adam * m + (1 - self.beta1adam) * grad
            v = self.beta2adam * v + (1 - self.beta2adam) * (grad ** 2)

            m_hat = m / (1 - self.beta1adam ** self.t_adam)
            v_hat = v / (1 - self.beta2adam ** self.t_adam)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilonadam)

            self.m[name] = m
            self.v[name] = v
            self.all_params[name] = param


val_start = int(len(data) * 0.9)
train_data = data[:val_start]
val_data = data[val_start:]

print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))

batches_per_epoch = len(train_data) // (BATCH_SIZE * CONTEXT_LENGTH)
batches_per_val = 200

nano_gpt = Transformer()

LOAD_CHECKPOINT = True 
CHECKPOINT_PATH = "nano_gpt_checkpoint.pkl"

if LOAD_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
    nano_gpt.load_checkpoint(CHECKPOINT_PATH)
else:
    print("Starting training...")


batches_per_epoch = len(train_data) // (BATCH_SIZE * CONTEXT_LENGTH) 
batches_per_val = 200 

# for epoch in range(EPOCHS):

#     for step in range(batches_per_epoch):

#         X_batch, Y_batch = get_batch('train', BATCH_SIZE, CONTEXT_LENGTH)

#         nano_gpt.forward_backprop(X_batch, batch_size=len(X_batch), Y=Y_batch, inference_only=False)

#         if step % batches_per_val == 0:
#             X_val_batch, Y_val_batch = get_batch('val', BATCH_SIZE, CONTEXT_LENGTH)
            
#             val_pred = nano_gpt.forward_backprop(X_val_batch, batch_size=len(X_val_batch), Y=None, inference_only=True)
#             val_loss = cross_entropy(val_pred, Y_val_batch)
#             print(f"Epoch {epoch}, step {step}, val_loss = {val_loss:.4f}")
#             if step % (batches_per_val * 5) == 0:
#                 nano_gpt.save_checkpoint(CHECKPOINT_PATH)

text = ""

while True:
    text = input("Say something: ")
    if text.lower() == "quit":
        break

    length = int(input("How many tokens to generate: "))

    encoded = tokenizer.encode(text).ids
    context = np.array(encoded, dtype=np.int32)[None, :]  

    for _ in range(length):

        context_cond = context[:, -CONTEXT_LENGTH:]

        preds = nano_gpt.forward_backprop(context_cond, batch_size=1, Y=None, inference_only=True)
        last_token_probs = preds[:, -1, :].flatten()
        next_token = np.random.choice(VOCAB_SIZE, p=last_token_probs)

        context = np.concatenate([context, [[next_token]]], axis=1)

    decoded = tokenizer.decode(context[0].tolist())
    print(f"\nResponse: {decoded}\n")

