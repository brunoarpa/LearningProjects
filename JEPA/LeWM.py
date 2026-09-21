import torch
import torch.nn as nn
import numpy as np

class SigReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t**2 / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, Z):
        A = torch.randn(Z.shape[-1], self.num_proj, device=Z.device)
        A = A / A.norm(dim=0)
        proj = Z @ A
        x_t = proj.unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(dim=0)
        sin_mean = x_t.sin().mean(dim=0)
        err = (cos_mean - self.phi)**2 + sin_mean**2
        statistic = (err @ self.weights) * Z.shape[0]
        return statistic.mean()

class ViT(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, num_patches):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, num_patches+1, d_model) * 0.02)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=14, stride=14)
        

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            activation="gelu", dropout=0.0, norm_first=True, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)  # stacks N of them

        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, o_t):                    
        B, C, H, W = o_t.shape
        x = self.conv1(o_t).flatten(2).transpose(1, 2)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos
        x = self.blocks(x)
        x = self.ln_f(x)
        return x[:, 0]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn_f = nn.BatchNorm1d(hidden_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, T, D = x.shape
        a = self.fc1(x.reshape(B * T, D))
        a = self.gelu(self.bn_f(a))
        a = self.fc2(a)
        return a.reshape(B, T, -1)

class ActionEncoder(nn.Module):
    def __init__(self, action_dim=2, d_model=192, mlp_scale=4):
        super().__init__()
        self.fc1 = nn.Linear(action_dim, 10)
        self.fc2 = nn.Linear(10, mlp_scale * d_model)
        self.act = nn.SiLU()
        self.fc3 = nn.Linear(mlp_scale * d_model, d_model)

    def forward(self, a):

        a = self.fc1(a)
        a = self.fc2(a)
        a = self.act(a)
        a = self.fc3(a)
        return a

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

class ARPRedictor(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, num_frames):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.pos = nn.Parameter(torch.randn(1, num_frames, d_model) * 0.02)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        self.attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.mlp = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model,num_frames)
        self.head_ln = nn.LayerNorm(d_model)
    def forward(self, z_t):           
        B, T, D = z_t.shape
        pos = torch.arange(T, device=z_t.device) 
        x = (z_t) + self.pos(pos) 
        x = self.blocks(x)
        logits = self.head(self.head_ln(z1))
        
        return logits

class ARPRedictor(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, num_frames):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.pos = nn.Parameter(torch.randn(1, num_frames, d_model) * 0.02)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        
        self.attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)




    def forward(self, z_t): 
        B, T, D = z_t.shape
        x = z_t + self.pos
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + sc1) + sh1)
        x = x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

