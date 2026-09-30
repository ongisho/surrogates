import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        Act = activations[act]
        self.res = res

        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), Act())
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), Act())
            for _ in range(n_layers)
        ])
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = x + layer(x) if self.res else layer(x)
        return self.linear_post(x)


class PhysicsAttentionIrregularMesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_weights = self.softmax(slice_logits)                 # [B,H,N,M]

        slice_norm = slice_weights.sum(dim=2)                      # [B,H,M]
        slice_token = torch.einsum("bhnd,bhnm->bhmd", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(attn))
        out_slice_token = torch.matmul(attn, v)

        out = torch.einsum("bhmd,bhnm->bhnd", out_slice_token, slice_weights)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out)
    
class Transolver(nn.Module):
    def __init__(self, space_dim=2, fun_dim=0, out_dim=1, n_hidden=128, n_layers=8,
                 n_heads=8, slice_num=64, mlp_ratio=1, dropout=0.0):
        super().__init__()
        if n_hidden % n_heads != 0:
            raise ValueError("n_hidden must be divisible by n_heads")

        self.preprocess = MLP(
            n_input=space_dim + fun_dim,
            n_hidden=2 * n_hidden,
            n_output=n_hidden,
            n_layers=0,
            res=False,
        )

        self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)
        self.blocks = nn.ModuleList([
            TransolverBlock(
                hidden_dim=n_hidden,
                num_heads=n_heads,
                slice_num=slice_num,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                last_layer=(i == n_layers - 1),
                out_dim=out_dim,
            )
            for i in range(n_layers)
        ])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, fx=None):
        if fx is not None:
            x = torch.cat([x, fx], dim=-1)

        x = self.preprocess(x)
        x = x + self.placeholder[None, None, :]

        for block in self.blocks:
            x = block(x)

        return x
