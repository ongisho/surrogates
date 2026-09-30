"""
TrTtransolver
=============

Tiny Recursive Transolver:
TRM-style recursive reasoning + Transolver Physics-Attention.

x : fixed encoded PDE/problem state        [B, N, C]
y : evolving solution state                [B, N, C]
z : evolving latent reasoning state        [B, N, C]

Shared recursion:
    z = F_theta(x + y + z)   repeated L_cycles times
    y = F_theta(y + z)

The full cycle is repeated H_cycles times.
Early cycles may run under no_grad(), with gradients only through the final cycle.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ):
        super().__init__()

        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }

        if act not in activations:
            raise ValueError(f"Unsupported activation: {act}")

        Act = activations[act]
        self.res = res

        self.linear_pre = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            Act(),
        )

        self.linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    Act(),
                )
                for _ in range(n_layers)
            ]
        )

        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)

        for layer in self.linears:
            x = x + layer(x) if self.res else layer(x)

        return self.linear_post(x)


class PhysicsAttentionIrregularMesh(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 16,
        dropout: float = 0.0,
        slice_num: int = 64,
    ):
        super().__init__()

        inner_dim = heads * dim_head

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.temperature = nn.Parameter(
            torch.ones(1, heads, 1, 1) * 0.5
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, _ = x.shape

        x_mid = self.in_project_x(x)
        fx_mid = self.in_project_fx(x)

        x_mid = x_mid.reshape(
            B, N, self.heads, self.dim_head
        ).permute(0, 2, 1, 3).contiguous()

        fx_mid = fx_mid.reshape(
            B, N, self.heads, self.dim_head
        ).permute(0, 2, 1, 3).contiguous()

        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_weights = self.softmax(slice_logits)

        slice_norm = slice_weights.sum(dim=2)

        slice_token = torch.einsum(
            "bhnd,bhnm->bhmd",
            fx_mid,
            slice_weights,
        )

        slice_token = slice_token / (
            slice_norm.unsqueeze(-1) + 1e-5
        )

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        attn = torch.matmul(
            q,
            k.transpose(-1, -2),
        ) * self.scale

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        out_slice = torch.matmul(attn, v)

        out = torch.einsum(
            "bhmd,bhnm->bhnd",
            out_slice,
            slice_weights,
        )

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(
            B,
            N,
            self.heads * self.dim_head,
        )

        return self.to_out(out)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        slice_num: int,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        act: str = "gelu",
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.attn = PhysicsAttentionIrregularMesh(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.mlp = MLP(
            n_input=hidden_dim,
            n_hidden=hidden_dim * mlp_ratio,
            n_output=hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyTransolverCore(nn.Module):
    """
    Shared recursive core F_theta:
        [B,N,C] -> [B,N,C]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        act: str = "gelu",
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    hidden_dim=hidden_dim,
                    num_heads=n_heads,
                    slice_num=slice_num,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    act=act,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TrTtransolver(nn.Module):
    """
    Tiny Recursive Transolver.

    condition:
        [B,N,condition_dim]

    optional initial_solution:
        [B,N,solution_init_dim]

    hidden states:
        x,y,z : [B,N,hidden_dim]

    output:
        prediction : [B,N,out_dim]
    """

    def __init__(
        self,
        condition_dim: int,
        out_dim: int,
        solution_init_dim: int | None = None,
        hidden_dim: int = 128,
        core_layers: int = 2,
        n_heads: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        act: str = "gelu",
        L_cycles: int = 4,
        H_cycles: int = 3,
    ):
        super().__init__()

        self.condition_dim = condition_dim
        self.out_dim = out_dim
        self.solution_init_dim = solution_init_dim

        self.hidden_dim = hidden_dim
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles

        self.condition_encoder = MLP(
            n_input=condition_dim,
            n_hidden=2 * hidden_dim,
            n_output=hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )

        if solution_init_dim is not None:
            self.solution_encoder = MLP(
                n_input=solution_init_dim,
                n_hidden=2 * hidden_dim,
                n_output=hidden_dim,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.solution_encoder = None

        self.y_init = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )

        self.z_init = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )

        self.core = TinyTransolverCore(
            hidden_dim=hidden_dim,
            n_layers=core_layers,
            n_heads=n_heads,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.solution_decoder = nn.Linear(hidden_dim, out_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def encode_condition(self, condition):
        return self.condition_encoder(condition)

    def initialize_states(
        self,
        x,
        initial_solution=None,
    ):
        B, N, _ = x.shape

        if initial_solution is not None:
            if self.solution_encoder is None:
                raise ValueError(
                    "initial_solution was supplied, but "
                    "solution_init_dim=None."
                )

            y = self.solution_encoder(initial_solution)
        else:
            y = self.y_init.expand(B, N, -1)

        z = self.z_init.expand(B, N, -1)

        return y, z

    def reasoning_cycle(
        self,
        x,
        y,
        z,
    ):
        for _ in range(self.L_cycles):
            z = self.core(x + y + z)

        y = self.core(y + z)

        return y, z

    def forward_from_encoded(
        self,
        x,
        y,
        z,
        truncate_early_cycles: bool = True,
    ):
        if self.H_cycles < 1:
            raise ValueError("H_cycles must be >= 1.")

        if truncate_early_cycles:
            with torch.no_grad():
                for _ in range(self.H_cycles - 1):
                    y, z = self.reasoning_cycle(x, y, z)

            y, z = self.reasoning_cycle(x, y, z)

        else:
            for _ in range(self.H_cycles):
                y, z = self.reasoning_cycle(x, y, z)

        prediction = self.solution_decoder(
            self.output_norm(y)
        )

        return prediction, y, z

    def forward(
        self,
        condition,
        initial_solution=None,
        y=None,
        z=None,
        truncate_early_cycles: bool = True,
    ):
        x = self.encode_condition(condition)

        if (y is None) != (z is None):
            raise ValueError(
                "Either provide both y and z, or neither."
            )

        if y is None:
            y, z = self.initialize_states(
                x,
                initial_solution=initial_solution,
            )

        return self.forward_from_encoded(
            x=x,
            y=y,
            z=z,
            truncate_early_cycles=truncate_early_cycles,
        )

