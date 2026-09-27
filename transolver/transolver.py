"""
Standalone Transolver for the irregular-mesh Elasticity benchmark.

This implementation follows the official Transolver ICML 2024 standard-benchmark
code, but uses only PyTorch (no timm/einops dependency).

Input for Elasticity:
    x: [B, N, 2]  -> XY coordinates
Output:
    y: [B, N, 1]  -> normalized scalar stress prediction
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
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "relu": nn.ReLU,
            "softplus": nn.Softplus,
            "elu": nn.ELU,
            "silu": nn.SiLU,
        }
        if act not in activations:
            raise ValueError(f"Unsupported activation: {act}")

        Act = activations[act]
        self.n_layers = n_layers
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)

        for layer in self.linears:
            if self.res:
                x = x + layer(x)
            else:
                x = layer(x)

        return self.linear_post(x)


class PhysicsAttentionIrregularMesh(nn.Module):
    """
    Physics-Attention:
        point features
          -> slice
          -> attention between slice tokens
          -> de-slice
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
    ):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Learned per-head slicing temperature.
        self.temperature = nn.Parameter(
            torch.ones(1, heads, 1, 1) * 0.5
        )

        # x_mid: determines slice membership.
        self.in_project_x = nn.Linear(dim, inner_dim)

        # fx_mid: content aggregated into each slice.
        self.in_project_fx = nn.Linear(dim, inner_dim)

        # M slice logits per point/head.
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # Attention on slice tokens.
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, _ = x.shape

        # --------------------------------------------------------------
        # 1. Slice
        # --------------------------------------------------------------
        fx_mid = self.in_project_fx(x)
        fx_mid = fx_mid.reshape(B, N, self.heads, self.dim_head)
        fx_mid = fx_mid.permute(0, 2, 1, 3).contiguous()
        # [B, H, N, D]

        x_mid = self.in_project_x(x)
        x_mid = x_mid.reshape(B, N, self.heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()
        # [B, H, N, D]

        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_weights = self.softmax(slice_logits)
        # [B, H, N, M], softmax over M

        slice_norm = slice_weights.sum(dim=2)
        # [B, H, M]

        slice_token = torch.einsum(
            "bhnd,bhnm->bhmd",
            fx_mid,
            slice_weights,
        )
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)
        # [B, H, M, D]

        # --------------------------------------------------------------
        # 2. Self-attention among slice tokens
        # --------------------------------------------------------------
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        out_slice_token = torch.matmul(attn, v)
        # [B, H, M, D]

        # --------------------------------------------------------------
        # 3. De-slice
        # --------------------------------------------------------------
        out = torch.einsum(
            "bhmd,bhnm->bhnd",
            out_slice_token,
            slice_weights,
        )
        # [B, H, N, D]

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, N, self.heads * self.dim_head)
        # [B, N, H*D]

        return self.to_out(out)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 1,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 64,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.last_layer = last_layer

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

        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        if self.last_layer:
            return self.output(self.ln_3(x))

        return x


class Transolver(nn.Module):
    """
    Irregular-mesh Transolver.

    For Elasticity:
        space_dim = 2
        fun_dim   = 0
        out_dim   = 1

    If fx is provided for another problem, it is concatenated with x along
    the last dimension before preprocessing.
    """

    def __init__(
        self,
        space_dim: int = 2,
        fun_dim: int = 0,
        out_dim: int = 1,
        n_layers: int = 8,
        n_hidden: int = 128,
        n_heads: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        act: str = "gelu",
    ):
        super().__init__()

        if n_hidden % n_heads != 0:
            raise ValueError("n_hidden must be divisible by n_heads.")

        self.space_dim = space_dim
        self.fun_dim = fun_dim
        self.n_hidden = n_hidden

        # Official implementation:
        # MLP(fun_dim + space_dim, 2*n_hidden, n_hidden, n_layers=0)
        self.preprocess = MLP(
            n_input=space_dim + fun_dim,
            n_hidden=2 * n_hidden,
            n_output=n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )

        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    num_heads=n_heads,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    last_layer=(i == n_layers - 1),
                    out_dim=out_dim,
                    slice_num=slice_num,
                )
                for i in range(n_layers)
            ]
        )

        # Learnable feature offset used by the official model.
        self.placeholder = nn.Parameter(
            (1.0 / n_hidden) * torch.rand(n_hidden)
        )

        # Match the official model's final initialization pass.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        fx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:  [B, N, space_dim]
            fx: optional [B, N, fun_dim]

        Returns:
            [B, N, out_dim]
        """
        if fx is not None:
            x = torch.cat((x, fx), dim=-1)

        x = self.preprocess(x)
        x = x + self.placeholder[None, None, :]

        for block in self.blocks:
            x = block(x)

        return x


# Official Elasticity reproduction settings from Transolver_Elas.sh
# plus defaults from exp_elas.py that the script does not override.
TRANSOLVER_ELASTICITY_HPARAMS = {
    # Model
    "space_dim": 2,
    "fun_dim": 0,
    "out_dim": 1,
    "n_hidden": 128,
    "n_layers": 8,
    "n_heads": 8,
    "slice_num": 64,
    "mlp_ratio": 1,
    "dropout": 0.0,

    # Training
    "batch_size": 1,
    "epochs": 500,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "max_grad_norm": 0.1,

    # Data split
    "n_train": 1000,
    "n_test": 200,
}
