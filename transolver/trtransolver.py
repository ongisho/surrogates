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
from typing import Optional, Tuple

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

        self.linear_post = nn.Linear(
            n_hidden,
            n_output,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)

        for layer in self.linears:
            x = x + layer(x) if self.res else layer(x)

        return self.linear_post(x)


class PhysicsAttentionIrregularMesh(nn.Module):
    """
    Physics-Attention used by Transolver.

    Input:
        x [B, N, C]

    Output:
        y [B, N, C]
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        slice_num: int,
        dropout: float,
    ):
        super().__init__()

        inner_dim = heads * dim_head

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Learned temperature for slice assignment.
        self.temperature = nn.Parameter(
            torch.ones(1, heads, 1, 1) * 0.5
        )

        # x_mid: routing / slice assignment representation.
        self.in_project_x = nn.Linear(
            dim,
            inner_dim,
        )

        # fx_mid: content representation aggregated into slices.
        self.in_project_fx = nn.Linear(
            dim,
            inner_dim,
        )

        # One logit for each slice.
        self.in_project_slice = nn.Linear(
            dim_head,
            slice_num,
        )

        # Attention on slice tokens.
        self.to_q = nn.Linear(
            dim_head,
            dim_head,
            bias=False,
        )
        self.to_k = nn.Linear(
            dim_head,
            dim_head,
            bias=False,
        )
        self.to_v = nn.Linear(
            dim_head,
            dim_head,
            bias=False,
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        # ----------------------------------------------------
        # Point projections
        # ----------------------------------------------------

        x_mid = self.in_project_x(x)
        fx_mid = self.in_project_fx(x)

        x_mid = (
            x_mid
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        fx_mid = (
            fx_mid
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # [B, H, N, D]

        # ----------------------------------------------------
        # Slice assignment
        # ----------------------------------------------------

        slice_logits = (
            self.in_project_slice(x_mid)
            / self.temperature
        )

        slice_weights = self.softmax(slice_logits)
        # [B, H, N, M], softmax over M

        # ----------------------------------------------------
        # Slice
        # ----------------------------------------------------

        slice_norm = slice_weights.sum(dim=2)
        # [B, H, M]

        slice_tokens = torch.einsum(
            "bhnd,bhnm->bhmd",
            fx_mid,
            slice_weights,
        )

        slice_tokens = (
            slice_tokens
            / (slice_norm.unsqueeze(-1) + 1e-5)
        )

        # [B, H, M, D]

        # ----------------------------------------------------
        # Attention between slice tokens
        # ----------------------------------------------------

        q = self.to_q(slice_tokens)
        k = self.to_k(slice_tokens)
        v = self.to_v(slice_tokens)

        attn = (
            torch.matmul(
                q,
                k.transpose(-1, -2),
            )
            * self.scale
        )

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        out_slice_tokens = torch.matmul(
            attn,
            v,
        )

        # ----------------------------------------------------
        # De-slice
        # ----------------------------------------------------

        out = torch.einsum(
            "bhmd,bhnm->bhnd",
            out_slice_tokens,
            slice_weights,
        )

        out = (
            out
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(
                B,
                N,
                self.heads * self.dim_head,
            )
        )

        return self.to_out(out)


class TransolverBlock(nn.Module):
    """
    Standard pre-LN Transolver block:

        h = x + PhysicsAttention(LN(x))
        y = h + FFN(LN(h))
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float,
        act: str,
    ):
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by n_heads."
            )

        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.attn = PhysicsAttentionIrregularMesh(
            dim=hidden_dim,
            heads=n_heads,
            dim_head=hidden_dim // n_heads,
            slice_num=slice_num,
            dropout=dropout,
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.mlp = MLP(
            n_input=hidden_dim,
            n_hidden=hidden_dim * mlp_ratio,
            n_output=hidden_dim,
            n_layers=0,
            act=act,
            res=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x)
        )

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x


class TinyTransolverCore(nn.Module):
    """
    Tiny shared recursive core F_theta.

    The `core_layers` blocks have independent parameters from each other,
    but THE WHOLE CORE is reused on every recursive call.

    Example:
        core_layers = 2

        F_theta = Block1 -> Block2

    Every call to F_theta reuses those same Block1/Block2 parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        core_layers: int,
        n_heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float,
        act: str,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    slice_num=slice_num,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    act=act,
                )
                for _ in range(core_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        return x


# ============================================================
# TrTtransolver
# ============================================================

class TrTtransolver(nn.Module):
    """
    Tiny Recursive Transolver.

    Fixed problem state:
        x [B,N,C]

    Evolving solution state:
        y [B,N,C]

    Evolving latent reasoning state:
        z [B,N,C]

    One H-cycle:
        repeat L_cycles:
            z = F_theta(x + y + z)

        y = F_theta(y + z)

    H_cycles repeats this cycle.

    If truncate_early_cycles=True:
        first H_cycles-1 cycles -> no_grad()
        final H-cycle           -> gradients
    """

    def __init__(
        self,
        condition_dim: int,
        solution_init_dim: Optional[int],
        out_dim: int,
        hidden_dim: int,
        core_layers: int,
        n_heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float,
        act: str,
        L_cycles: int,
        H_cycles: int,
    ):
        super().__init__()

        if H_cycles < 1:
            raise ValueError("H_cycles must be >= 1.")

        if L_cycles < 1:
            raise ValueError("L_cycles must be >= 1.")

        self.condition_dim = condition_dim
        self.solution_init_dim = solution_init_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles

        # ----------------------------------------------------
        # Encode fixed PDE/problem conditioning.
        #
        # Elasticity:
        #   [B,972,2] -> [B,972,128]
        # ----------------------------------------------------

        self.condition_encoder = MLP(
            n_input=condition_dim,
            n_hidden=2 * hidden_dim,
            n_output=hidden_dim,
            n_layers=0,
            act=act,
            res=False,
        )

        # Optional encoder for a physical initial solution.
        if solution_init_dim is not None:
            self.solution_encoder = MLP(
                n_input=solution_init_dim,
                n_hidden=2 * hidden_dim,
                n_output=hidden_dim,
                n_layers=0,
                act=act,
                res=False,
            )
        else:
            self.solution_encoder = None

        # Learned initial y/z states.
        self.y_init = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )

        self.z_init = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )

        # One tiny core, reused recursively.
        self.core = TinyTransolverCore(
            hidden_dim=hidden_dim,
            core_layers=core_layers,
            n_heads=n_heads,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act,
        )

        # Decode current y state into physical output channels.
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.solution_decoder = nn.Linear(
            hidden_dim,
            out_dim,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0.0,
                )

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(
                module.bias,
                0.0,
            )

            nn.init.constant_(
                module.weight,
                1.0,
            )

    def encode_condition(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.condition_encoder(condition)

    def initialize_states(
        self,
        x: torch.Tensor,
        initial_solution: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape

        if initial_solution is not None:
            if self.solution_encoder is None:
                raise ValueError(
                    "initial_solution was supplied, but "
                    "solution_init_dim=None."
                )

            y = self.solution_encoder(
                initial_solution
            )
        else:
            y = self.y_init.expand(
                B,
                N,
                -1,
            )

        z = self.z_init.expand(
            B,
            N,
            -1,
        )

        return y, z

    def reasoning_cycle(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Latent/reasoning updates.
        for _ in range(self.L_cycles):
            z = self.core(
                x + y + z
            )

        # Solution-state update.
        y = self.core(
            y + z
        )

        return y, z

    def forward_from_encoded(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        truncate_early_cycles: bool,
    ):
        if truncate_early_cycles:
            # Early H-cycles: no autograd graph.
            with torch.no_grad():
                for _ in range(
                    self.H_cycles - 1
                ):
                    y, z = self.reasoning_cycle(
                        x,
                        y,
                        z,
                    )

            # Final H-cycle: gradients enabled.
            y, z = self.reasoning_cycle(
                x,
                y,
                z,
            )

        else:
            # Full backprop through all H-cycles.
            for _ in range(self.H_cycles):
                y, z = self.reasoning_cycle(
                    x,
                    y,
                    z,
                )

        prediction = self.solution_decoder(
            self.output_norm(y)
        )

        return prediction, y, z

    def forward(
        self,
        condition: torch.Tensor,
        initial_solution: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        truncate_early_cycles: bool = True,
    ):
        """
        condition:
            [B,N,condition_dim]

        initial_solution:
            optional [B,N,solution_init_dim]

        y,z:
            optional carried hidden states from previous
            deep-supervision step.

        returns:
            prediction [B,N,out_dim]
            y          [B,N,C]
            z          [B,N,C]
        """
        x = self.encode_condition(
            condition
        )

        if (y is None) != (z is None):
            raise ValueError(
                "Provide both y and z, or neither."
            )

        if y is None:
            y, z = self.initialize_states(
                x=x,
                initial_solution=initial_solution,
            )

        return self.forward_from_encoded(
            x=x,
            y=y,
            z=z,
            truncate_early_cycles=truncate_early_cycles,
        )
