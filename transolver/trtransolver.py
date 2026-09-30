import torch
import torch.nn as nn
        super().__init__()
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
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False)

        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if self.last_layer:
            x = self.output(self.ln_3(x))
        return x


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
