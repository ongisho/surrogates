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


TRTTRANSOLVER_ELASTICITY_HPARAMS = {
    "condition_dim": 2,
    "out_dim": 1,
    "solution_init_dim": None,

    "hidden_dim": 128,
    "core_layers": 2,
    "n_heads": 8,
    "slice_num": 64,
    "mlp_ratio": 1,
    "dropout": 0.0,

    "L_cycles": 4,
    "H_cycles": 3,
}
