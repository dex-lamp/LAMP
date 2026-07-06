"""Hand-action VAE network used by BCVAE policies.

This supplement copy intentionally contains only pure model definitions: no
checkpoint I/O, no replay buffer logic, and no environment transport code.

The VAE consumes a history window of absolute hand poses with shape
`(batch, window_size, action_dim)` and learns a latent representation whose
decoder predicts the next absolute hand pose with shape `(batch, action_dim)`.
BCVAE and residual/TD3 agents use the same encode/decode methods during
inference and actor optimization.
"""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax


Array = jax.Array


def _torch_uniform(key, shape, dtype, fan_in):
    """Return PyTorch-like uniform initialization for checkpoint compatibility."""
    bound = 1.0 / jnp.sqrt(float(fan_in))
    return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)


class TorchDense(nn.Module):
    """Dense layer whose parameter names and initialization match the source BC/VAE code."""

    features: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        in_features = x.shape[-1]
        kernel = self.param(
            "kernel",
            lambda key, shape: _torch_uniform(key, shape, x.dtype, in_features),
            (in_features, self.features),
        )
        bias = self.param(
            "bias",
            lambda key, shape: _torch_uniform(key, shape, x.dtype, in_features),
            (self.features,),
        )
        return jnp.matmul(x, kernel) + bias


class CausalConv1d(nn.Module):
    """One-dimensional causal convolution over hand-pose history windows."""

    out_channels: int
    kernel_size: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Left padding makes the convolution causal: output at time t cannot see
        # future hand poses. This matches the original VAE training convention.
        pad = self.kernel_size - 1
        x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
        in_channels = x.shape[-1]
        fan_in = self.kernel_size * in_channels
        kernel = self.param(
            "kernel",
            lambda key, shape: _torch_uniform(key, shape, x.dtype, fan_in),
            (self.kernel_size, in_channels, self.out_channels),
        )
        bias = self.param(
            "bias",
            lambda key, shape: _torch_uniform(key, shape, x.dtype, fan_in),
            (self.out_channels,),
        )
        y = lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return y + bias


class CausalConvEncoder(nn.Module):
    """Encode a hand-pose window with causal temporal convolutions."""

    action_dim: int = 6
    hidden_dim: int = 256

    def setup(self) -> None:
        self.layers = [
            CausalConv1d(64, kernel_size=3, name="layer_0"),
            CausalConv1d(128, kernel_size=3, name="layer_1"),
            CausalConv1d(self.hidden_dim, kernel_size=3, name="layer_2"),
        ]

    def __call__(self, x: Array) -> Array:
        h = x
        for layer in self.layers:
            h = nn.silu(layer(h))
        return h[:, -1, :]


class MLPEncoder(nn.Module):
    """Encode a hand-pose window by flattening it into an MLP."""

    action_dim: int = 6
    window_size: int = 8
    hidden_dim: int = 256
    num_hidden_layers: int = 1

    def setup(self) -> None:
        layer_dims = [self.hidden_dim] * (self.num_hidden_layers + 1)
        self.layers = [
            TorchDense(dim, name=f"layer_{idx}") for idx, dim in enumerate(layer_dims)
        ]

    def __call__(self, x: Array) -> Array:
        h = x.reshape((x.shape[0], self.action_dim * self.window_size))
        for layer in self.layers:
            h = nn.silu(layer(h))
        return h


class MLPDecoder(nn.Module):
    """Decode a latent hand code into an absolute hand pose."""

    latent_dim: int
    hidden_dim: int
    out_dim: int
    num_hidden_layers: int = 1

    def setup(self) -> None:
        layer_dims = [self.hidden_dim] * (self.num_hidden_layers + 1) + [self.out_dim]
        self.layers = [
            TorchDense(dim, name=f"layer_{idx}") for idx, dim in enumerate(layer_dims)
        ]

    def __call__(self, x: Array) -> Array:
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx < len(self.layers) - 1:
                h = nn.silu(h)
        return h


class HandActionVAE(nn.Module):
    """VAE over absolute hand actions.

    Inputs:
        x: Hand history, shape `(batch, window_size, action_dim)`.
        target: Next absolute hand pose, shape `(batch, action_dim)`.

    Outputs:
        `__call__` returns prediction and training losses.  Runtime policy code
        usually calls `encode`, `decode`, or `predict` directly so the VAE can
        serve as a frozen hand prior.
    """

    action_dim: int = 6
    window_size: int = 8
    hidden_dim: int = 256
    latent_dim: int = 32
    beta: float = 0.01
    encoder_type: str = "mlp"
    num_hidden_layers: int = 1
    recon_aux_weight: float = 0.0
    free_bits: float = 0.0

    def setup(self) -> None:
        # Keep the encoder variants and parameter names aligned with the
        # original VAE checkpoints so existing payloads restore without surgery.
        if self.encoder_type == "causal_conv":
            self.encoder = CausalConvEncoder(
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                name="encoder",
            )
        else:
            self.encoder = MLPEncoder(
                action_dim=self.action_dim,
                window_size=self.window_size,
                hidden_dim=self.hidden_dim,
                num_hidden_layers=self.num_hidden_layers,
                name="encoder",
            )

        self.fc_mu = TorchDense(self.latent_dim, name="fc_mu")
        self.fc_log_var = TorchDense(self.latent_dim, name="fc_log_var")
        self.decoder = MLPDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.action_dim,
            num_hidden_layers=self.num_hidden_layers,
            name="decoder",
        )
        if self.recon_aux_weight > 0:
            self.aux_recon_head = MLPDecoder(
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                out_dim=self.action_dim * self.window_size,
                num_hidden_layers=self.num_hidden_layers,
                name="aux_recon_head",
            )
        else:
            self.aux_recon_head = None

    def encode(self, x: Array) -> Tuple[Array, Array]:
        """Return latent mean and log variance for hand history `x`."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(
        self,
        mu: Array,
        log_var: Array,
        eps: Optional[Array] = None,
    ) -> Array:
        """Sample a latent code from `(mu, log_var)` using the VAE reparameterization."""
        std = jnp.exp(0.5 * log_var)
        if eps is None:
            eps = jax.random.normal(self.make_rng("sample"), std.shape, dtype=std.dtype)
        return mu + std * eps

    def decode(self, z: Array) -> Array:
        """Decode latent code `z` into an absolute hand pose."""
        return self.decoder(z)

    def __call__(
        self,
        x: Array,
        target: Array,
        beta: Optional[float] = None,
        eps: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """Run the training forward pass and return prediction/loss tensors."""
        beta_value = self.beta if beta is None else beta
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, eps=eps)
        pred = self.decode(z)

        # Main objective: reconstruct the next absolute hand pose while keeping
        # the latent posterior close to the unit Gaussian prior.
        recon_loss = jnp.mean(jnp.square(pred - target))
        kl_per_dim = -0.5 * (1.0 + log_var - jnp.square(mu) - jnp.exp(log_var))
        kl_per_dim_avg = jnp.mean(kl_per_dim, axis=0)
        if self.free_bits > 0:
            floor = jnp.full_like(kl_per_dim_avg, self.free_bits)
            kl_per_dim_avg = jnp.maximum(kl_per_dim_avg, floor)
        kl_loss = jnp.mean(kl_per_dim_avg)

        total_loss = recon_loss + beta_value * kl_loss
        if self.aux_recon_head is not None:
            x_flat = x.reshape((x.shape[0], self.action_dim * self.window_size))
            aux_pred = self.aux_recon_head(z)
            aux_recon_loss = jnp.mean(jnp.square(aux_pred - x_flat))
            total_loss = total_loss + self.recon_aux_weight * aux_recon_loss

        return pred, recon_loss, kl_loss, total_loss, mu, log_var

    def predict(
        self,
        x: Array,
        deterministic: bool = True,
        eps: Optional[Array] = None,
    ) -> Array:
        """Predict the next absolute hand pose from a hand history window."""
        mu, log_var = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, log_var, eps=eps)
        return self.decode(z)


def count_params(params: Sequence[Array]) -> int:
    """Return the total number of scalar parameters in a pytree."""
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves))
