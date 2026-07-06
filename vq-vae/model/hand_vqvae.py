"""DQ-RISE aligned Flax/JAX VQ-VAE for hand-action tokenization."""

from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp


Array = jax.Array


def _torch_orthogonal_kernel(key, shape, dtype=jnp.float32):
    """Initialize a Flax Dense kernel like torch.nn.init.orthogonal_ on Linear.weight."""
    in_features, out_features = shape
    rows, cols = out_features, in_features
    flat_shape = (cols, rows) if rows < cols else (rows, cols)
    a = jax.random.normal(key, flat_shape, dtype)
    q, r = jnp.linalg.qr(a)
    ph = jnp.sign(jnp.diag(r))
    q = q * ph
    if rows < cols:
        q = q.T
    weight = q.reshape((rows, cols))
    return weight.T.astype(dtype)


def _dq_codebook_init(key, shape, dtype=jnp.float32):
    """Approximate vector_quantize_pytorch's kaiming_uniform_ codebook init."""
    _, codebook_size, dim = shape
    bound = 1.0 / jnp.sqrt(float(codebook_size * dim))
    return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)


def _laplace_smoothing(x: Array, n_categories: int, eps: float) -> Array:
    denom = jnp.sum(x, axis=-1, keepdims=True)
    return (x + eps) / (denom + float(n_categories) * eps)


class DQRiseDense(nn.Module):
    """Dense layer matching DQ-RISE's Linear + orthogonal weight + zero bias."""

    features: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        kernel = self.param("kernel", _torch_orthogonal_kernel, (x.shape[-1], self.features))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        return jnp.matmul(x, kernel) + bias


class EncoderMLP(nn.Module):
    """DQ-RISE EncoderMLP: Linear-ReLU, `layer_num` hidden repeats, final Linear."""

    input_dim: int
    output_dim: int
    hidden_dim: int = 512
    layer_num: int = 5

    def setup(self) -> None:
        self.first = DQRiseDense(self.hidden_dim, name="first")
        self.hidden = [DQRiseDense(self.hidden_dim, name=f"hidden_{i}") for i in range(self.layer_num)]
        self.fc = DQRiseDense(self.output_dim, name="fc")

    def __call__(self, x: Array) -> Array:
        h = nn.relu(self.first(x))
        for layer in self.hidden:
            h = nn.relu(layer(h))
        return self.fc(h)


class DQRiseResidualVQ(nn.Module):
    """ResidualVQ aligned with DQ-RISE's vector_quantize_pytorch usage."""

    dim: int
    num_layers: int = 2
    codebook_size: int = 4
    decay: float = 0.8
    eps: float = 1e-5
    threshold_ema_dead_code: float = 0.0

    def setup(self) -> None:
        self.layer_weights = self.param(
            "layer_weights",
            lambda _key, shape: jnp.full(shape, 0.5, dtype=jnp.float32),
            (self.num_layers,),
        )
        self.codebooks = self.variable(
            "vq_state",
            "codebooks",
            lambda shape: _dq_codebook_init(self.make_rng("params"), shape),
            (self.num_layers, self.codebook_size, self.dim),
        )
        self.embed_avg = self.variable(
            "vq_state",
            "embed_avg",
            lambda: self.codebooks.value,
        )
        self.cluster_size = self.variable(
            "vq_state",
            "cluster_size",
            lambda: jnp.zeros((self.num_layers, self.codebook_size), dtype=jnp.float32),
        )

    def _expire_codes(self, codebook: Array, embed_avg: Array, cluster_size: Array, residual: Array) -> tuple[Array, Array, Array]:
        if self.threshold_ema_dead_code <= 0:
            return codebook, embed_avg, cluster_size
        expired = cluster_size < self.threshold_ema_dead_code
        sample_ids = jnp.arange(self.codebook_size) % residual.shape[0]
        sampled = residual[sample_ids]
        reset_cluster = jnp.full_like(cluster_size, self.threshold_ema_dead_code)
        codebook = jnp.where(expired[:, None], sampled, codebook)
        cluster_size = jnp.where(expired, reset_cluster, cluster_size)
        embed_avg = jnp.where(expired[:, None], sampled * reset_cluster[:, None], embed_avg)
        return codebook, embed_avg, cluster_size

    def __call__(self, z: Array, *, training: bool) -> Dict[str, Array]:
        residual = z
        quantized_sum = jnp.zeros_like(z)
        all_indices = []
        all_distances = []
        all_losses = []

        codebooks = self.codebooks.value
        embed_avg = self.embed_avg.value
        cluster_size = self.cluster_size.value
        new_codebooks = codebooks
        new_embed_avg = embed_avg
        new_cluster_size = cluster_size
        weights = nn.softmax(self.layer_weights, axis=0)

        for layer_idx in range(self.num_layers):
            codebook = new_codebooks[layer_idx]
            diff = residual[:, None, :] - codebook[None, :, :]
            distances = jnp.sum(jnp.square(diff), axis=-1)
            indices = jnp.argmin(distances, axis=-1)
            one_hot = jax.nn.one_hot(indices, self.codebook_size, dtype=z.dtype)
            quantized = one_hot @ codebook

            commit_loss = jnp.mean(jnp.square(jax.lax.stop_gradient(quantized) - residual))
            quantized_st = residual + jax.lax.stop_gradient(quantized - residual)

            if training:
                counts = jnp.sum(one_hot, axis=0)
                sums = one_hot.T @ residual
                layer_cluster = self.decay * new_cluster_size[layer_idx] + (1.0 - self.decay) * counts
                layer_embed_avg = self.decay * new_embed_avg[layer_idx] + (1.0 - self.decay) * sums
                smoothed = _laplace_smoothing(layer_cluster, self.codebook_size, self.eps) * jnp.sum(layer_cluster)
                layer_codebook = layer_embed_avg / jnp.maximum(smoothed[:, None], self.eps)
                layer_codebook, layer_embed_avg, layer_cluster = self._expire_codes(
                    layer_codebook,
                    layer_embed_avg,
                    layer_cluster,
                    residual,
                )
                new_codebooks = new_codebooks.at[layer_idx].set(layer_codebook)
                new_embed_avg = new_embed_avg.at[layer_idx].set(layer_embed_avg)
                new_cluster_size = new_cluster_size.at[layer_idx].set(layer_cluster)

            residual = residual - jax.lax.stop_gradient(quantized_st)
            quantized_sum = quantized_sum + quantized_st * weights[layer_idx]
            all_indices.append(indices)
            all_distances.append(distances)
            all_losses.append(commit_loss)

        if training:
            self.codebooks.value = new_codebooks
            self.embed_avg.value = new_embed_avg
            self.cluster_size.value = new_cluster_size

        return {
            "quantized": quantized_sum,
            "indices": jnp.stack(all_indices, axis=-1),
            "distances": jnp.stack(all_distances, axis=1),
            "vq_loss_state": jnp.sum(jnp.stack(all_losses)),
            "layer_weights": weights,
        }

    def lookup_dqrise_export(self, indices: Array) -> Array:
        """DQ-RISE export uses fixed 0.5 / 0.5 codebook averaging."""
        quantized = jnp.zeros((indices.shape[0], self.dim), dtype=self.codebooks.value.dtype)
        fixed_weight = 1.0 / float(self.num_layers)
        for layer_idx in range(self.num_layers):
            codes = jnp.take(self.codebooks.value[layer_idx], indices[:, layer_idx], axis=0)
            quantized = quantized + codes * fixed_weight
        return quantized


class HandVQVAE(nn.Module):
    """Single-step hand tokenizer mirroring DQ-RISE's VqVae."""

    action_dim: int = 6
    latent_dim: int = 256
    hidden_dim: int = 512
    num_vq_layers: int = 2
    codebook_size: int = 4
    vae_layer_num: int = 5
    encoder_loss_multiplier: float = 1.0
    recon_loss_weight: float = 3.0
    vq_loss_weight: float = 5.0
    normalize_actions: bool = True
    ema_decay: float = 0.8
    eps: float = 1e-5
    threshold_ema_dead_code: float = 0.0

    def setup(self) -> None:
        self.encoder = EncoderMLP(
            input_dim=self.action_dim,
            output_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            layer_num=self.vae_layer_num,
            name="encoder",
        )
        self.vq = DQRiseResidualVQ(
            dim=self.latent_dim,
            num_layers=self.num_vq_layers,
            codebook_size=self.codebook_size,
            decay=self.ema_decay,
            eps=self.eps,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            name="vq",
        )
        self.decoder = EncoderMLP(
            input_dim=self.latent_dim,
            output_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            layer_num=self.vae_layer_num,
            name="decoder",
        )

    def normalize_raw_action(self, x: Array) -> Array:
        if not self.normalize_actions:
            return x
        return x * 2.0 - 1.0

    def denormalize_action(self, x: Array) -> Array:
        if not self.normalize_actions:
            return x
        return (x + 1.0) / 2.0

    def encode(self, x: Array, *, training: bool = False) -> Dict[str, Array]:
        z = self.encoder(x)
        return self.vq(z, training=training)

    def decode_latent(self, z_q: Array) -> Array:
        return self.decoder(z_q)

    def decode_indices(self, indices: Array) -> Array:
        z_q = self.vq.lookup_dqrise_export(indices.astype(jnp.int32))
        decoded_norm = self.decode_latent(z_q)
        return self.denormalize_action(decoded_norm)

    def __call__(self, x: Array, *, training: bool = True) -> Dict[str, Array]:
        target = self.normalize_raw_action(x)
        vq_out = self.encode(target, training=training)
        recon_norm = self.decode_latent(vq_out["quantized"])
        loss_weight = jnp.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 1.0], dtype=target.dtype)
        encoder_loss = jnp.mean(jnp.abs(target - recon_norm) * loss_weight)
        recon_mse = jnp.mean(jnp.square(target - recon_norm))
        total_loss = encoder_loss * self.encoder_loss_multiplier * self.recon_loss_weight + vq_out["vq_loss_state"] * self.vq_loss_weight
        return {
            "recon": self.denormalize_action(recon_norm),
            "recon_norm": recon_norm,
            "indices": vq_out["indices"],
            "distances": vq_out["distances"],
            "encoder_loss": encoder_loss,
            "recon_loss": recon_mse,
            "commitment_loss": vq_out["vq_loss_state"],
            "vq_loss_state": vq_out["vq_loss_state"],
            "codebook_loss": jnp.asarray(0.0, dtype=target.dtype),
            "usage_loss": jnp.asarray(0.0, dtype=target.dtype),
            "layer_weights": vq_out["layer_weights"],
            "total_loss": total_loss,
        }


def count_params(tree) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(leaf.size for leaf in leaves))
