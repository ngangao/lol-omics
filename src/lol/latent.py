"""Latent representation extraction: PCA and a small Variational Autoencoder.

Neither method is presented here as superior. The literature on small
bulk-expression cohorts doesn't settle that question -- a VAE tends to
edge out PCA on reconstruction only at small latent dimensions before
it starts to overfit, and PCA's own quality depends heavily on having
enough samples to begin with. LOL's job is to make it cheap to run
both and compare, not to declare a winner up front.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PCALatent:
    """Latent extraction via PCA. Instant, deterministic, CPU-only."""

    def __init__(self, n_components: int = 20, random_state: int = 0):
        self.n_components = n_components
        self.model = PCA(n_components=n_components, random_state=random_state)
        self.feature_names_: pd.Index | None = None

    def fit_transform(self, expression: pd.DataFrame) -> pd.DataFrame:
        n_components = min(self.n_components, *expression.shape)
        if n_components != self.model.n_components:
            self.model = PCA(n_components=n_components, random_state=0)
        latents = self.model.fit_transform(expression.to_numpy(dtype=float))
        self.feature_names_ = expression.columns
        columns = [f"pc{i + 1}" for i in range(latents.shape[1])]
        return pd.DataFrame(latents, index=expression.index, columns=columns)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.model.explained_variance_ratio_

    def gene_loadings(self, component: int = 0) -> pd.Series:
        """Fitted PCA loadings for one component, indexed by gene, sorted ascending.

        Call :meth:`fit_transform` first. ``component`` is 0-indexed
        (``0`` = PC1). Positive and negative loadings indicate genes
        driving opposite ends of that component; the largest-magnitude
        entries (``.head(n)`` / ``.tail(n)``) are the genes most
        strongly associated with it.

        This reuses the already-fitted PCA model rather than refitting
        a separate one -- refitting a second PCA on the same data to
        get loadings is redundant and, if any preprocessing step is
        later changed without also updating a standalone script doing
        that refit, a way for the two to silently diverge.
        """
        if self.feature_names_ is None:
            raise RuntimeError("Call fit_transform() before gene_loadings().")
        if component >= self.model.components_.shape[0]:
            raise ValueError(
                f"component={component} is out of range for a fitted model with "
                f"{self.model.components_.shape[0]} components."
            )
        return pd.Series(
            self.model.components_[component], index=self.feature_names_
        ).sort_values()


class _VAEModule(nn.Module):
    """A deliberately small VAE: one hidden layer in, one out.

    Sized for bulk RNA-seq/microarray cohorts (hundreds of samples,
    a few thousand genes after HVG selection), not single-cell scale.
    """

    def __init__(self, n_genes: int, n_hidden: int, n_latent: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_genes, n_hidden), nn.ReLU())
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_genes),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


@dataclass
class VAETrainingHistory:
    train_loss: list[float]


class VAELatent:
    """Latent extraction via a small VAE trained with PyTorch.

    Trains on CPU by default; on the gene/sample scale LOL targets
    (a few thousand genes, hundreds of samples), a run typically
    finishes in well under a minute on a laptop.
    """

    def __init__(
        self,
        n_latent: int = 20,
        n_hidden: int = 128,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        random_state: int = 0,
        device: str = "cpu",
    ):
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.device = device
        self._model: _VAEModule | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        torch.manual_seed(random_state)

    def fit_transform(self, expression: pd.DataFrame) -> pd.DataFrame:
        x = expression.to_numpy(dtype=np.float32)
        self._mean = x.mean(axis=0, keepdims=True)
        self._scale = x.std(axis=0, keepdims=True) + 1e-6
        x_scaled = (x - self._mean) / self._scale

        n_samples, n_genes = x_scaled.shape
        batch_size = min(self.batch_size, n_samples)

        self._model = _VAEModule(n_genes, self.n_hidden, self.n_latent).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_scaled)),
            batch_size=batch_size,
            shuffle=True,
        )

        history: list[float] = []
        self._model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self._model(batch)
                recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum")
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            history.append(epoch_loss / n_samples)
        self.history = VAETrainingHistory(train_loss=history)

        self._model.eval()
        with torch.no_grad():
            mu, _ = self._model.encode(torch.from_numpy(x_scaled).to(self.device))
        columns = [f"z{i + 1}" for i in range(self.n_latent)]
        return pd.DataFrame(mu.cpu().numpy(), index=expression.index, columns=columns)
