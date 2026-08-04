from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def fit_known_noise_gp_1d(
    x_train: np.ndarray,
    y_train: np.ndarray,
    noise_sigma: float = 0.0,
    random_state: int = 42,
) -> tuple[GaussianProcessRegressor, float, float]:
    """
    Fit a stable 1D GP when the observation noise sigma is known.

    Targets are scaled explicitly so the user-facing sigma is applied in the
    correct units without destabilizing the GP fit.
    """
    x_arr = np.asarray(x_train, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y_train, dtype=float).reshape(-1)

    y_mean = float(np.mean(y_arr))
    y_scale = max(float(np.std(y_arr)), 1.0)
    y_scaled = (y_arr - y_mean) / y_scale

    scaled_sigma = max(float(noise_sigma), 1e-4) / y_scale
    noise_var = float(scaled_sigma**2)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=12.0,
        length_scale_bounds=(1e-2, 1e3),
        nu=2.5,
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        alpha=noise_var,
        n_restarts_optimizer=4,
        random_state=int(random_state),
    )
    model.fit(x_arr, y_scaled)
    return model, y_mean, y_scale


def predict_rescaled_gp(
    model: GaussianProcessRegressor,
    x_pred: np.ndarray,
    y_mean: float,
    y_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mean and std in the original target scale."""
    x_arr = np.asarray(x_pred, dtype=float).reshape(-1, 1)
    mu_scaled, std_scaled = model.predict(x_arr, return_std=True)
    mu = np.asarray(mu_scaled, dtype=float).reshape(-1) * float(y_scale) + float(y_mean)
    std = np.asarray(std_scaled, dtype=float).reshape(-1) * float(y_scale)
    return mu, std
