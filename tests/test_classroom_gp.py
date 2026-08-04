import unittest

import numpy as np
import pandas as pd

from core.sim.chem_functions import chem_eval_row
from core.utils.classroom_gp import fit_known_noise_gp_1d, predict_rescaled_gp
from core.utils.init_designs import generate_initial_points


def _build_demo_observations(noise_sigma: float) -> pd.DataFrame:
    one_d_space = [("Temperature", 20.0, 120.0, "C", "continuous")]
    obs_x = generate_initial_points(one_d_space, 8, method="lhs", seed=24)
    rng = np.random.default_rng(24)
    rows = []
    for x in obs_x:
        t = float(x[0])
        y_true = float(chem_eval_row([t, 0.6], mode="basic"))
        y_meas = float(np.clip(y_true + rng.normal(0.0, float(noise_sigma)), 0.0, 100.0))
        rows.append({"Temperature": t, "TrueYield": y_true, "MeasuredYield": y_meas})
    return (
        pd.DataFrame(rows)
        .groupby("Temperature", as_index=False)
        .agg({"TrueYield": "mean", "MeasuredYield": "mean"})
        .sort_values("Temperature")
        .reset_index(drop=True)
    )


class ClassroomGPTests(unittest.TestCase):
    def test_known_noise_gp_returns_finite_predictions(self) -> None:
        df_obs = _build_demo_observations(noise_sigma=4.0)
        x_train = df_obs["Temperature"].to_numpy(dtype=float).reshape(-1, 1)
        y_train = -df_obs["MeasuredYield"].to_numpy(dtype=float)
        model, y_mean, y_scale = fit_known_noise_gp_1d(x_train, y_train, noise_sigma=4.0)
        x_pred = np.linspace(20.0, 120.0, 128).reshape(-1, 1)
        mu, std = predict_rescaled_gp(model, x_pred, y_mean=y_mean, y_scale=y_scale)
        self.assertTrue(np.isfinite(mu).all())
        self.assertTrue(np.isfinite(std).all())
        self.assertGreaterEqual(float(np.min(std)), 0.0)

    def test_uncertainty_increases_with_larger_noise(self) -> None:
        df_low = _build_demo_observations(noise_sigma=0.0)
        df_high = _build_demo_observations(noise_sigma=8.0)

        low_model, low_mean, low_scale = fit_known_noise_gp_1d(
            df_low["Temperature"].to_numpy(dtype=float).reshape(-1, 1),
            -df_low["MeasuredYield"].to_numpy(dtype=float),
            noise_sigma=0.0,
        )
        high_model, high_mean, high_scale = fit_known_noise_gp_1d(
            df_high["Temperature"].to_numpy(dtype=float).reshape(-1, 1),
            -df_high["MeasuredYield"].to_numpy(dtype=float),
            noise_sigma=8.0,
        )

        x_pred = np.linspace(20.0, 120.0, 128).reshape(-1, 1)
        _, low_std = predict_rescaled_gp(low_model, x_pred, y_mean=low_mean, y_scale=low_scale)
        _, high_std = predict_rescaled_gp(high_model, x_pred, y_mean=high_mean, y_scale=high_scale)

        self.assertGreater(float(np.mean(high_std)), float(np.mean(low_std)))
        self.assertLess(float(np.max(high_std)), 50.0)


if __name__ == "__main__":
    unittest.main()
