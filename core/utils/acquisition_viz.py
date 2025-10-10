import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from skopt.acquisition import gaussian_ei


def plot_acquisition_1d(optimizer, resolution=200):
    """Legacy helper: plots EI for pure 1D spaces."""
    skopt_opt = optimizer.skopt_optimizer
    if len(skopt_opt.space.dimensions) != 1:
        st.warning("Acquisition function visualization only supported for 1D at the moment.")
        return

    dim = skopt_opt.space.dimensions[0]
    x = np.linspace(dim.low, dim.high, resolution).reshape(-1, 1)
    Xt = skopt_opt.space.transform(x)
    model = skopt_opt.models[-1]
    # best (minimal) observed y in skopt's convention
    try:
        y_opt = float(np.min(np.asarray(skopt_opt.yi)))
    except Exception:
        y_opt = None
    acq = gaussian_ei(Xt, model, y_opt=y_opt)

    fig, ax = plt.subplots()
    ax.plot(x, acq, label="Expected Improvement", color="tab:orange")
    ax.set_xlabel(dim.name or "x")
    ax.set_ylabel("Acquisition Value")
    ax.set_title("Acquisition Function (EI)")
    ax.legend()
    st.pyplot(fig)


def plot_gp_and_acq_1d(optimizer, var_name, fixed_values: dict, data_df=None, response_name: str | None = None, direction: str = "Maximize", resolution=200, xi=0.01):
    """
    Plot GP posterior mean/std and Expected Improvement along one variable, holding others fixed.
    - optimizer: StepBayesianOptimizer wrapper
    - var_name: name of the variable to sweep
    - fixed_values: dict {var: value} for all other vars in the model space
    - data_df: optional DataFrame of observed data to overlay
    - response_name: column name for response in data_df (for overlay)
    - direction: 'Maximize' or 'Minimize' (used only for overlay orientation)
    """
    skopt_opt = optimizer.skopt_optimizer
    dims = skopt_opt.space.dimensions
    names = [d.name for d in dims]
    if var_name not in names:
        st.warning("Selected variable not found in the optimizer space.")
        return
    if not skopt_opt.models:
        st.warning("Optimizer has no trained model yet. Add observations first.")
        return

    idx = names.index(var_name)
    d = dims[idx]
    # Build sweep candidates in original space
    if hasattr(d, "low") and hasattr(d, "high"):
        xs = np.linspace(d.low, d.high, resolution)
    else:
        # Categorical
        xs = list(d.categories)

    X_candidates = []
    for val in xs:
        row = []
        for j, dj in enumerate(dims):
            namej = names[j]
            if j == idx:
                row.append(val)
            else:
                v = fixed_values.get(namej)
                # fallbacks for missing fixed values
                if v is None:
                    if hasattr(dj, "low") and hasattr(dj, "high"):
                        v = 0.5 * (dj.low + dj.high)
                    else:
                        v = dj.categories[0]
                row.append(v)
        X_candidates.append(row)

    # Transform candidates and compute acquisition values (EI)
    Xt = skopt_opt.space.transform(X_candidates)
    model = skopt_opt.models[-1]
    try:
        y_opt = float(np.min(np.asarray(skopt_opt.yi)))
    except Exception:
        y_opt = None
    acq = gaussian_ei(Xt, model, y_opt=y_opt, xi=xi)
    mu, std = model.predict(Xt, return_std=True)

    # For display: if user maximizes, flip sign so plotted mean aligns with user notion
    if direction == "Maximize":
        mu_disp = -mu  # model minimized -y
    else:
        mu_disp = mu

    fig, ax1 = plt.subplots(figsize=(7, 4))
    xaxis = xs if isinstance(xs, np.ndarray) else np.arange(len(xs))
    # Mean and uncertainty band
    ax1.plot(xaxis, mu_disp, color="tab:blue", label="GP mean")
    ax1.fill_between(xaxis, mu_disp - std, mu_disp + std, color="tab:blue", alpha=0.2, label="+/- 1s")
    ax1.set_xlabel(var_name)
    ax1.set_ylabel("Objective (display scale)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Overlay observed points if provided
    if data_df is not None and response_name and var_name in data_df.columns and response_name in data_df.columns:
        try:
            xv = pd.to_numeric(data_df[var_name], errors="coerce") if np.issubdtype(type(xs[0] if isinstance(xs, list) else xs[0]), np.number) else data_df[var_name]
            yv = pd.to_numeric(data_df[response_name], errors="coerce")
            mask = yv.notna()
            ax1.scatter(xv[mask], yv[mask], s=20, color="gray", alpha=0.6, label="Observed")
        except Exception:
            pass

    # Acquisition on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(xaxis, acq, color="tab:orange", label="EI")
    ax2.set_ylabel("Acquisition (EI)", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    # Mark argmax EI
    i_best = int(np.argmax(acq))
    ax2.axvline(x=xaxis[i_best], color="tab:red", linestyle="--", alpha=0.7)

    fig.tight_layout()
    # Compose legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    st.pyplot(fig)


