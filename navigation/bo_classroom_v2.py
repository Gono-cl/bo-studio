from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
from skopt.space import Real
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from core.sim.chem_functions import chem_eval_row
from core.utils.bo_manual import safe_build_optimizer
from core.utils.hypervolume import hypervolume_2d
from core.utils.init_designs import generate_initial_points
from core.utils.knee import knee_index_2d
from core.utils.n_init_guidance import recommend_n_init_range, format_n_init_range
from core.utils.pareto import pareto_front_indices


MODULE_LABELS = [
    "1) The Optimization Problem in Chemistry",
    "2) Learn BO Intuition",
    "3) Understand BO Mechanics",
    "4) Chemist Workflow",
    "5) Multiobjective Decisions",
]
MODULE_IDS = {
    "1) The Optimization Problem in Chemistry": "intro",
    "2) Learn BO Intuition": "learn",
    "3) Understand BO Mechanics": "mechanics",
    "4) Chemist Workflow": "workflow",
    "5) Multiobjective Decisions": "mo",
}


INFO_CARD_STYLE = (
    "color:#000000; background:#f7f9fc; border:1px solid #d8e0ea; border-radius:10px; "
    "padding:14px 16px; margin:6px 0 18px 0; line-height:1.7;"
)
INFO_CARD_STYLE_BLUE = (
    "color:#000000; background:#eaf4ff; border:1px solid #bfd8f6; border-radius:10px; "
    "padding:14px 16px; margin:6px 0 18px 0; line-height:1.7;"
)
INFO_MISSION_STYLE = (
    "margin-top:10px; padding:10px 12px; background:#eef4ff; border:1px solid #cddcff; border-radius:8px;"
)
INTRO_CARD_MAIN = (
    "color:#111827; background:linear-gradient(180deg,#fff7ed 0%,#ffedd5 100%); "
    "border:1px solid #fdba74; border-left:6px solid #ea580c; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
INTRO_CARD_OFAT = (
    "color:#111827; background:linear-gradient(180deg,#eef2ff 0%,#e0e7ff 100%); "
    "border:1px solid #a5b4fc; border-left:6px solid #4f46e5; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
INTRO_CARD_DOE = (
    "color:#111827; background:linear-gradient(180deg,#fdf4ff 0%,#f5d0fe 100%); "
    "border:1px solid #d8b4fe; border-left:6px solid #a21caf; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
INTRO_CARD_BO = (
    "color:#111827; background:linear-gradient(180deg,#ecfeff 0%,#cffafe 100%); "
    "border:1px solid #67e8f9; border-left:6px solid #0891b2; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
INTRO_CARD_ADV = (
    "color:#111827; background:linear-gradient(180deg,#ecfdf5 0%,#d1fae5 100%); "
    "border:1px solid #86efac; border-left:6px solid #15803d; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
LEARN_CARD_GOALS = (
    "color:#111827; background:linear-gradient(180deg,#eaf4ff 0%,#dbeafe 100%); "
    "border:1px solid #93c5fd; border-left:6px solid #2563eb; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
LEARN_CARD_SURFACE = (
    "color:#111827; background:linear-gradient(180deg,#fff7ed 0%,#ffedd5 100%); "
    "border:1px solid #fdba74; border-left:6px solid #ea580c; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
LEARN_CARD_READ = (
    "color:#111827; background:linear-gradient(180deg,#f5f3ff 0%,#ede9fe 100%); "
    "border:1px solid #c4b5fd; border-left:6px solid #7c3aed; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
LEARN_CARD_LEGEND = (
    "color:#111827; background:linear-gradient(180deg,#ecfeff 0%,#cffafe 100%); "
    "border:1px solid #67e8f9; border-left:6px solid #0e7490; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)
LEARN_CARD_STEP = (
    "color:#111827; background:linear-gradient(180deg,#ecfdf5 0%,#dcfce7 100%); "
    "border:1px solid #86efac; border-left:6px solid #15803d; border-radius:12px; "
    "padding:14px 16px; margin:8px 0 16px 0; line-height:1.72;"
)

MODULE_THEME = {
    "intro": {"bg": "#eef2ff", "border": "#c7d2fe", "accent": "#4f46e5", "short": "Optimization Problem"},
    "learn": {"bg": "#eaf4ff", "border": "#bfd8f6", "accent": "#1f77b4", "short": "BO Intuition"},
    "mechanics": {"bg": "#ecfdf5", "border": "#b7ebd0", "accent": "#0f9d58", "short": "BO Mechanics"},
    "workflow": {"bg": "#fff7ed", "border": "#fed7aa", "accent": "#c2410c", "short": "Chemist Workflow"},
    "mo": {"bg": "#f5f3ff", "border": "#ddd6fe", "accent": "#6d28d9", "short": "Multiobjective"},
}


def _render_info_card(
    title: str,
    paragraphs: list[str] | None = None,
    bullets: list[str] | None = None,
    note: str | None = None,
    mission_title: str | None = None,
    mission_text: str | None = None,
    card_style: str | None = None,
) -> None:
    parts: list[str] = [
        f"<div style='font-weight:700; margin-bottom:6px;'>{html.escape(title)}</div>"
    ]
    for p in paragraphs or []:
        parts.append(f"<div style='margin-bottom:6px;'>{html.escape(p)}</div>")

    if bullets:
        parts.append("<ul style='margin-top:0; margin-bottom:8px; padding-left:20px;'>")
        for b in bullets:
            parts.append(f"<li>{html.escape(b)}</li>")
        parts.append("</ul>")

    if note:
        parts.append(f"<div style='margin-bottom:6px;'>{html.escape(note)}</div>")

    if mission_title or mission_text:
        inner = ""
        if mission_title:
            inner += f"<b>{html.escape(mission_title)}</b>"
        if mission_text:
            if inner:
                inner += "<br>"
            inner += html.escape(mission_text)
        parts.append(f"<div style='{INFO_MISSION_STYLE}'>{inner}</div>")

    final_style = card_style if card_style else INFO_CARD_STYLE
    st.markdown(f"<div style='{final_style}'>{''.join(parts)}</div>", unsafe_allow_html=True)


def _render_classroom_guide(module_label: str, teach_mode: str) -> None:
    current_idx = MODULE_LABELS.index(module_label)
    current_id = MODULE_IDS[module_label]

    _render_info_card(
        title="Guided Classroom Navigation",
        paragraphs=[
            "Use the sidebar Learning path panel to move through the modules in order.",
            "Recommended sequence: BO Loop Primer -> Intuition -> Mechanics -> Workflow -> Multiobjective.",
        ],
        bullets=[
            "Beginner mode: concept-first explanations with lighter math.",
            "Advanced mode: deeper formulas, assumptions, and interpretation.",
            f"Current mode: {teach_mode}.",
        ],
        note="Goal: follow a step-by-step progression and tune settings based on what you learned in the previous section.",
    )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    cols = st.columns(len(MODULE_LABELS))
    for i, label in enumerate(MODULE_LABELS):
        mid = MODULE_IDS[label]
        theme = MODULE_THEME.get(mid, {"bg": "#f7f9fc", "border": "#d8e0ea", "accent": "#1f77b4", "short": label})
        is_current = (label == module_label)
        border_width = "2px" if is_current else "1px"
        badge = "You are here" if is_current else f"Step {i + 1}"
        cols[i].markdown(
            (
                f"<div style='background:{theme['bg']}; border:{border_width} solid {theme['border']}; "
                f"border-radius:12px; padding:14px 14px; min-height:110px; margin:2px 5px 10px 5px;'>"
                f"<div style='color:{theme['accent']}; font-weight:700; font-size:0.92em; margin-bottom:8px;'>{html.escape(badge)}</div>"
                f"<div style='color:#111111; font-weight:700; margin-bottom:6px; font-size:1.08em;'>{html.escape(theme['short'])}</div>"
                f"<div style='color:#374151; font-size:0.95em; line-height:1.35;'>{html.escape(label.split(') ', 1)[-1])}</div>"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3 = st.columns([1.2, 1.2, 2.6])
    prev_clicked = nav_col1.button("Previous Step", disabled=(current_idx == 0), key="classroom_prev_step")
    next_clicked = nav_col2.button(
        "Next Step",
        disabled=(current_idx == len(MODULE_LABELS) - 1),
        key="classroom_next_step",
    )
    if prev_clicked and current_idx > 0:
        st.session_state["classroom_module_pending"] = MODULE_LABELS[current_idx - 1]
        st.rerun()
    if next_clicked and current_idx < len(MODULE_LABELS) - 1:
        st.session_state["classroom_module_pending"] = MODULE_LABELS[current_idx + 1]
        st.rerun()
    nav_col3.markdown(
        f"<div style='margin-top:12px; color:#374151; line-height:1.5;'>Current step: <b>{html.escape(MODULE_THEME[current_id]['short'])}</b>.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)


def _chem_space() -> list[tuple[str, float, float, str, str]]:
    return [
        ("Temperature", 20.0, 120.0, "C", "continuous"),
        ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
    ]


def _surface_grid(n_t: int = 60, n_c: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_vals = np.linspace(20.0, 120.0, n_t)
    c_vals = np.linspace(0.0, 1.0, n_c)
    z = np.zeros((n_c, n_t), dtype=float)
    for i, c in enumerate(c_vals):
        for j, t in enumerate(t_vals):
            z[i, j] = float(chem_eval_row([t, c], mode="basic"))
    return t_vals, c_vals, z


def _surface_figure(
    title: str,
    points: pd.DataFrame | None = None,
    suggest_point: tuple[float, float] | None = None,
) -> go.Figure:
    t_vals, c_vals, z = _surface_grid()
    fig = go.Figure(
        data=go.Contour(
            x=t_vals,
            y=c_vals,
            z=z,
            colorscale="Viridis",
            colorbar=dict(title="Predicted Yield"),
            contours=dict(showlines=False),
        )
    )

    if points is not None and not points.empty:
        fig.add_trace(
            go.Scatter(
                x=points["Temperature"],
                y=points["Catalyst"],
                mode="markers",
                marker=dict(size=8, color="white", line=dict(color="black", width=1)),
                name="Experiments",
                text=[f"Objective function (yield): {y:.2f}" for y in points.get("Yield", pd.Series(dtype=float)).fillna(np.nan)],
                hovertemplate="Temp=%{x:.2f}<br>Catalyst=%{y:.3f}<br>%{text}<extra></extra>",
            )
        )

    if suggest_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[suggest_point[0]],
                y=[suggest_point[1]],
                mode="markers",
                marker=dict(size=14, symbol="star", color="red", line=dict(color="black", width=1)),
                name="Suggested Next Run",
                hovertemplate="Suggested: Temp=%{x:.2f}, Catalyst=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, pad=dict(b=16)),
        xaxis_title="Temperature (C)",
        yaxis_title="Catalyst Loading",
        height=470,
        margin=dict(l=20, r=20, t=82, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.20)",
            borderwidth=1,
        ),
    )
    return fig


def _min_pairwise_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    min_d = np.inf
    for i in range(points.shape[0] - 1):
        d = points[i + 1 :] - points[i]
        dist = np.sqrt(np.sum(d * d, axis=1))
        local_min = float(np.min(dist))
        if local_min < min_d:
            min_d = local_min
    return float(min_d)


def _mean_nearest_neighbor_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    n = points.shape[0]
    d_acc = 0.0
    for i in range(n):
        d = points - points[i]
        dist = np.sqrt(np.sum(d * d, axis=1))
        dist[i] = np.inf
        d_acc += float(np.min(dist))
    return d_acc / n


def _normalize_2d(pts: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (pts - lo) / (hi - lo + 1e-12)


def _collapse_1d_observations(df_obs: pd.DataFrame) -> pd.DataFrame:
    if df_obs.empty:
        return df_obs.copy()
    out = (
        df_obs.groupby("Temperature", as_index=False)
        .agg({"TrueYield": "mean", "MeasuredYield": "mean"})
        .sort_values("Temperature")
    )
    return out.reset_index(drop=True)


def _initialize_1d_campaign(
    fixed_catalyst: float,
    n_obs: int,
    init_method_label: str,
    noise_sigma: float,
    seed: int,
    gp_resolution: int,
    gp_af: str,
    gp_xi: float,
    gp_kappa: float,
) -> dict[str, object]:
    one_d_space = [("Temperature", 20.0, 120.0, "C", "continuous")]
    obs_x = generate_initial_points(
        one_d_space,
        int(n_obs),
        method=str(init_method_label).lower().replace(" ", "_"),
        seed=int(seed),
    )
    rng = np.random.default_rng(int(seed))
    rows = []
    for x in obs_x:
        t = float(x[0])
        y_true = float(chem_eval_row([t, float(fixed_catalyst)], mode="basic"))
        y_meas = float(np.clip(y_true + rng.normal(0.0, float(noise_sigma)), 0.0, 100.0))
        rows.append({"Temperature": t, "TrueYield": y_true, "MeasuredYield": y_meas})
    df_obs = pd.DataFrame(rows).sort_values("Temperature")
    df_obs = _collapse_1d_observations(df_obs)

    af_choice = str(gp_af).upper()
    if af_choice not in {"EI", "PI", "LCB"}:
        af_choice = "EI"

    return {
        "df_obs": df_obs,
        "fixed_catalyst": float(fixed_catalyst),
        "noise_sigma": float(noise_sigma),
        "seed": int(seed),
        "gp_resolution": int(gp_resolution),
        "af_choice": af_choice,
        "af_xi": float(gp_xi),
        "af_kappa": float(gp_kappa),
        "step_count": 0,
    }


def _largest_gap_midpoint(existing_t: np.ndarray) -> float:
    existing_t = np.asarray(existing_t, dtype=float).reshape(-1)
    if existing_t.size == 0:
        return 70.0
    clipped = np.clip(existing_t, 20.0, 120.0)
    anchors = np.sort(np.unique(np.concatenate(([20.0], clipped, [120.0]))))
    if anchors.size < 2:
        return 70.0
    gaps = anchors[1:] - anchors[:-1]
    idx = int(np.argmax(gaps))
    return float((anchors[idx] + anchors[idx + 1]) / 2.0)


def _extract_model_noise_variance(model) -> float:
    """Best-effort extraction of WhiteKernel noise variance from a fitted GP model."""
    # skopt wrappers sometimes expose estimated noise directly.
    for attr in ("noise_", "noise"):
        try:
            v = float(getattr(model, attr))
            if np.isfinite(v) and v >= 0.0:
                return v
        except Exception:
            pass

    def _scan_kernel(k) -> float | None:
        if k is None:
            return None
        try:
            if hasattr(k, "noise_level"):
                v = float(getattr(k, "noise_level"))
                if np.isfinite(v) and v >= 0.0:
                    return v
        except Exception:
            pass
        for child in (getattr(k, "k1", None), getattr(k, "k2", None)):
            v = _scan_kernel(child)
            if v is not None:
                return v
        return None

    try:
        kval = _scan_kernel(getattr(model, "kernel_", None))
        if kval is not None:
            return float(kval)
    except Exception:
        pass
    return 0.0


def _valid_1d_candidate_mask(
    t_grid: np.ndarray,
    observed_t: np.ndarray,
    min_distance: float = 0.25,
) -> np.ndarray:
    """Mask out points that are too close to already observed temperatures."""
    if observed_t.size == 0:
        return np.ones_like(t_grid, dtype=bool)
    nearest = np.min(np.abs(t_grid.reshape(-1, 1) - observed_t.reshape(1, -1)), axis=1)
    return nearest > float(min_distance)


def _fit_stable_1d_model(df_obs: pd.DataFrame) -> tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]:
    """
    Fit a stable GP for classroom intuition plots/suggestions.

    We fix an extremely small noise level to avoid high-noise local optima that can
    make uncertainty remain unrealistically wide at observed points in low-data cases.
    """
    x_train = df_obs["Temperature"].to_numpy(dtype=float).reshape(-1, 1)
    y_train = -df_obs["MeasuredYield"].to_numpy(dtype=float)  # skopt AFs are minimization-oriented
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=12.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-8, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-10,
        n_restarts_optimizer=4,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model, x_train, y_train


def _suggest_unique_1d_temperature(
    df_obs: pd.DataFrame,
    acq_func: str,
    gp_xi: float = 0.01,
    gp_kappa: float = 1.96,
    gp_resolution: int = 220,
) -> float:
    try:
        model, _, y_train = _fit_stable_1d_model(df_obs)
    except Exception:
        return _largest_gap_midpoint(df_obs["Temperature"].to_numpy(dtype=float))

    t_grid = np.linspace(20.0, 120.0, int(max(80, gp_resolution)))
    xt = t_grid.reshape(-1, 1)
    y_opt = float(np.min(np.asarray(y_train, dtype=float)))

    acq = str(acq_func).upper()
    if acq == "PI":
        af_values = gaussian_pi(xt, model, y_opt=y_opt, xi=float(gp_xi))
    elif acq == "LCB":
        af_values = -gaussian_lcb(xt, model, kappa=float(gp_kappa))
    else:
        af_values = gaussian_ei(xt, model, y_opt=y_opt, xi=float(gp_xi))

    af_values = np.asarray(af_values, dtype=float).reshape(-1)
    observed_t = df_obs["Temperature"].to_numpy(dtype=float)
    valid_mask = _valid_1d_candidate_mask(t_grid, observed_t, min_distance=0.25)

    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        best_local = int(valid_idx[int(np.argmax(af_values[valid_mask]))])
        return float(t_grid[best_local])
    return _largest_gap_midpoint(observed_t)


def _advance_1d_campaign(campaign: dict[str, object], n_steps: int) -> dict[str, object]:
    updated = dict(campaign)
    df_obs = updated["df_obs"].copy()
    fixed_catalyst = float(updated["fixed_catalyst"])
    noise_sigma = float(updated["noise_sigma"])
    seed = int(updated["seed"])
    acq_func = str(updated.get("af_choice", "EI")).upper()
    gp_xi = float(updated.get("af_xi", 0.01))
    gp_kappa = float(updated.get("af_kappa", 1.96))
    gp_resolution = int(updated.get("gp_resolution", 220))
    if acq_func not in {"EI", "PI", "LCB"}:
        acq_func = "EI"

    for _ in range(max(1, int(n_steps))):
        if len(df_obs) < 3:
            break
        t_next = _suggest_unique_1d_temperature(
            df_obs,
            acq_func=acq_func,
            gp_xi=gp_xi,
            gp_kappa=gp_kappa,
            gp_resolution=gp_resolution,
        )
        y_true = float(chem_eval_row([float(t_next), fixed_catalyst], mode="basic"))

        step_idx = int(updated.get("step_count", 0))
        noise_rng = np.random.default_rng(seed + 10000 + step_idx)
        y_meas = float(np.clip(y_true + noise_rng.normal(0.0, noise_sigma), 0.0, 100.0))

        df_obs = pd.concat(
            [
                df_obs,
                pd.DataFrame(
                    [{"Temperature": float(t_next), "TrueYield": y_true, "MeasuredYield": y_meas}]
                ),
            ],
            ignore_index=True,
        )
        df_obs = _collapse_1d_observations(df_obs)
        updated["step_count"] = step_idx + 1

    updated["df_obs"] = df_obs
    return updated


def _fit_1d_gp_result(campaign: dict[str, object]) -> dict[str, object]:
    df_obs = campaign["df_obs"].copy()
    fixed_catalyst = float(campaign["fixed_catalyst"])
    noise_sigma = float(campaign["noise_sigma"])
    gp_resolution = int(campaign["gp_resolution"])
    af_choice = str(campaign.get("af_choice", "EI")).upper()
    gp_xi = float(campaign.get("af_xi", 0.01))
    gp_kappa = float(campaign.get("af_kappa", 1.96))
    step_count = int(campaign.get("step_count", 0))

    if len(df_obs) < 3:
        return {
            "error": "Need at least 3 unique observations to fit a stable 1D GP. Try increasing points or changing seed."
        }
    try:
        model, x_train, y_train = _fit_stable_1d_model(df_obs)
    except Exception as ex:
        return {"error": f"Could not fit the 1D GP model. Technical detail: {ex}"}

    base_grid = np.linspace(20.0, 120.0, int(gp_resolution))
    obs_t = df_obs["Temperature"].to_numpy(dtype=float)
    # Include observed temperatures explicitly so uncertainty minima are visible at measured points.
    t_grid = np.sort(np.unique(np.concatenate([base_grid, obs_t])))
    xt = t_grid.reshape(-1, 1)
    mu, std = model.predict(xt, return_std=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if mu.shape[0] != t_grid.shape[0] or std.shape[0] != t_grid.shape[0]:
        return {"error": "GP output shape mismatch. Try a different configuration."}

    # Use epistemic uncertainty for visualization by removing the fitted noise floor.
    noise_var = _extract_model_noise_variance(model)
    std_ep = np.sqrt(np.clip(std**2 - float(noise_var), 0.0, None))

    mean_y = -mu
    lo_y = mean_y - 1.96 * std_ep
    hi_y = mean_y + 1.96 * std_ep
    true_grid = np.array(
        [float(chem_eval_row([float(t), float(fixed_catalyst)], mode="basic")) for t in t_grid],
        dtype=float,
    )
    y_opt = float(np.min(np.asarray(y_train, dtype=float)))

    if af_choice == "PI":
        af_values = gaussian_pi(xt, model, y_opt=y_opt, xi=gp_xi)
    elif af_choice == "LCB":
        # Convert minimization LCB into a maximization-style score for intuitive plotting.
        af_values = -gaussian_lcb(xt, model, kappa=gp_kappa)
    else:
        af_choice = "EI"
        af_values = gaussian_ei(xt, model, y_opt=y_opt, xi=gp_xi)
    af_values = np.asarray(af_values, dtype=float).reshape(-1)
    valid_mask = _valid_1d_candidate_mask(t_grid, obs_t, min_distance=0.25)
    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        af_argmax = int(valid_idx[int(np.argmax(af_values[valid_mask]))])
    else:
        af_argmax = int(np.argmax(af_values))
    af_values_display = af_values.copy()
    af_values_display[~valid_mask] = np.nan

    nearest_dist = np.min(np.abs(t_grid.reshape(-1, 1) - obs_t.reshape(1, -1)), axis=1)
    near_mask = nearest_dist <= 5.0
    far_mask = nearest_dist >= 15.0
    near_std = float(np.mean(std_ep[near_mask])) if np.any(near_mask) else float("nan")
    far_std = float(np.mean(std_ep[far_mask])) if np.any(far_mask) else float("nan")

    return {
        "df_obs": df_obs,
        "t_grid": t_grid,
        "true_grid": true_grid,
        "mean_y": mean_y,
        "lo_y": lo_y,
        "hi_y": hi_y,
        "near_std": near_std,
        "far_std": far_std,
        "noise_sigma": float(noise_sigma),
        "fixed_catalyst": float(fixed_catalyst),
        "af_choice": af_choice,
        "af_values": af_values,
        "af_values_display": af_values_display,
        "af_argmax": af_argmax,
        "af_peak_temperature": float(t_grid[af_argmax]),
        "af_peak_score": float(af_values[af_argmax]),
        "af_xi": gp_xi,
        "af_kappa": gp_kappa,
        "step_count": step_count,
        "error": None,
    }


def _apply_optimizer_acq_settings(optimizer, acq_func: str, xi: float, kappa: float) -> None:
    """Best-effort: apply exploration knobs to underlying skopt optimizer."""
    try:
        sk = getattr(optimizer, "skopt_optimizer", None) or getattr(optimizer, "_optimizer", None)
        if sk is None:
            return

        acq = str(acq_func).upper()
        kwargs = dict(getattr(sk, "acq_func_kwargs", {}) or {})
        if acq in {"EI", "PI"}:
            kwargs["xi"] = float(xi)
        elif acq == "LCB":
            kwargs["kappa"] = float(kappa)

        if hasattr(sk, "acq_func_kwargs"):
            setattr(sk, "acq_func_kwargs", kwargs)
        # Fallback attributes for versions that expose direct knobs.
        if hasattr(sk, "xi"):
            setattr(sk, "xi", float(xi))
        if hasattr(sk, "kappa"):
            setattr(sk, "kappa", float(kappa))
    except Exception:
        pass


def _suggest_unique_2d_mechanics_point(
    df: pd.DataFrame,
    acq_func: str,
    seed: int,
    acq_xi: float = 0.01,
    acq_kappa: float = 1.96,
) -> list[float]:
    dims = [Real(20.0, 120.0, name="Temperature"), Real(0.0, 1.0, name="Catalyst")]
    opt = safe_build_optimizer(dims, n_initial_points_remaining=0, acq_func=acq_func)
    _apply_optimizer_acq_settings(opt, acq_func=acq_func, xi=acq_xi, kappa=acq_kappa)
    for _, row in df.iterrows():
        opt.observe([float(row["Temperature"]), float(row["Catalyst"])], -float(row["Yield"]))

    existing = df[["Temperature", "Catalyst"]].to_numpy(dtype=float)
    for _ in range(20):
        x_next = [float(v) for v in opt.suggest()]
        if existing.size == 0:
            return x_next
        d = np.sqrt(np.sum((existing - np.array(x_next, dtype=float)) ** 2, axis=1))
        if float(np.min(d)) > 1e-8:
            return x_next

    # Fallback: sample random candidates and choose farthest from existing observations.
    rng = np.random.default_rng(int(seed))
    candidates = np.column_stack((rng.uniform(20.0, 120.0, size=256), rng.uniform(0.0, 1.0, size=256)))
    if existing.size == 0:
        best = candidates[0]
    else:
        min_d = []
        for c in candidates:
            d = np.sqrt(np.sum((existing - c.reshape(1, 2)) ** 2, axis=1))
            min_d.append(float(np.min(d)))
        best = candidates[int(np.argmax(np.array(min_d, dtype=float)))]
    return [float(best[0]), float(best[1])]


def _initialize_mechanics_campaign(
    n_init: int,
    init_method_label: str,
    acq: str,
    seed: int,
    acq_xi: float = 0.01,
    acq_kappa: float = 1.96,
    noise_sigma: float = 0.0,
) -> dict[str, object]:
    space = _chem_space()
    init_method = init_method_label.lower().replace(" ", "_")
    x_init = generate_initial_points(space, int(n_init), method=init_method, seed=int(seed))
    rng = np.random.default_rng(int(seed))

    rows = []
    for x in x_init:
        y_true = float(chem_eval_row(x, mode="basic"))
        y = float(np.clip(y_true + rng.normal(0.0, float(noise_sigma)), 0.0, 100.0))
        rows.append(
            {
                "Temperature": float(x[0]),
                "Catalyst": float(x[1]),
                "Yield": y,
                "TrueYield": y_true,
                "Source": "initial_design",
            }
        )
    df = pd.DataFrame(rows)
    x_next = _suggest_unique_2d_mechanics_point(
        df,
        acq_func=acq,
        seed=int(seed),
        acq_xi=float(acq_xi),
        acq_kappa=float(acq_kappa),
    )
    y_next = float(chem_eval_row(x_next, mode="basic"))
    return {
        "df": df,
        "x_next": (float(x_next[0]), float(x_next[1]), y_next),
        "n_init": int(n_init),
        "acq": str(acq),
        "acq_xi": float(acq_xi),
        "acq_kappa": float(acq_kappa),
        "noise_sigma": float(noise_sigma),
        "init_method_label": str(init_method_label),
        "seed": int(seed),
        "step_count": 0,
    }


def _advance_mechanics_campaign(campaign: dict[str, object], n_steps: int = 1) -> dict[str, object]:
    updated = dict(campaign)
    df = updated["df"].copy()
    acq = str(updated["acq"])
    acq_xi = float(updated.get("acq_xi", 0.01))
    acq_kappa = float(updated.get("acq_kappa", 1.96))
    noise_sigma = float(updated.get("noise_sigma", 0.0))
    seed = int(updated.get("seed", 42))
    step_count = int(updated.get("step_count", 0))
    x_next_tuple = updated.get("x_next")
    if not isinstance(x_next_tuple, tuple) or len(x_next_tuple) < 2:
        x_next = _suggest_unique_2d_mechanics_point(
            df,
            acq_func=acq,
            seed=seed + step_count,
            acq_xi=acq_xi,
            acq_kappa=acq_kappa,
        )
    else:
        x_next = [float(x_next_tuple[0]), float(x_next_tuple[1])]

    for _ in range(max(1, int(n_steps))):
        y_true = float(chem_eval_row(x_next, mode="basic"))
        noise_rng = np.random.default_rng(seed + 10000 + step_count)
        y = float(np.clip(y_true + noise_rng.normal(0.0, noise_sigma), 0.0, 100.0))
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Temperature": float(x_next[0]),
                            "Catalyst": float(x_next[1]),
                            "Yield": y,
                            "TrueYield": y_true,
                            "Source": "bo_suggestion",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        step_count += 1
        x_next = _suggest_unique_2d_mechanics_point(
            df,
            acq_func=acq,
            seed=seed + step_count,
            acq_xi=acq_xi,
            acq_kappa=acq_kappa,
        )

    y_next = float(chem_eval_row(x_next, mode="basic"))
    updated["df"] = df
    updated["x_next"] = (float(x_next[0]), float(x_next[1]), y_next)
    updated["step_count"] = step_count
    return updated


def _summarize_mechanics_campaign(campaign: dict[str, object], snapshot_label: str) -> dict[str, object]:
    df = campaign.get("df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "Snapshot": snapshot_label,
            "Observations": 0,
            "BO steps": int(campaign.get("step_count", 0)),
            "Best Yield": np.nan,
            "Mean Yield": np.nan,
            "Obs to Yield >= 90": None,
            "n_init": int(campaign.get("n_init", 0)),
            "Init method": str(campaign.get("init_method_label", "")),
            "Acquisition": str(campaign.get("acq", "")),
            "xi": float(campaign.get("acq_xi", 0.01)),
            "kappa": float(campaign.get("acq_kappa", 1.96)),
            "Noise sigma": float(campaign.get("noise_sigma", 0.0)),
            "Seed": int(campaign.get("seed", 42)),
        }

    y_vals = pd.to_numeric(df.get("Yield", pd.Series(dtype=float)), errors="coerce")
    best_y = float(y_vals.max()) if not y_vals.empty else float("nan")
    mean_y = float(y_vals.mean()) if not y_vals.empty else float("nan")
    hit_mask = (y_vals >= 90.0).fillna(False)
    obs_to_90 = int(np.argmax(hit_mask.to_numpy()) + 1) if bool(hit_mask.any()) else None

    return {
        "Snapshot": snapshot_label,
        "Observations": int(len(df)),
        "BO steps": int(campaign.get("step_count", 0)),
        "Best Yield": round(best_y, 3) if pd.notna(best_y) else np.nan,
        "Mean Yield": round(mean_y, 3) if pd.notna(mean_y) else np.nan,
        "Obs to Yield >= 90": obs_to_90,
        "n_init": int(campaign.get("n_init", int((df.get("Source") == "initial_design").sum()) if "Source" in df.columns else 0)),
        "Init method": str(campaign.get("init_method_label", "")),
        "Acquisition": str(campaign.get("acq", "")),
        "xi": float(campaign.get("acq_xi", 0.01)),
        "kappa": float(campaign.get("acq_kappa", 1.96)),
        "Noise sigma": float(campaign.get("noise_sigma", 0.0)),
        "Seed": int(campaign.get("seed", 42)),
    }


def _render_global_theory(teach_mode: str) -> None:
    with st.expander("Core BO Theory Reference", expanded=False):
        if teach_mode == "Beginner":
            st.markdown(
                """
BO loop (plain language):
1. Run a few experiments.
2. Fit a model to learn pattern + uncertainty.
3. Score candidate experiments with an acquisition rule.
4. Run the best candidate and update the model.
5. Repeat until budget is used.

Why this works in chemistry:
- Experiments are expensive and slow.
- You need both optimization and learning efficiency.
- BO uses every new run to improve future decisions.
"""
            )
        else:
            st.markdown("**Intuition**")
            st.write("BO uses current experiments to build a probabilistic model, then selects the next experiment with highest expected decision value.")
            st.markdown("**Equations**")
            st.write("Data after t experiments:")
            st.latex(r"\mathcal{D}_t = \{(x_i, y_i)\}_{i=1}^t")
            st.write("Surrogate posterior:")
            st.latex(r"p(f \mid \mathcal{D}_t)")
            st.write("Acquisition policy:")
            st.latex(r"x_{t+1} = \arg\max_x a_t(x)")
            st.markdown("**Chemist interpretation**")
            st.markdown(
                """
- x: reaction conditions (temperature, catalyst, solvent, time, etc.)
- y: measured objective function (yield, purity, cost proxy, E-factor, ...)
- Acquisition encodes exploration/exploitation strategy under limited budget.
"""
            )
            st.markdown("**Practical takeaway**")
            st.write("Each new lab run updates both predicted performance and uncertainty, which changes the ranking of next candidates.")


def _render_bo_loop_scheme() -> None:
    st.markdown(
        """
<div style="display:flex; justify-content:center; margin:6px 0 10px 0;">
<svg viewBox="0 0 920 420" width="100%" style="max-width:860px; background:#f8fafc; border:1px solid #dbe6f1; border-radius:12px;">
  <defs>
    <marker id="arrowHead" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
      <polygon points="0 0, 10 4, 0 8" fill="#cbd5e1"></polygon>
    </marker>
  </defs>

  <!-- Loop arrows -->
  <path d="M 305 95 Q 460 30 615 95" stroke="#cbd5e1" stroke-width="10" fill="none" marker-end="url(#arrowHead)"></path>
  <path d="M 675 145 Q 745 210 675 280" stroke="#cbd5e1" stroke-width="10" fill="none" marker-end="url(#arrowHead)"></path>
  <path d="M 615 328 Q 460 390 305 328" stroke="#cbd5e1" stroke-width="10" fill="none" marker-end="url(#arrowHead)"></path>
  <path d="M 245 280 Q 175 210 245 145" stroke="#cbd5e1" stroke-width="10" fill="none" marker-end="url(#arrowHead)"></path>

  <!-- Top box -->
  <rect x="365" y="38" rx="34" ry="34" width="190" height="70" fill="#f58b6b"></rect>
  <text x="460" y="83" text-anchor="middle" font-size="24" font-weight="700" fill="#111111">Make</text>

  <!-- Right box -->
  <rect x="670" y="165" rx="30" ry="30" width="190" height="90" fill="#b68ae8"></rect>
  <text x="765" y="220" text-anchor="middle" font-size="24" font-weight="700" fill="#111111">Analyze</text>

  <!-- Bottom box -->
  <rect x="365" y="302" rx="34" ry="34" width="190" height="74" fill="#69c5e7"></rect>
  <text x="460" y="349" text-anchor="middle" font-size="24" font-weight="700" fill="#111111">Model</text>

  <!-- Left box -->
  <rect x="58" y="158" rx="30" ry="30" width="245" height="110" fill="#f59a2f"></rect>
  <text x="180" y="208" text-anchor="middle" font-size="20" font-weight="700" fill="#111111">Propose a new</text>
  <text x="180" y="242" text-anchor="middle" font-size="20" font-weight="700" fill="#111111">experiment</text>

  <!-- Center block -->
  <rect x="366" y="156" rx="24" ry="24" width="188" height="110" fill="#7ee7ea" stroke="#0f172a" stroke-width="3"></rect>
  <text x="460" y="224" text-anchor="middle" font-size="20" font-weight="700" fill="#111111">Active Learning</text>
</svg>
</div>
""",
        unsafe_allow_html=True,
    )


def _module_intro(teach_mode: str) -> None:
    st.subheader("1) The Optimization Problem in Chemistry")
    _render_info_card(
        title="What is the optimization problem in chemistry?",
        paragraphs=[
            "Most reaction systems depend on several variables at the same time, not one-by-one. Typical examples are temperature, catalyst loading, pressure, residence time, equivalents, solvent, and reagent ratio.",
            "The goal is usually to maximize an objective function (for example yield) while using as few experiments as possible.",
        ],
        bullets=[
            "Search spaces are multivariable and can include continuous + categorical parameters.",
            "Variable interactions are common, so local trends can be misleading.",
            "Each experiment has cost, time, and material impact.",
        ],
        note="So the key question is not only 'what gives the best yield?' but also 'how can we find it efficiently?'",
        card_style=INTRO_CARD_MAIN,
    )

    _render_info_card(
        title="Limitation of OFAT (One-Factor-At-a-Time)",
        paragraphs=[
            "OFAT is intuitive and easy to run, but it changes one variable while keeping others fixed.",
            "In chemistry this can miss strong interactions and can lock decisions around suboptimal regions.",
        ],
        bullets=[
            "Good for quick screening or basic troubleshooting.",
            "Weak for global optimization in multivariable spaces.",
        ],
        note="OFAT can identify trends, but often not the true best combination of conditions.",
        card_style=INTRO_CARD_OFAT,
    )

    _render_info_card(
        title="Limitation of classical DoE in expensive campaigns",
        paragraphs=[
            "DoE is statistically rigorous and very useful, but required runs can grow quickly with dimensionality and constraints.",
            "When each experiment is slow/expensive, fixed designs may be less practical than adaptive sequential decisions.",
        ],
        bullets=[
            "Strength: structured coverage and interaction estimation.",
            "Challenge: budget pressure in high-dimensional or iterative workflows.",
        ],
        note="Many labs combine DoE principles with adaptive BO to reduce run count while keeping decision quality.",
        card_style=INTRO_CARD_DOE,
    )

    st.markdown("#### Active Learning Loop in BO")
    _render_bo_loop_scheme()
    _render_info_card(
        title="What Bayesian Optimization does (simple words)",
        paragraphs=[
            "Bayesian Optimization is a strategy to optimize expensive experiments with as few trials as possible.",
            "At each step, it combines current data with a probabilistic model to estimate both expected performance and uncertainty across the search space.",
        ],
        bullets=[
            "Use available data to estimate where the objective is likely high.",
            "Quantify uncertainty in regions that are still poorly explored.",
            "Select the next experiment by balancing expected gain and information value.",
            "Update decisions after every new measured result.",
        ],
        note="BO is data-driven and sequential: each new experiment is chosen using what has already been learned.",
        card_style=INTRO_CARD_BO,
    )

    _render_info_card(
        title="Why Bayesian Optimization is an active learning algorithm",
        paragraphs=[
            "Active learning means the algorithm does not passively consume random data; it actively chooses which data point to acquire next.",
            "In BO, the next experiment is selected because it is expected to be the most useful for improving optimization decisions.",
        ],
        bullets=[
            "It asks: which experiment should I run next to learn the most and improve performance fastest?",
            "The decision changes after each new observation.",
            "This adaptive loop is what makes BO especially effective for expensive chemical campaigns.",
        ],
        note="In short: BO learns from data and also decides what data to collect next.",
        card_style=INTRO_CARD_ADV,
    )

    if teach_mode == "Advanced":
        _render_info_card(
            title="Advanced note (why this matters mathematically)",
            paragraphs=[
                "BO can be framed as sequential decision-making under expensive, noisy black-box evaluations.",
                "Performance depends on both components: surrogate model assumptions and acquisition policy behavior.",
            ],
            note="The next sections connect this high-level loop to GP posterior updates and AF scoring.",
            card_style=INTRO_CARD_ADV,
        )


def _module_learn(teach_mode: str) -> None:
    st.subheader("2) Learn BO Intuition")
    _render_global_theory(teach_mode)
    _render_info_card(
        title="Learning goals for this section",
        bullets=[
            "Understand how the Gaussian Process represents expected objective function (yield) and uncertainty.",
            "See how the Acquisition Function converts model outputs into the next experiment suggestion.",
            "Build intuition for why BO recommendations change after every new measured point.",
        ],
        note=(
            "Beginner mode keeps this conceptual; Advanced mode adds formal equations for the same ideas."
            if teach_mode == "Beginner"
            else "You are in Advanced mode: same intuition, with deeper mathematical interpretation."
        ),
        card_style=LEARN_CARD_GOALS,
    )

    _render_info_card(
        title="Synthetic chemistry objective function (yield) surface (2D starting point)",
        paragraphs=[
            "A 2D visualization is a good starting point to build BO intuition before moving to higher-dimensional campaigns.",
            "Here we map Temperature and Catalyst loading against predicted objective function (yield) to show peaks, valleys, and narrow high-performing regions.",
        ],
        note="This simple map helps explain why BO is useful: it guides experiments toward promising zones while learning uncertainty.",
        card_style=LEARN_CARD_SURFACE,
    )
    _render_info_card(
        title="How to read this surface plot",
        bullets=[
            "Bright regions correspond to higher predicted objective-function (yield) values.",
            "Wide low-value areas show why random exploration can waste experiments.",
            "Small high-performing zones explain why model-guided search is useful.",
        ],
        card_style=LEARN_CARD_READ,
    )

    t_vals, c_vals, z = _surface_grid()
    best_idx = np.unravel_index(np.argmax(z), z.shape)
    best_t = float(t_vals[best_idx[1]])
    best_c = float(c_vals[best_idx[0]])
    best_y = float(z[best_idx])

    st.plotly_chart(_surface_figure("Synthetic Chemistry Objective Function (Yield) Surface"), use_container_width=True)
    st.caption(
        f"Approximate optimum in this teaching model: Temperature={best_t:.1f} C, "
        f"Catalyst={best_c:.3f}, objective function (yield)~{best_y:.2f}."
    )

    st.markdown("#### 1D Gaussian Process (Mean + Uncertainty)")
    if teach_mode == "Beginner":
        _render_info_card(
            title="Gaussian Process + Acquisition Function: How BO chooses experiments",
            paragraphs=[
                "Use this section to build intuition about how Gaussian Processes and acquisition functions work together to define the next experiment to run.",
                "Modify parameter values to see how this affects the output. You are optimizing the model shown above, where high objective-function (yield) conditions occupy only a small region of the full search space.",
                "How to run the simulation:",
            ],
            bullets=[
                "1) Set the fixed catalyst slice and number of observed points.",
                "2) Choose initialization method (Random/LHS/Halton).",
                "3) Choose AF (EI/PI/LCB) and exploration settings (xi or kappa).",
                "4) Click 'Initialize/Re-run 1D GP Campaign'.",
                "5) Use 'Run Next BO Suggestion' to continue optimization step by step.",
            ],
            note="Track how the GP and AF behavior changes as more measured points are added.",
        )
    else:
        _render_info_card(
            title="Gaussian Process + Acquisition Functions: The Twin Engines Behind BO",
            paragraphs=[
                "The Gaussian Process (GP) provides posterior mean + uncertainty from observed experiments.",
                "The Acquisition Function (AF) transforms that posterior into a decision score for the next run.",
            ],
            bullets=[
                "1) Configure slice, initialization, AF, and noise assumptions below.",
                "2) Initialize the campaign, then add BO steps sequentially.",
                "3) Compare how EI, PI, and LCB alter suggested regions over time.",
            ],
            note="This section connects model assumptions to practical campaign behavior.",
        )

    if teach_mode != "Beginner":
        with st.expander("Advanced math: GP model + AF score formulas", expanded=False):
            st.markdown("**Intuition**")
            st.write("The GP estimates both expected objective function (yield) and uncertainty across conditions. AF converts those two outputs into one score for selecting the next run.")
            st.markdown("**Equations**")
            st.write("Noisy chemistry measurements:")
            st.latex(r"y_i = f(x_i) + \varepsilon_i,\quad \varepsilon_i \sim \mathcal{N}(0,\sigma_n^2)")
            st.write("GP prior over latent response:")
            st.latex(r"f(x) \sim \mathcal{GP}(m(x), k(x,x'))")
            st.write("Posterior mean and variance after t experiments:")
            st.latex(r"\mu_t(x)=k_t(x)^\top\left(K_t+\sigma_n^2 I\right)^{-1}y_t")
            st.latex(r"\sigma_t^2(x)=k(x,x)-k_t(x)^\top\left(K_t+\sigma_n^2 I\right)^{-1}k_t(x)")
            st.write("Displayed uncertainty band:")
            st.latex(r"\mu_t(x)\pm 1.96\,\sigma_t(x)")
            st.write("Kernel example (RBF):")
            st.latex(r"k_{\mathrm{RBF}}(x,x')=\sigma_f^2\exp\!\left(-\frac{(x-x')^2}{2l^2}\right)")
            st.write("Acquisition score formulas:")
            st.write("Let current best measured objective function (yield) be:")
            st.latex(r"y^+ = \max_{i\le t} y_i")
            st.write("Standardized improvement term:")
            st.latex(r"z(x)=\frac{\mu_t(x)-y^+ - \xi}{\sigma_t(x)+10^{-12}}")
            st.write("PI and EI formulas:")
            st.latex(r"\mathrm{PI}(x)=\Phi\!\left(z(x)\right)")
            st.latex(
                r"\mathrm{EI}(x)=\left(\mu_t(x)-y^+-\xi\right)\Phi(z)+\sigma_t(x)\phi(z)"
            )
            st.write("LCB/UCB-style exploration score:")
            st.latex(r"\mathrm{UCB}(x)=\mu_t(x)+\kappa\,\sigma_t(x)")
            st.markdown("**Chemist interpretation**")
            st.markdown(
                """
- mu_t(x): expected objective function (yield) at condition x.
- sigma_t(x): epistemic uncertainty from limited observations.
- Larger xi or kappa increases exploration pressure.
"""
            )
            st.markdown("**Practical takeaway**")
            st.caption("In this app, optimization is implemented as minimization of negative objective function (yield); the plotted LCB score is sign-converted so larger score still means more preferred next experiment.")
            st.markdown("**Worked examples (step-by-step interpretation)**")
            st.caption("Shared setup: current best measured objective function (yield) is y+=80 and exploration margin is xi=1.")

            st.markdown("**Example A: high mean, moderate uncertainty**")
            st.latex(r"\mu=82,\ \sigma=4,\ y^+=80,\ \xi=1,\ z=\frac{82-80-1}{4}=0.25")
            st.latex(r"\mathrm{PI}\approx 0.60,\quad \mathrm{EI}\approx 2.14")
            st.markdown(
                """
- Mean prediction is above the current best (after exploration margin), so `z` is positive.
- `PI~0.60` means about a 60% chance to beat the improvement target.
- `EI~2.14` says the expected gain is solid, but uncertainty is only moderate.
"""
            )

            st.markdown("**Example B: lower mean, higher uncertainty**")
            st.latex(r"\mu=78,\ \sigma=9,\ y^+=80,\ \xi=1,\ z\approx\frac{-3}{9}\approx-0.33")
            st.latex(r"\mathrm{PI}\approx 0.37,\quad \mathrm{EI}\approx 2.28")
            st.markdown(
                """
- Mean prediction is below the target, so `z` is negative and `PI` drops.
- Large uncertainty (`sigma=9`) increases possible upside if this point is actually better than predicted.
- That is why `EI` can be higher than in Example A even though `PI` is lower.
"""
            )

            st.markdown("**Takeaway for decisions**")
            st.markdown(
                """
- Use `PI` when you want safer, near-term improvement.
- Use `EI` when you want a balance of improvement + uncertainty-driven upside.
- Increase `xi` (EI/PI) or `kappa` (LCB/UCB) to push BO toward exploration.
"""
            )

    widget_defaults = {
        "learn_gp_fixed_c_input": 0.6,
        "learn_gp_n_obs_input": 8,
        "learn_gp_init_input": "LHS",
        "learn_gp_noise_input": 1.0,
        "learn_gp_seed_input": 24,
        "learn_gp_res_input": 220,
        "learn_gp_af_input": "EI",
        "learn_gp_xi_input": 0.01,
        "learn_gp_kappa_input": 1.96,
    }
    for k, v in widget_defaults.items():
        st.session_state.setdefault(k, v)

    _render_info_card(
        title="Plot legend and controls",
        bullets=[
            "GP plot: blue line = GP mean, blue band = uncertainty, black points = measured experiments.",
            "AF plot: red line = acquisition score, black diamond = current best next-run suggestion.",
            "Controls let you test how initialization, AF choice, and exploration strength affect BO behavior.",
        ],
        card_style=LEARN_CARD_LEGEND,
    )
    with st.form("learn_gp_form"):
        gp_col1, gp_col2, gp_col3, gp_col4 = st.columns(4)
        with gp_col1:
            fixed_catalyst = st.slider(
                "Fixed catalyst for 1D slice",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="learn_gp_fixed_c_input",
                help="Sets the catalyst value while temperature varies in 1D. Think of this as a fixed reaction slice.",
            )
            n_obs = st.slider(
                "Observed points",
                min_value=4,
                max_value=25,
                key="learn_gp_n_obs_input",
                help="Number of initial experiments used to start the 1D BO campaign.",
            )
        with gp_col2:
            gp_init_label = st.selectbox(
                "1D init method",
                ["Random", "LHS", "Halton"],
                key="learn_gp_init_input",
                help="How initial training temperatures are distributed before sequential BO steps.",
            )
            noise_sigma = st.slider(
                "Measurement noise sigma",
                min_value=0.0,
                max_value=8.0,
                step=0.1,
                key="learn_gp_noise_input",
                help="Standard deviation of simulated measurement noise added to observed objective function (yield).",
            )
        with gp_col3:
            gp_seed = st.number_input(
                "1D GP seed",
                min_value=0,
                max_value=9999,
                key="learn_gp_seed_input",
                help="Random seed controlling reproducibility of initialization and simulated noise.",
            )
            gp_resolution = st.slider(
                "Plot resolution",
                min_value=80,
                max_value=500,
                step=20,
                key="learn_gp_res_input",
                help="Number of temperature grid points used for plotting GP mean, uncertainty, and AF.",
            )
        with gp_col4:
            gp_af = st.selectbox(
                "Acquisition Function",
                ["EI", "PI", "LCB"],
                key="learn_gp_af_input",
                help="Acquisition function used for AF visualization and sequential BO suggestions.",
            )
            gp_xi = st.slider(
                "xi (EI/PI)",
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                key="learn_gp_xi_input",
                help="Exploration margin for EI/PI. Higher xi encourages exploration.",
            )
            gp_kappa = st.slider(
                "kappa (LCB)",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key="learn_gp_kappa_input",
                help="Exploration weight for LCB. Higher kappa increases exploration.",
            )
        submitted = st.form_submit_button("Initialize/Re-run 1D GP Campaign")

    if submitted:
        try:
            st.session_state.learn_gp_campaign = _initialize_1d_campaign(
                fixed_catalyst=float(fixed_catalyst),
                n_obs=int(n_obs),
                init_method_label=str(gp_init_label),
                noise_sigma=float(noise_sigma),
                seed=int(gp_seed),
                gp_resolution=int(gp_resolution),
                gp_af=str(gp_af),
                gp_xi=float(gp_xi),
                gp_kappa=float(gp_kappa),
            )
        except Exception as ex:
            st.session_state.learn_gp_result = {
                "error": f"Could not initialize the 1D GP campaign. Technical detail: {ex}"
            }

    if "learn_gp_campaign" not in st.session_state:
        try:
            st.session_state.learn_gp_campaign = _initialize_1d_campaign(
                fixed_catalyst=float(st.session_state["learn_gp_fixed_c_input"]),
                n_obs=int(st.session_state["learn_gp_n_obs_input"]),
                init_method_label=str(st.session_state["learn_gp_init_input"]),
                noise_sigma=float(st.session_state["learn_gp_noise_input"]),
                seed=int(st.session_state["learn_gp_seed_input"]),
                gp_resolution=int(st.session_state["learn_gp_res_input"]),
                gp_af=str(st.session_state["learn_gp_af_input"]),
                gp_xi=float(st.session_state["learn_gp_xi_input"]),
                gp_kappa=float(st.session_state["learn_gp_kappa_input"]),
            )
        except Exception as ex:
            st.session_state.learn_gp_result = {
                "error": f"Could not initialize the default 1D GP campaign. Technical detail: {ex}"
            }

    run_col1, run_col2, run_col3 = st.columns(3)
    run_next = run_col1.button("Run Next BO Suggestion", key="learn_gp_campaign_step1")
    run_five = run_col2.button("Run 5 BO Suggestions", key="learn_gp_campaign_step5")
    reset_campaign = run_col3.button("Reset 1D Campaign", key="learn_gp_campaign_reset")

    if reset_campaign:
        try:
            st.session_state.learn_gp_campaign = _initialize_1d_campaign(
                fixed_catalyst=float(st.session_state["learn_gp_fixed_c_input"]),
                n_obs=int(st.session_state["learn_gp_n_obs_input"]),
                init_method_label=str(st.session_state["learn_gp_init_input"]),
                noise_sigma=float(st.session_state["learn_gp_noise_input"]),
                seed=int(st.session_state["learn_gp_seed_input"]),
                gp_resolution=int(st.session_state["learn_gp_res_input"]),
                gp_af=str(st.session_state["learn_gp_af_input"]),
                gp_xi=float(st.session_state["learn_gp_xi_input"]),
                gp_kappa=float(st.session_state["learn_gp_kappa_input"]),
            )
            st.session_state.learn_gp_result = _fit_1d_gp_result(st.session_state.learn_gp_campaign)
        except Exception as ex:
            st.session_state.learn_gp_result = {
                "error": f"Could not reset the 1D GP campaign. Technical detail: {ex}"
            }

    if run_next or run_five:
        campaign = st.session_state.get("learn_gp_campaign")
        if isinstance(campaign, dict):
            try:
                steps = 5 if run_five else 1
                st.session_state.learn_gp_campaign = _advance_1d_campaign(campaign, n_steps=steps)
            except Exception as ex:
                st.session_state.learn_gp_result = {
                    "error": f"Could not advance the 1D BO campaign. Technical detail: {ex}"
                }

    campaign = st.session_state.get("learn_gp_campaign")
    if isinstance(campaign, dict):
        try:
            st.session_state.learn_gp_result = _fit_1d_gp_result(campaign)
        except Exception as ex:
            st.session_state.learn_gp_result = {
                "error": f"Could not render the 1D GP plot for this configuration. Technical detail: {ex}"
            }

    gp_result = st.session_state.get("learn_gp_result", {})
    error = gp_result.get("error")
    if error:
        st.error(error)
    elif gp_result:
        df_obs = gp_result["df_obs"]
        t_grid = gp_result["t_grid"]
        true_grid = gp_result["true_grid"]
        mean_y = gp_result["mean_y"]
        lo_y = gp_result["lo_y"]
        hi_y = gp_result["hi_y"]
        near_std = gp_result["near_std"]
        far_std = gp_result["far_std"]
        fixed_c = gp_result["fixed_catalyst"]
        noise_used = gp_result["noise_sigma"]
        af_choice = gp_result.get("af_choice", "EI")
        af_values = np.asarray(gp_result.get("af_values", np.zeros_like(t_grid)), dtype=float)
        af_values_display = np.asarray(gp_result.get("af_values_display", af_values), dtype=float)
        af_argmax = int(gp_result.get("af_argmax", int(np.argmax(af_values)) if af_values.size else 0))
        af_peak_t = float(gp_result.get("af_peak_temperature", t_grid[af_argmax] if af_values.size else t_grid[0]))
        af_peak_score = float(gp_result.get("af_peak_score", af_values[af_argmax] if af_values.size else 0.0))
        af_xi = float(gp_result.get("af_xi", 0.01))
        af_kappa = float(gp_result.get("af_kappa", 1.96))
        step_count = int(gp_result.get("step_count", 0))

        st.caption(
            f"Campaign status: {len(df_obs)} unique temperatures observed, "
            f"{step_count} BO step(s) added after initialization."
        )

        fig_gp = go.Figure()
        fig_gp.add_trace(
            go.Scatter(
                x=t_grid,
                y=true_grid,
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="True response",
            )
        )
        fig_gp.add_trace(
            go.Scatter(
                x=t_grid,
                y=mean_y,
                mode="lines",
                line=dict(color="#1f77b4", width=3),
                name="GP mean",
            )
        )
        fig_gp.add_trace(
            go.Scatter(
                x=np.concatenate([t_grid, t_grid[::-1]]),
                y=np.concatenate([hi_y, lo_y[::-1]]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% uncertainty",
                hoverinfo="skip",
            )
        )
        fig_gp.add_trace(
            go.Scatter(
                x=df_obs["Temperature"],
                y=df_obs["MeasuredYield"],
                mode="markers",
                marker=dict(color="black", size=8),
                name="Observed experiments",
            )
        )
        fig_gp.update_layout(
            title=f"1D GP at fixed Catalyst={fixed_c:.2f}",
            xaxis_title="Temperature (C)",
            yaxis_title="Objective function (yield)",
            height=460,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_gp, use_container_width=True)

        fig_af = go.Figure()
        fig_af.add_trace(
            go.Scatter(
                x=t_grid,
                y=af_values_display,
                mode="lines",
                connectgaps=True,
                line=dict(color="#d62728", width=3),
                name=f"{af_choice} score",
            )
        )
        fig_af.add_vline(x=af_peak_t, line=dict(color="black", dash="dash"))
        fig_af.add_trace(
            go.Scatter(
                x=[af_peak_t],
                y=[af_peak_score],
                mode="markers",
                marker=dict(color="black", size=9, symbol="diamond"),
                name="AF peak",
            )
        )
        fig_af.update_layout(
            title=f"Acquisition Function ({af_choice}) for this GP",
            xaxis_title="Temperature (C)",
            yaxis_title="Acquisition score",
            height=320,
            margin=dict(l=20, r=20, t=45, b=20),
            yaxis=dict(
                exponentformat="e",
                showexponent="all",
                tickformat=".2e",
            ),
        )
        st.plotly_chart(fig_af, use_container_width=True)

        best_obs_idx = int(np.argmax(df_obs["MeasuredYield"].to_numpy())) if len(df_obs) else 0
        best_obs_t = float(df_obs.iloc[best_obs_idx]["Temperature"]) if len(df_obs) else float("nan")
        best_obs_y = float(df_obs.iloc[best_obs_idx]["MeasuredYield"]) if len(df_obs) else float("nan")
        explanation_bullets = [
            f"Current acquisition strategy: {af_choice}. Suggested next run is near Temperature={af_peak_t:.2f} C.",
            f"Current uncertainty snapshot: near sampled temperatures ~{near_std:.2f}, far from sampled temperatures ~{far_std:.2f}.",
            f"Best measured objective so far: ~{best_obs_y:.2f} at Temperature={best_obs_t:.2f} C.",
        ]

        _render_info_card(
            title="What changed after each BO step",
            paragraphs=[
                "Each new experiment updates both plots. The model confidence usually increases near sampled points, and the AF peak can move to a different temperature as the campaign learns.",
                "Use this snapshot to connect your parameter choices (init method, AF, xi/kappa) with BO behavior.",
            ],
            bullets=explanation_bullets,
            card_style=LEARN_CARD_STEP,
        )

    deep_mode_note = (
        "For deeper equations, AF scoring formulas (EI/PI/LCB), and stronger mathematical interpretation, switch to Advanced mode in this same module."
        if teach_mode == "Beginner"
        else "You are in Advanced mode, where this same intuition is connected to the full mathematical BO formulation."
    )
    st.markdown(
        (
            "<div style='background:linear-gradient(180deg,#f5f3ff 0%,#ede9fe 100%); border:1px solid #c4b5fd; border-left:6px solid #7c3aed; border-radius:12px; "
            "padding:14px 16px; margin:14px 0 8px 0; color:#000000; line-height:1.7;'>"
            "<div style='font-weight:700; margin-bottom:6px;'>Take-home messages from this section</div>"
            "<ul style='margin-top:0; margin-bottom:8px; padding-left:20px;'>"
            "<li>How a Gaussian Process models expected objective function (yield) and uncertainty from measured experiments.</li>"
            "<li>How acquisition functions use that model to propose the next experiment.</li>"
            "<li>Why BO suggestions change as new data is added step by step.</li>"
            "</ul>"
            f"<div>{html.escape(deep_mode_note)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    _render_info_card(
        title="Checkpoint Quiz (2 minutes)",
        paragraphs=[
            "Use this quick check to confirm your BO intuition before moving to the next section.",
            "Focus on the roles of the Gaussian Process, Acquisition Function, and initialization strategy.",
        ],
    )
    with st.form("learn_checkpoint_quiz_form"):
        q1 = st.radio(
            "1) What is the main job of the Gaussian Process (GP) in this module?",
            [
                "Choose the next experiment directly without scoring candidates.",
                "Estimate objective-function (yield) mean and uncertainty across conditions.",
                "Guarantee the globally optimal condition after initialization.",
            ],
            key="learn_quiz_q1",
        )
        q2 = st.radio(
            "2) What is the main job of the Acquisition Function (AF)?",
            [
                "Remove measurement noise from all observed data.",
                "Replace the need for initialization experiments.",
                "Convert GP predictions into a score to select the next experiment.",
            ],
            key="learn_quiz_q2",
        )
        q3 = st.radio(
            "3) In general, what happens when you increase xi (EI/PI) or kappa (LCB/UCB)?",
            [
                "BO ignores uncertainty and only picks highest current mean.",
                "BO becomes more exploratory and samples more uncertain regions.",
                "BO stops updating after each new experiment.",
            ],
            key="learn_quiz_q3",
        )
        q4 = st.radio(
            "4) Why do initialization experiments matter?",
            [
                "They are only needed for plotting and do not affect BO decisions.",
                "They make AF choice irrelevant for the rest of the campaign.",
                "They improve early coverage of the search space and reduce blind spots.",
            ],
            key="learn_quiz_q4",
        )
        quiz_submitted = st.form_submit_button("Check my answers")

    if quiz_submitted:
        answer_key = {
            "q1": "Estimate objective-function (yield) mean and uncertainty across conditions.",
            "q2": "Convert GP predictions into a score to select the next experiment.",
            "q3": "BO becomes more exploratory and samples more uncertain regions.",
            "q4": "They improve early coverage of the search space and reduce blind spots.",
        }
        answers = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
        total = len(answer_key)
        score = sum(1 for k, v in answer_key.items() if answers.get(k) == v)
        st.metric("Checkpoint score", f"{score}/{total}")
        st.progress(score / total)

        if score == total:
            st.success("Strong understanding. You are ready to move to BO Mechanics.")
        elif score >= 3:
            st.info("Good understanding. Review the GP/AF interpretation card once, then continue.")
        else:
            st.warning("Review the GP + AF cards and run a few more BO steps, then retry this checkpoint.")

        with st.expander("Answer key", expanded=False):
            st.markdown(
                """
1) GP: estimate objective-function (yield) mean and uncertainty.
2) AF: convert GP output into a next-experiment decision score.
3) Larger xi/kappa: typically more exploration.
4) Better initialization: better early coverage and more reliable BO behavior.
"""
            )


def _module_mechanics(teach_mode: str) -> None:
    st.subheader("3) Understand BO Mechanics")
    _render_info_card(
        title="Mechanics focus: initialization quality + acquisition behavior",
        paragraphs=["Compare initialization design and acquisition behavior on the same chemistry landscape."],
        note="Goal: understand why BO suggests certain points and how your settings change that behavior.",
    )
    with st.expander("Theory: why initialization and acquisition both matter", expanded=False):
        st.markdown(
            """
Initialization controls what the model knows at the beginning:
- Better spread (e.g., LHS, Halton) gives a less biased first surrogate.
- Clumped points create blind spots and unreliable uncertainty.

Acquisition controls what BO does next:
- EI balances expected gain and uncertainty.
- PI prefers high-probability local improvements.
- LCB tends to sample farther from known points to reduce uncertainty.
"""
        )
    _render_info_card(
        title="From Intuition to Real Optimization Campaigns",
        paragraphs=[
            "In the Intuition section, you learned how the Gaussian Process and Acquisition Function work together to define the next experiment to run.",
            "That 1D view fixed one variable to simplify interpretation, but real chemistry campaigns typically optimize multiple variables simultaneously with explicit bounds that define the search space.",
            "In this section, we focus on critical campaign parameters and how they affect optimization of the objective function (yield), especially the number of experiments required to approach the global optimum in the defined space.",
        ],
        note="Goal: understand which settings accelerate convergence and which settings waste budget.",
    )
    _render_info_card(
        title="How to run this Mechanics section",
        paragraphs=[
            "Use this sequence to test how campaign settings affect optimization speed and final performance.",
        ],
        bullets=[
            "1) Choose the number of initial experiments (n_init).",
            "2) Select an initialization method (Random/LHS/Halton/Maximin LHS).",
            "3) Select an acquisition function (EI/PI/LCB) and exploration setting (xi or kappa).",
            "4) Optionally set measurement noise (sigma) to mimic lab variability.",
            "5) Click 'Initialize/Re-run Mechanics' to start the campaign.",
            "6) Use 'Run Next BO Suggestion' (or 5 steps) to continue until strong results are reached.",
            "7) Click 'Save Current Campaign for Comparison' to store this run.",
            "8) Repeat with different settings and compare saved runs below the graph.",
        ],
        note="Compare different settings and identify which parameter combinations reach high yield with fewer observations.",
        card_style=INFO_CARD_STYLE_BLUE,
    )
    if teach_mode != "Beginner":
        with st.expander("Advanced math: initialization geometry + acquisition scoring", expanded=False):
            st.markdown("**Intuition**")
            st.write("Strong BO campaigns start with broad initial coverage, then use acquisition scoring to choose informative next runs.")
            st.markdown("**Equations**")
            st.write("Initial design matrix in d variables:")
            st.latex(r"X_0 = [x_1,\ldots,x_{n_{\mathrm{init}}}]^\top \in \mathbb{R}^{n_{\mathrm{init}}\times d}")
            st.write("Maximin design objective (space-filling):")
            st.latex(r"\max_{X_0}\ \min_{i\neq j}\|x_i-x_j\|_2")
            st.write("BO chooses:")
            st.latex(r"x_{t+1}=\arg\max_x a_t(x)")
            st.write("For objective function (yield) maximization, common choices are:")
            st.latex(r"\mathrm{PI}(x)=\Phi\!\left(\frac{\mu_t(x)-y^+-\xi}{\sigma_t(x)+10^{-12}}\right)")
            st.latex(
                r"\mathrm{EI}(x)=\left(\mu_t(x)-y^+-\xi\right)\Phi(z)+\sigma_t(x)\phi(z),\ z=\frac{\mu_t(x)-y^+-\xi}{\sigma_t(x)+10^{-12}}"
            )
            st.latex(r"\mathrm{UCB}(x)=\mu_t(x)+\kappa\,\sigma_t(x)")
            st.write("Minimum pairwise distance:")
            st.latex(r"d_{\min}=\min_{i\neq j}\|x_i-x_j\|_2")
            st.write("Mean nearest-neighbor distance:")
            st.latex(r"\bar d_{\mathrm{NN}}=\frac{1}{n}\sum_{i=1}^{n}\min_{j\neq i}\|x_i-x_j\|_2")
            st.markdown("**Chemist interpretation**")
            st.write("Broader initial spacing reduces blind spots, and acquisition hyperparameters xi/kappa control risk appetite in the next run choice.")
            st.markdown("**Practical takeaway**")
            st.caption(
                "Larger spread metrics generally indicate better initial coverage and more reliable uncertainty estimates."
            )

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_init = st.slider("Initial experiments", min_value=3, max_value=30, value=8, key="classroom_mech_n_init")
        with col2:
            init_method_label = st.selectbox("Init method", ["Random", "LHS", "Halton", "Maximin LHS"], key="classroom_mech_init")
        with col3:
            acq = st.selectbox("Acquisition", ["EI", "PI", "LCB"], index=0, key="classroom_mech_acq")
        with col4:
            seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, key="classroom_mech_seed")

        col5, col6, col7 = st.columns(3)
        with col5:
            mech_xi = st.number_input(
                "Exploration xi (EI/PI)",
                min_value=0.0,
                max_value=0.5,
                value=0.01,
                step=0.01,
                key="classroom_mech_xi",
                help="Used by EI/PI. Larger xi increases exploration pressure.",
            )
        with col6:
            mech_kappa = st.number_input(
                "Exploration kappa (LCB)",
                min_value=0.1,
                max_value=10.0,
                value=1.96,
                step=0.1,
                key="classroom_mech_kappa",
                help="Used by LCB. Larger kappa increases exploration pressure.",
            )
        with col7:
            mech_noise_sigma = st.number_input(
                "Measurement noise (sigma)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                key="classroom_mech_noise",
                help="Adds Gaussian measurement noise to observed objective-function values.",
            )

        rec_low, rec_high, rec_text = recommend_n_init_range(2, total_budget=None, mixed=False, multiobjective=False)
        rec_range_text = format_n_init_range(rec_low, rec_high)
        st.caption(f"n_init guidance: {rec_text}")
        if int(n_init) < rec_low or int(n_init) > rec_high:
            st.info(
                f"Current n_init={int(n_init)} is outside the suggested range ({rec_range_text}) "
                "for this 2-variable mechanics demo."
            )

    if "classroom_mech_history" not in st.session_state:
        st.session_state.classroom_mech_history = []
    if "classroom_mech_compare_counter" not in st.session_state:
        st.session_state.classroom_mech_compare_counter = 1

    campaign = st.session_state.get("classroom_mech_campaign")
    if isinstance(campaign, dict):
        st.session_state.classroom_mech_df = campaign.get("df")
        st.session_state.classroom_mech_next = campaign.get("x_next")

    campaign_exists = isinstance(st.session_state.get("classroom_mech_campaign"), dict)
    init_campaign = False
    if not campaign_exists:
        init_campaign = st.button("Initialize/Re-run Mechanics", key="classroom_mech_run")

    if init_campaign:
        try:
            st.session_state.classroom_mech_campaign = _initialize_mechanics_campaign(
                n_init=int(n_init),
                init_method_label=str(init_method_label),
                acq=str(acq),
                seed=int(seed),
                acq_xi=float(mech_xi),
                acq_kappa=float(mech_kappa),
                noise_sigma=float(mech_noise_sigma),
            )
        except Exception as ex:
            st.error(f"Could not initialize mechanics campaign. Technical detail: {ex}")

    campaign = st.session_state.get("classroom_mech_campaign")
    if isinstance(campaign, dict):
        st.session_state.classroom_mech_df = campaign.get("df")
        st.session_state.classroom_mech_next = campaign.get("x_next")

    df = st.session_state.get("classroom_mech_df")
    x_next = st.session_state.get("classroom_mech_next")
    if isinstance(df, pd.DataFrame) and not df.empty and x_next is not None:
        run_col1, run_col2, run_col3, run_col4 = st.columns(4)
        init_campaign_after = run_col1.button("Initialize/Re-run Mechanics", key="classroom_mech_run_after_table")
        run_next_after = run_col2.button("Run Next BO Suggestion", key="classroom_mech_step1_after_table")
        run_five_after = run_col3.button("Run 5 BO Suggestions", key="classroom_mech_step5_after_table")
        reset_campaign_after = run_col4.button("Reset Mechanics", key="classroom_mech_reset_after_table")

        if init_campaign_after:
            try:
                st.session_state.classroom_mech_campaign = _initialize_mechanics_campaign(
                    n_init=int(n_init),
                    init_method_label=str(init_method_label),
                    acq=str(acq),
                    seed=int(seed),
                    acq_xi=float(mech_xi),
                    acq_kappa=float(mech_kappa),
                    noise_sigma=float(mech_noise_sigma),
                )
            except Exception as ex:
                st.error(f"Could not initialize mechanics campaign. Technical detail: {ex}")
        if reset_campaign_after:
            st.session_state.pop("classroom_mech_campaign", None)
            st.session_state.pop("classroom_mech_df", None)
            st.session_state.pop("classroom_mech_next", None)
            st.info("Mechanics campaign reset. Initialize again to continue.")
            return
        if run_next_after or run_five_after:
            campaign_after = st.session_state.get("classroom_mech_campaign")
            if isinstance(campaign_after, dict):
                try:
                    steps_after = 5 if run_five_after else 1
                    st.session_state.classroom_mech_campaign = _advance_mechanics_campaign(campaign_after, n_steps=steps_after)
                except Exception as ex:
                    st.error(f"Could not advance mechanics campaign. Technical detail: {ex}")

        campaign = st.session_state.get("classroom_mech_campaign")
        if isinstance(campaign, dict):
            st.session_state.classroom_mech_df = campaign.get("df")
            st.session_state.classroom_mech_next = campaign.get("x_next")
            df = st.session_state.get("classroom_mech_df")
            x_next = st.session_state.get("classroom_mech_next")
            if not (isinstance(df, pd.DataFrame) and not df.empty and x_next is not None):
                return

        st.plotly_chart(
            _surface_figure(
                "Mechanics Campaign Coverage + Suggested Next Run",
                points=df,
                suggest_point=(x_next[0], x_next[1]),
            ),
            use_container_width=True,
        )

        st.success(
            "Suggested next run: "
            f"Temperature={x_next[0]:.2f}, Catalyst={x_next[1]:.3f}, "
            f"expected synthetic objective function (yield) around {x_next[2]:.2f}."
        )

        save_col, clear_col = st.columns(2)
        save_snapshot = save_col.button("Save Current Campaign for Comparison", key="classroom_mech_save_snapshot")
        clear_snapshots = clear_col.button("Clear Saved Comparisons", key="classroom_mech_clear_snapshots")

        if save_snapshot:
            campaign_now = st.session_state.get("classroom_mech_campaign")
            if isinstance(campaign_now, dict):
                snap_id = int(st.session_state.get("classroom_mech_compare_counter", 1))
                snap_label = f"Run {snap_id}"
                snapshot = _summarize_mechanics_campaign(campaign_now, snapshot_label=snap_label)
                st.session_state.classroom_mech_history = [
                    *st.session_state.get("classroom_mech_history", []),
                    snapshot,
                ]
                st.session_state.classroom_mech_compare_counter = snap_id + 1
                st.success(f"Saved {snap_label} for comparison.")
            else:
                st.warning("No active mechanics campaign available to save.")

        if clear_snapshots:
            st.session_state.classroom_mech_history = []
            st.session_state.classroom_mech_compare_counter = 1
            st.info("Saved mechanics comparisons were cleared.")

        history = st.session_state.get("classroom_mech_history", [])
        if isinstance(history, list) and history:
            history_df = pd.DataFrame(history)
            st.markdown("#### Campaign Comparison")
            st.dataframe(history_df, use_container_width=True)

            fig_compare = px.scatter(
                history_df,
                x="Observations",
                y="Best Yield",
                color="Acquisition",
                symbol="Init method",
                hover_data=[
                    "Snapshot",
                    "n_init",
                    "BO steps",
                    "Obs to Yield >= 90",
                    "xi",
                    "kappa",
                    "Noise sigma",
                    "Seed",
                ],
                text="Snapshot",
                title="Saved campaigns: settings impact on achieved yield",
            )
            fig_compare.update_traces(marker=dict(size=11, line=dict(width=1, color="black")), textposition="top center")
            fig_compare.update_layout(height=370, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_compare, use_container_width=True)

            history_eval = history_df.copy()
            history_eval["Best Yield_num"] = pd.to_numeric(history_eval.get("Best Yield"), errors="coerce")
            history_eval["Observations_num"] = pd.to_numeric(history_eval.get("Observations"), errors="coerce")
            valid_best = history_eval.dropna(subset=["Best Yield_num"])
            if not valid_best.empty:
                best_idx = valid_best.sort_values(
                    ["Best Yield_num", "Observations_num"],
                    ascending=[False, True],
                ).index[0]
                best_row = history_df.loc[best_idx]
                _render_info_card(
                    title="Best Saved Run So Far",
                    paragraphs=[
                        f"{best_row['Snapshot']} currently leads with best yield={best_row['Best Yield']}, "
                        f"reached with {best_row['Observations']} observations ({best_row['BO steps']} BO steps).",
                        f"Settings: n_init={best_row['n_init']}, init={best_row['Init method']}, "
                        f"acquisition={best_row['Acquisition']}, xi={best_row['xi']}, "
                        f"kappa={best_row['kappa']}, noise sigma={best_row['Noise sigma']}.",
                    ],
                    note="Use this as your benchmark while testing new parameter combinations.",
                    card_style=INFO_CARD_STYLE_BLUE,
                )
    else:
        st.info("Initialize mechanics campaign to generate initial points and start visualization.")

    _render_info_card(
        title="Mechanics Checkpoint Quiz",
        paragraphs=[
            "Quick self-check before moving forward: can you explain how mechanics settings change campaign efficiency?",
        ],
    )
    with st.form("mechanics_checkpoint_quiz_form"):
        mq1 = st.radio(
            "1) Why does n_init matter in BO mechanics?",
            [
                "It only changes plot appearance, not BO behavior.",
                "It controls early coverage quality and affects the surrogate model reliability.",
                "It guarantees the global optimum is already sampled.",
            ],
            key="mechanics_quiz_q1",
        )
        mq2 = st.radio(
            "2) Which statement about AF choice is most accurate?",
            [
                "EI, PI, and LCB are identical once enough points are observed.",
                "AF only affects initialization, not sequential BO suggestions.",
                "AF changes how BO trades off exploiting strong regions vs exploring uncertainty.",
            ],
            key="mechanics_quiz_q2",
        )
        mq3 = st.radio(
            "3) In GP-based BO, why does predictive uncertainty appear in acquisition functions?",
            [
                "Because BO needs uncertainty to trade off information gain against immediate reward.",
                "Because uncertainty directly equals experimental measurement noise in all cases.",
                "Because uncertainty is only used to scale the plotting axis, not decisions.",
            ],
            key="mechanics_quiz_q3",
        )
        mq4 = st.radio(
            "4) Conceptually, what does Expected Improvement (EI) measure at a candidate point?",
            [
                "The expected amount by which the point could improve over the current best, accounting for uncertainty.",
                "The probability that the point is globally optimal with zero model error.",
                "The deterministic gradient magnitude of the true objective function.",
            ],
            key="mechanics_quiz_q4",
        )
        mechanics_quiz_submitted = st.form_submit_button("Check mechanics answers")

    if mechanics_quiz_submitted:
        mechanics_answer_key = {
            "q1": "It controls early coverage quality and affects the surrogate model reliability.",
            "q2": "AF changes how BO trades off exploiting strong regions vs exploring uncertainty.",
            "q3": "Because BO needs uncertainty to trade off information gain against immediate reward.",
            "q4": "The expected amount by which the point could improve over the current best, accounting for uncertainty.",
        }
        mechanics_answers = {"q1": mq1, "q2": mq2, "q3": mq3, "q4": mq4}
        mechanics_total = len(mechanics_answer_key)
        mechanics_score = sum(1 for k, v in mechanics_answer_key.items() if mechanics_answers.get(k) == v)

        st.metric("Mechanics checkpoint score", f"{mechanics_score}/{mechanics_total}")
        st.progress(mechanics_score / mechanics_total)
        if mechanics_score == mechanics_total:
            st.success("Excellent. You are ready for Chemist Workflow.")
        elif mechanics_score >= 3:
            st.info("Good result. You can continue, but one quick review of parameter effects will help.")
        else:
            st.warning("Review the Mechanics cards and comparison table, then retry the checkpoint.")

        with st.expander("Mechanics answer key", expanded=False):
            st.markdown(
                """
1) n_init: controls early coverage quality and surrogate reliability.
2) AF: sets exploration vs exploitation behavior.
3) Uncertainty term: enables exploration vs exploitation tradeoff.
4) EI: expected improvement over current best, weighted by uncertainty.
"""
            )


def _workflow_flow_yield(
    temperature: float,
    catalyst: float,
    pressure: float,
    residence_time_s: float,
) -> float:
    # Synthetic 4-variable flow-reaction yield with a high-performing region near ~90%+.
    t = (float(temperature) - 92.0) / 16.0
    c = (float(catalyst) - 0.62) / 0.14
    p = (float(pressure) - 12.0) / 4.5
    r = (float(residence_time_s) - 180.0) / 55.0

    base = np.exp(-(t**2)) * np.exp(-(c**2)) * np.exp(-0.7 * (p**2)) * np.exp(-0.9 * (r**2))
    tc_synergy = 0.16 * np.exp(-((t - 0.75 * c) ** 2))
    pr_balance = 0.11 * np.exp(-((p + 0.35 * r) ** 2))
    residence_ridge = 0.09 * np.exp(-((r - 0.30 * t) ** 2))

    y = 100.0 * (0.58 * base + tc_synergy + pr_balance + residence_ridge)

    # Mild practical penalties to avoid unrealistic extremes.
    if catalyst > 0.9:
        y -= 6.0
    if pressure > 18.0:
        y -= 4.0
    if residence_time_s < 45.0:
        y -= 5.0

    return float(np.clip(y, 0.0, 100.0))


def _workflow_productivity(
    yield_pct: float,
    residence_time_s: float,
    pressure: float,
    catalyst: float,
) -> float:
    """
    Synthetic flow productivity proxy:
    - Improves with conversion (yield),
    - Improves with shorter residence time (higher throughput),
    - Includes mild pressure/catalyst dependence.
    """
    y_frac = float(np.clip(yield_pct / 100.0, 0.0, 1.0))
    rt = float(np.clip(residence_time_s, 30.0, 300.0))
    throughput = 300.0 / rt  # 1..10 across configured residence-time window
    pressure_factor = 0.90 + 0.20 * (float(pressure) / 20.0)
    catalyst_factor = 0.85 + 0.25 * np.exp(-((float(catalyst) - 0.55) ** 2) / 0.12)
    prod = 10.0 * y_frac * throughput * pressure_factor * catalyst_factor
    return float(np.clip(prod, 0.0, 120.0))


def _module_workflow(teach_mode: str) -> None:
    st.subheader("4) Chemist Workflow")
    with st.expander("Theory: why real lab campaigns look messy", expanded=False):
        st.markdown(
            """
Real campaigns differ from ideal optimization because:
- Measurements are noisy (instrument variation, handling, batch effects).
- Some runs fail and return no usable result.
- Replicates are needed to estimate confidence around promising conditions.

BO still works by updating only on valid observations and continuously re-ranking where information value is highest.
"""
        )
    if teach_mode != "Beginner":
        with st.expander("Advanced math: noisy observations, failures, and replicates", expanded=False):
            st.markdown("**Intuition**")
            st.write("Real campaigns include noise and failed runs; BO should update only on valid data while replicates reduce decision risk.")
            st.markdown("**Equations**")
            st.latex(r"y_t = f(x_t) + \varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,\sigma_n^2)")
            st.write("Only successful runs are added to BO update:")
            st.latex(r"\mathcal{D}_t^{\mathrm{valid}}=\{(x_i,y_i)\ |\ \text{run }i\text{ not failed}\}")
            st.latex(r"x_{t+1}=\arg\max_x a_t\!\left(x;\mathcal{D}_t^{\mathrm{valid}}\right)")
            st.latex(r"m_t\sim\mathrm{Bernoulli}(p_{\mathrm{fail}}),\quad m_t=1\Rightarrow y_t\ \text{missing}")
            st.write("Observed success rate estimate:")
            st.latex(r"\hat p_{\mathrm{ok}}=\frac{N_{\mathrm{ok}}}{N_{\mathrm{ok}}+N_{\mathrm{fail}}}")
            st.write("For r replicate measurements at the same condition x:")
            st.latex(r"\bar y(x)=\frac{1}{r}\sum_{j=1}^r y_j(x)")
            st.latex(r"s^2(x)=\frac{1}{r-1}\sum_{j=1}^r\left(y_j(x)-\bar y(x)\right)^2")
            st.latex(r"\mathrm{SE}(\bar y)=\frac{s(x)}{\sqrt{r}}")
            st.markdown("**Chemist interpretation**")
            st.write("High noise or high failure rates reduce information per experiment; replicate statistics quantify confidence at promising conditions.")
            st.markdown("**Practical takeaway**")
            st.caption(
                "Chemistry interpretation: replicates reduce uncertainty about a promising condition before committing budget."
            )

    _render_info_card(
        title="Optimization target: increase the objective function (yield) of a flow chemical reaction",
        paragraphs=[
            "After learning how BO parameter selection changes behavior in a 2D search space, we now move to a 4D search space that is closer to a real chemistry campaign.",
            "As dimensionality increases, the search space grows and campaign design becomes more important, often requiring more experiments to approach the best conditions.",
            "Simulate a realistic 4-variable campaign with noise, failed runs, and optional replicate checks.",
            "We optimize four operational variables: Temperature, Catalyst loading, Pressure, and Residence Time.",
        ],
        bullets=[
            "Temperature: 20 to 120 C",
            "Catalyst loading: 0.00 to 1.00 (fraction)",
            "Pressure: 1 to 20 bar",
            "Residence Time: 30 to 300 s",
        ],
        note="This simulator is calibrated to produce a realistic high-performing region (best runs around ~90%+).",
        mission_title="Your mission:",
        mission_text=(
            "Reach the highest possible objective function (yield) using the minimum number of iterations. "
            "Tune AF, total iterations, n_init, measurement noise, failure probability, and replicate frequency. "
            "Defaults are intentionally challenging so improvements are visible."
        ),
    )

    preset_total = int(st.session_state.get("wf_total_iters", 12))
    preset_n_init = int(st.session_state.get("wf_n_init", 4))
    preset_noise = float(st.session_state.get("wf_noise", 4.0))
    preset_fail = float(st.session_state.get("wf_fail", 0.20))
    preset_repl = int(st.session_state.get("wf_replicate_every", 0))

    c1, c2, c3 = st.columns(3)
    with c1:
        total_iters = st.number_input(
            "Total iterations",
            min_value=5,
            max_value=120,
            value=preset_total,
            key="wf_total_iters",
            help="Total experiment budget for the campaign. Higher values give BO more chances to improve the objective function (yield).",
        )
        n_init = st.number_input(
            "Initial experiments",
            min_value=4,
            max_value=60,
            value=preset_n_init,
            key="wf_n_init",
            help="Number of design points run before BO starts sequential suggestions. Too low can bias the model; too high spends budget early.",
        )
    with c2:
        init_label = st.selectbox(
            "Init method",
            ["Random", "LHS", "Halton", "Maximin LHS"],
            index=0,
            key="wf_init_method",
            help="Strategy for placing initial experiments. LHS/Halton/Maximin LHS usually provide broader early coverage than pure random.",
        )
        acq = st.selectbox(
            "Acquisition",
            ["EI", "PI", "LCB"],
            index=1,
            key="wf_acq",
            help="Rule BO uses to pick the next experiment. EI is balanced, PI is greedier, LCB explores uncertainty more.",
        )
    with c3:
        noise_sigma = st.number_input(
            "Measurement noise (sigma)",
            min_value=0.0,
            max_value=10.0,
            value=preset_noise,
            step=0.1,
            key="wf_noise",
            help="Standard deviation of measurement error added to the true objective function (yield). Higher noise makes learning harder and increases uncertainty.",
        )
        failure_prob = st.number_input(
            "Failure probability",
            min_value=0.0,
            max_value=0.9,
            value=preset_fail,
            step=0.01,
            key="wf_fail",
            help="Chance that a run fails and returns no usable measurement. Failed runs consume iteration budget but do not update BO.",
        )

    rec_low, rec_high, rec_text = recommend_n_init_range(
        4,
        total_budget=int(total_iters),
        noisy=float(noise_sigma) > 1.0,
        mixed=False,
        multiobjective=False,
    )
    rec_range_text = format_n_init_range(rec_low, rec_high)
    st.caption(f"n_init guidance: {rec_text}")
    if int(n_init) < rec_low or int(n_init) > rec_high:
        st.info(
            f"Current n_init={int(n_init)} is outside the suggested range ({rec_range_text}) "
            "for this 4-variable workflow under the selected budget/noise settings."
        )

    replicate_every = st.number_input(
        "Replicate best point every N runs (0 disables)",
        min_value=0,
        max_value=20,
        value=preset_repl,
        key="wf_replicate_every",
        help="Every N iterations, rerun the current best condition to verify robustness. Use 0 to disable replicates.",
    )
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=9999,
        value=123,
        key="wf_seed",
        help="Random seed for reproducibility (initial design, failure events, and noise).",
    )

    if st.button("Run simulated campaign", key="wf_run"):
        rng = np.random.default_rng(int(seed))
        space = [
            ("Temperature", 20.0, 120.0, "C", "continuous"),
            ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
            ("Pressure", 1.0, 20.0, "bar", "continuous"),
            ("ResidenceTimeSec", 30.0, 300.0, "s", "continuous"),
        ]
        dims = [
            Real(20.0, 120.0, name="Temperature"),
            Real(0.0, 1.0, name="Catalyst"),
            Real(1.0, 20.0, name="Pressure"),
            Real(30.0, 300.0, name="ResidenceTimeSec"),
        ]
        opt = safe_build_optimizer(dims, n_initial_points_remaining=0, acq_func=acq)

        init_method = init_label.lower().replace(" ", "_")
        x_init = generate_initial_points(space, int(n_init), method=init_method, seed=int(seed))

        records: list[dict] = []
        best_x: list[float] | None = None
        best_y = -np.inf

        for i in range(int(total_iters)):
            if i < len(x_init):
                x = list(x_init[i])
                source = "initial_design"
            else:
                if int(replicate_every) > 0 and best_x is not None and ((i + 1) % int(replicate_every) == 0):
                    x = list(best_x)
                    source = "replicate_best"
                else:
                    x = list(opt.suggest())
                    source = "bo_suggestion"

            true_y = _workflow_flow_yield(
                temperature=float(x[0]),
                catalyst=float(x[1]),
                pressure=float(x[2]),
                residence_time_s=float(x[3]),
            )
            failed = rng.random() < float(failure_prob)
            if failed:
                measured = np.nan
                status = "failed"
            else:
                measured = float(np.clip(true_y + rng.normal(0.0, float(noise_sigma)), 0.0, 100.0))
                status = "ok"
                opt.observe(x, -measured)
                if measured > best_y:
                    best_y = measured
                    best_x = list(x)

            records.append(
                {
                    "Iteration": i + 1,
                    "Temperature": float(x[0]),
                    "Catalyst": float(x[1]),
                    "Pressure": float(x[2]),
                    "ResidenceTimeSec": float(x[3]),
                    "TrueYield": true_y,
                    "MeasuredYield": measured,
                    "Status": status,
                    "Source": source,
                }
            )

        st.session_state.wf_results = pd.DataFrame(records)

    df = st.session_state.get("wf_results")
    if isinstance(df, pd.DataFrame) and not df.empty:
        n_ok = int((df["Status"] == "ok").sum())
        n_fail = int((df["Status"] == "failed").sum())
        best = pd.to_numeric(df["MeasuredYield"], errors="coerce").max()
        ok_df = df[df["Status"] == "ok"].copy()

        m1, m2, m3 = st.columns(3)
        m1.metric("Successful runs", n_ok)
        m2.metric("Failed runs", n_fail)
        m3.metric("Best measured objective function (yield)", f"{best:.2f}" if pd.notna(best) else "N/A")

        st.dataframe(df, use_container_width=True)
        _render_info_card(
            title="How to read the Workflow plots",
            bullets=[
                "Trend plot: tracks measured objective function (yield) across iterations and highlights failed runs.",
                "Parallel coordinates plot: each line is one successful 4D condition; color shows measured yield level.",
                "Look for repeated high-yield color patterns to identify robust operating regions, not only single best points.",
            ],
            note="Use these plots together to judge both optimization progress and parameter-pattern consistency.",
            card_style=INFO_CARD_STYLE_BLUE,
        )

        fig_trend = px.line(df, x="Iteration", y="MeasuredYield", color="Status", markers=True)
        fig_trend.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Measured objective function (yield)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        parallel_df = ok_df.copy()
        req_cols = ["Temperature", "Catalyst", "Pressure", "ResidenceTimeSec", "MeasuredYield"]
        for col in req_cols:
            parallel_df[col] = pd.to_numeric(parallel_df[col], errors="coerce")
        parallel_df = parallel_df.dropna(subset=req_cols)
        if not parallel_df.empty:
            fig_parallel = go.Figure(
                data=go.Parcoords(
                    domain=dict(x=[0.03, 1.0], y=[0.0, 1.0]),
                    line=dict(
                        color=parallel_df["MeasuredYield"],
                        colorscale=px.colors.sequential.Viridis[::-1],
                        showscale=True,
                        colorbar=dict(title="Measured objective function (yield)"),
                    ),
                    labelfont=dict(color="black", size=13),
                    tickfont=dict(color="black", size=14),
                    dimensions=[
                        dict(
                            label="Temperature (C)",
                            values=parallel_df["Temperature"],
                            range=[20.0, 120.0],
                            tickvals=[20, 40, 60, 80, 100, 120],
                        ),
                        dict(
                            label="Catalyst loading",
                            values=parallel_df["Catalyst"],
                            range=[0.0, 1.0],
                            tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
                        ),
                        dict(
                            label="Pressure (bar)",
                            values=parallel_df["Pressure"],
                            range=[1.0, 20.0],
                            tickvals=[1, 5, 10, 15, 20],
                        ),
                        dict(
                            label="Residence Time (s)",
                            values=parallel_df["ResidenceTimeSec"],
                            range=[30.0, 300.0],
                            tickvals=[30, 60, 120, 180, 240, 300],
                        ),
                        dict(
                            label="Measured objective function (yield)",
                            values=parallel_df["MeasuredYield"],
                            range=[0.0, 100.0],
                            tickvals=[0, 20, 40, 60, 80, 100],
                        ),
                    ],
                )
            )
            fig_parallel.update_layout(
                title=dict(
                    text="Explored 4D Conditions (Parallel Coordinates)",
                    x=0.01,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    pad=dict(t=36, b=10),
                ),
                height=430,
                margin=dict(l=56, r=20, t=112, b=20),
                paper_bgcolor="white",
                font=dict(color="black"),
            )
            st.plotly_chart(fig_parallel, use_container_width=True)
            st.markdown(
                "<div style='color:#000000;'>Each line is one successful flow-reaction experiment. "
                "Line color encodes the measured objective function (yield).</div>",
                unsafe_allow_html=True,
            )

def _compute_mo_dataset(n_points: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temps = rng.uniform(20.0, 120.0, size=n_points)
    cats = rng.uniform(0.0, 1.0, size=n_points)
    press = rng.uniform(1.0, 20.0, size=n_points)
    rt_s = rng.uniform(30.0, 300.0, size=n_points)

    rows = []
    for t, c, p, r in zip(temps, cats, press, rt_s):
        y = _workflow_flow_yield(
            temperature=float(t),
            catalyst=float(c),
            pressure=float(p),
            residence_time_s=float(r),
        )
        productivity = _workflow_productivity(
            yield_pct=float(y),
            residence_time_s=float(r),
            pressure=float(p),
            catalyst=float(c),
        )
        rows.append(
            {
                "Temperature": float(t),
                "Catalyst": float(c),
                "Pressure": float(p),
                "ResidenceTimeSec": float(r),
                "Yield": y,
                "Productivity": float(productivity),
            }
        )
    return pd.DataFrame(rows)


def _module_mo(teach_mode: str) -> None:
    st.subheader("5) Multiobjective Decisions")
    _render_info_card(
        title="Multiobjective decision goal",
        paragraphs=[
            "Use the same 4D flow-reaction setup as in Chemist Workflow.",
            "Explore the tradeoff between objective function (yield) and productivity.",
        ],
        note="There is no single universal best point; you choose a compromise based on project priorities.",
    )
    with st.expander("Theory: why there is no single best point in MO", expanded=False):
        st.markdown(
            r"""
A point is Pareto-optimal if no other point is better in all objectives simultaneously.

So multiobjective optimization returns a **front of tradeoffs**, not one universal optimum.
- Hypervolume: how much objective space is dominated by the front (larger is better).
- Knee point: region where small gain in one objective causes large loss in another.
- Weighted scoring is a decision policy, not a universal truth.
"""
        )
    if teach_mode != "Beginner":
        with st.expander("Advanced math: Pareto dominance, hypervolume, and weighted choice", expanded=False):
            st.markdown("**Intuition**")
            st.write("With multiple objectives, no single point is universally best; decisions come from tradeoffs along the Pareto set.")
            st.markdown("**Equations**")
            st.write("After converting all objectives to maximize form, point u dominates v if:")
            st.latex(r"u \succ v \iff \left[\forall k,\ u_k\ge v_k\right]\ \wedge\ \left[\exists k,\ u_k>v_k\right]")
            st.write("Pareto set:")
            st.latex(r"\mathcal{P}=\{x_i\ |\ \nexists x_j: f(x_j)\succ f(x_i)\}")
            st.write("Given a reference point r, hypervolume of Pareto front PF is:")
            st.latex(r"\mathrm{HV}(\mathrm{PF};r)=\lambda\!\left(\bigcup_{p\in \mathrm{PF}} [r_1,p_1]\times[r_2,p_2]\right)")
            st.write("Objectives are normalized before weighting:")
            st.latex(r"\tilde f_k(x)=\frac{f_k(x)-\min f_k}{\max f_k-\min f_k+10^{-12}}")
            st.latex(r"s(x)=\sum_{k=1}^{m} w_k\,\tilde f_k(x),\quad \sum_k w_k=1")
            st.write("Recommended point:")
            st.latex(r"x^\star=\arg\max_x s(x)")
            st.markdown("**Chemist interpretation**")
            st.write(
                "Hypervolume tracks frontier quality, while weighted scores encode project priorities such as maximizing objective function (yield) vs maximizing productivity."
            )
            st.markdown("**Practical takeaway**")
            st.caption(
                "Chemistry interpretation: changing weights changes business/science priorities, so recommendation is policy-dependent."
            )

    m1, m2, m3 = st.columns(3)
    with m1:
        mo_total_iters = st.number_input(
            "Total iterations",
            min_value=5,
            max_value=120,
            value=int(st.session_state.get("mo_total_iters", 30)),
            key="mo_total_iters",
        )
        mo_n_init = st.number_input(
            "Initial experiments",
            min_value=4,
            max_value=60,
            value=int(st.session_state.get("mo_n_init", 8)),
            key="mo_n_init",
        )
    with m2:
        mo_init_label = st.selectbox(
            "Init method",
            ["Random", "LHS", "Halton", "Maximin LHS"],
            index=1,
            key="mo_init_method",
        )
        mo_acq = st.selectbox(
            "Acquisition",
            ["EI", "PI", "LCB"],
            index=0,
            key="mo_acq",
        )
    with m3:
        mo_noise_sigma = st.number_input(
            "Measurement noise (sigma)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.get("mo_noise", 3.0)),
            step=0.1,
            key="mo_noise",
        )
        mo_failure_prob = st.number_input(
            "Failure probability",
            min_value=0.0,
            max_value=0.9,
            value=float(st.session_state.get("mo_fail", 0.15)),
            step=0.01,
            key="mo_fail",
        )

    mo_seed = int(st.session_state.get("mo_seed_internal", 99))
    rec_low, rec_high, rec_text = recommend_n_init_range(
        4,
        total_budget=int(mo_total_iters),
        noisy=float(mo_noise_sigma) > 1.0,
        mixed=False,
        multiobjective=True,
    )
    rec_range_text = format_n_init_range(rec_low, rec_high)
    st.caption(f"n_init guidance: {rec_text}")
    if int(mo_n_init) < rec_low or int(mo_n_init) > rec_high:
        st.info(
            f"Current n_init={int(mo_n_init)} is outside the suggested range ({rec_range_text}) "
            "for this 4-variable multiobjective setup under the selected budget/noise settings."
        )

    if st.button("Run simulated MO campaign", key="mo_generate"):
        rng = np.random.default_rng(mo_seed)
        space = [
            ("Temperature", 20.0, 120.0, "C", "continuous"),
            ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
            ("Pressure", 1.0, 20.0, "bar", "continuous"),
            ("ResidenceTimeSec", 30.0, 300.0, "s", "continuous"),
        ]
        dims = [
            Real(20.0, 120.0, name="Temperature"),
            Real(0.0, 1.0, name="Catalyst"),
            Real(1.0, 20.0, name="Pressure"),
            Real(30.0, 300.0, name="ResidenceTimeSec"),
        ]
        opt = safe_build_optimizer(dims, n_initial_points_remaining=0, acq_func=str(mo_acq))
        x_init = generate_initial_points(
            space,
            int(mo_n_init),
            method=str(mo_init_label).lower().replace(" ", "_"),
            seed=mo_seed,
        )

        rows: list[dict] = []
        for i in range(int(mo_total_iters)):
            if i < len(x_init):
                x = list(x_init[i])
                source = "initial_design"
            else:
                x = list(opt.suggest())
                source = "bo_suggestion"

            true_y = _workflow_flow_yield(
                temperature=float(x[0]),
                catalyst=float(x[1]),
                pressure=float(x[2]),
                residence_time_s=float(x[3]),
            )
            true_prod = _workflow_productivity(
                yield_pct=float(true_y),
                residence_time_s=float(x[3]),
                pressure=float(x[2]),
                catalyst=float(x[1]),
            )

            failed = rng.random() < float(mo_failure_prob)
            if failed:
                measured_y = np.nan
                measured_prod = np.nan
                status = "failed"
            else:
                measured_y = float(np.clip(true_y + rng.normal(0.0, float(mo_noise_sigma)), 0.0, 100.0))
                measured_prod = _workflow_productivity(
                    yield_pct=float(measured_y),
                    residence_time_s=float(x[3]),
                    pressure=float(x[2]),
                    catalyst=float(x[1]),
                )
                status = "ok"
                # Campaign is driven by BO on yield while MO analysis is performed on Yield vs Productivity.
                opt.observe(x, -float(measured_y))

            rows.append(
                {
                    "Iteration": i + 1,
                    "Temperature": float(x[0]),
                    "Catalyst": float(x[1]),
                    "Pressure": float(x[2]),
                    "ResidenceTimeSec": float(x[3]),
                    "TrueYield": float(true_y),
                    "Yield": measured_y,
                    "TrueProductivity": float(true_prod),
                    "Productivity": measured_prod,
                    "Status": status,
                    "Source": source,
                }
            )

        st.session_state.mo_classroom_df = pd.DataFrame(rows)

    df = st.session_state.get("mo_classroom_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Run a simulated MO campaign to start Pareto analysis.")
        return

    required_cols = {
        "Iteration",
        "Temperature",
        "Catalyst",
        "Pressure",
        "ResidenceTimeSec",
        "Yield",
        "Productivity",
        "Status",
    }
    if not required_cols.issubset(set(df.columns)):
        st.info("MO setup was updated. Run a new MO campaign to refresh this section.")
        return

    n_ok = int((df["Status"] == "ok").sum())
    n_fail = int((df["Status"] == "failed").sum())
    best_y = pd.to_numeric(df["Yield"], errors="coerce").max()
    best_prod = pd.to_numeric(df["Productivity"], errors="coerce").max()
    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("Successful runs", n_ok)
    mm2.metric("Failed runs", n_fail)
    mm3.metric("Best yield", f"{best_y:.2f}" if pd.notna(best_y) else "N/A")
    mm4.metric("Best productivity", f"{best_prod:.2f}" if pd.notna(best_prod) else "N/A")

    st.dataframe(df.head(30), use_container_width=True)

    mo_options = ["Yield", "Productivity"]
    mo_sel = st.session_state.get("mo_obj_select")
    if isinstance(mo_sel, list):
        filtered = [o for o in mo_sel if o in mo_options]
        if filtered != mo_sel:
            st.session_state["mo_obj_select"] = filtered if filtered else ["Yield", "Productivity"]
    objectives = st.multiselect(
        "Objectives to analyze",
        mo_options,
        default=["Yield", "Productivity"],
        key="mo_obj_select",
    )
    if len(objectives) < 2:
        st.warning("Select at least two objectives.")
        return

    df_ok = df[df["Status"] == "ok"].copy()
    df_ok = df_ok.dropna(subset=objectives)
    if df_ok.empty:
        st.warning("No valid successful runs available for Pareto analysis. Try lower failure probability or rerun.")
        return

    directions = {"Yield": "Maximize", "Productivity": "Maximize"}
    signs = np.array([1.0 if directions[o] == "Maximize" else -1.0 for o in objectives], dtype=float)
    pts = df_ok[objectives].to_numpy(dtype=float) * signs
    idx_pf = pareto_front_indices(pts)

    st.success(f"Pareto front size: {len(idx_pf)}")
    dominated_share = 1.0 - (len(idx_pf) / max(1, len(df_ok)))
    st.caption(
        f"{dominated_share:.1%} of candidates are dominated by better tradeoffs. "
        "This is why Pareto filtering is essential before final decision-making."
    )

    if len(objectives) == 2:
        df_plot = df_ok.reset_index(drop=True)
        pf_mask = df_plot.index.isin(idx_pf)
        fig = px.scatter(df_plot, x=objectives[0], y=objectives[1], color=pf_mask, labels={"color": "Pareto"})
        df_pf = df_plot.iloc[idx_pf].sort_values(by=objectives[0])
        fig.add_trace(
            go.Scatter(
                x=df_pf[objectives[0]],
                y=df_pf[objectives[1]],
                mode="lines+markers",
                line=dict(color="red", width=3),
                name="Pareto front",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        p_pf = (df_pf[objectives].to_numpy(dtype=float) * signs[:2]) if len(df_pf) else np.empty((0, 2))
        if p_pf.shape[0] >= 2:
            knee_idx = knee_index_2d(p_pf)
            ref = tuple((df_plot[objectives].to_numpy(dtype=float) * signs[:2]).min(axis=0))
            hv = hypervolume_2d(p_pf, ref)
            st.write(f"Approximate 2D hypervolume: {hv:.4g}")
            if knee_idx is not None:
                st.write(f"Approximate knee point index on sorted Pareto curve: {knee_idx}")
                knee_row = df_pf.iloc[knee_idx]
                st.caption(
                    "Knee interpretation: near this point, improving one objective starts to cost "
                    "disproportionately in the other."
                )
                st.write(
                    {
                        f"knee_{objectives[0]}": round(float(knee_row[objectives[0]]), 4),
                        f"knee_{objectives[1]}": round(float(knee_row[objectives[1]]), 4),
                    }
                )

            st.markdown("#### Weight-based recommendation")
            w1 = st.slider(f"Weight for {objectives[0]}", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="mo_weight_1")
            w2 = 1.0 - w1
            values = df_plot[objectives].to_numpy(dtype=float) * signs[:2]
            norm = (values - values.min(axis=0)) / (np.ptp(values, axis=0) + 1e-12)
            score = w1 * norm[:, 0] + w2 * norm[:, 1]
            best_i = int(np.argmax(score))
            best_row = df_plot.iloc[best_i]
            st.info(
                "Recommended compromise condition: "
                f"Temperature={best_row['Temperature']:.2f}, Catalyst={best_row['Catalyst']:.3f}, "
                f"Pressure={best_row['Pressure']:.2f}, ResidenceTimeSec={best_row['ResidenceTimeSec']:.1f}, "
                f"{objectives[0]}={best_row[objectives[0]]:.2f}, {objectives[1]}={best_row[objectives[1]]:.2f}"
            )
            st.caption(
                "Explanation: objectives are normalized to comparable scales first, then combined by your weights. "
                "Changing weights changes what 'best compromise' means."
            )
    else:
        fig3d = px.scatter_3d(df, x=objectives[0], y=objectives[1], z=objectives[2], color=df.index.isin(idx_pf), labels={"color": "Pareto"})
        st.plotly_chart(fig3d, use_container_width=True)


st.title("Bayesian Optimization Classroom")
st.caption("A guided, chemistry-first path for users who are new to Bayesian Optimization.")

st.sidebar.markdown("### Classroom Settings")
# Apply deferred module switch before the radio widget is instantiated.
pending_module = st.session_state.pop("classroom_module_pending", None)
if pending_module in MODULE_LABELS:
    st.session_state["classroom_module"] = pending_module
if st.session_state.get("classroom_module") not in MODULE_LABELS:
    st.session_state["classroom_module"] = MODULE_LABELS[0]
teach_mode = st.sidebar.radio("Teaching mode", ["Beginner", "Advanced"], key="classroom_teach_mode")
module_label = st.sidebar.radio("Learning path", MODULE_LABELS, key="classroom_module")
_render_classroom_guide(module_label, teach_mode)

module_id = MODULE_IDS[module_label]
if module_id == "intro":
    _module_intro(teach_mode)
elif module_id == "learn":
    _module_learn(teach_mode)
elif module_id == "mechanics":
    _module_mechanics(teach_mode)
elif module_id == "workflow":
    _module_workflow(teach_mode)
elif module_id == "mo":
    _module_mo(teach_mode)

