import os
import re
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from skopt.space import Real, Categorical
import dill as pickle  # persistence

from core.optimization.bayesian_optimization import StepBayesianOptimizer
from core.utils import db_handler

# =========================================================
# Config & Helpers
# =========================================================
if os.getenv("RENDER") == "true":
    SAVE_DIR = "/mnt/data/resumable_manual_runs"
else:
    SAVE_DIR = "resumable_manual_runs"
os.makedirs(SAVE_DIR, exist_ok=True)

def sanitize_name(name: str) -> str:
    name = (name or "").strip() or "manual_experiment"
    return re.sub(r'[^A-Za-z0-9_\- ]+', '_', name)

def _list_valid_campaigns(base_dir: str):
    if not os.path.exists(base_dir):
        return []
    valid = []
    for d in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, d)
        if os.path.isdir(p) and \
           os.path.exists(os.path.join(p, "manual_data.csv")) and \
           os.path.exists(os.path.join(p, "metadata.json")):
            valid.append(d)
    return valid

def _safe_build_optimizer(space, n_initial_points_remaining=0, acq_func="EI"):
    """
    Build StepBayesianOptimizer while being tolerant to different __init__ signatures.
    """
    try:
        return StepBayesianOptimizer(
            space,
            n_initial_points=n_initial_points_remaining,
            acq_func=acq_func
        )
    except TypeError:
        opt = StepBayesianOptimizer(space)
        # best-effort to set remaining initials on skopt.Optimizer
        try:
            if hasattr(opt, "_opt"):
                if hasattr(opt._opt, "_n_initial_points"):
                    opt._opt._n_initial_points = n_initial_points_remaining
                if hasattr(opt._opt, "n_initial_points_"):
                    opt._opt.n_initial_points_ = n_initial_points_remaining
                if hasattr(opt._opt, "acq_func"):
                    opt._opt.acq_func = acq_func
        except Exception:
            pass
        return opt

def force_model_based(optimizer):
    """Force next suggest() to be acquisition-driven (no random initials)."""
    try:
        if hasattr(optimizer, "_opt"):
            if hasattr(optimizer._opt, "_n_initial_points"):
                optimizer._opt._n_initial_points = 0
            if hasattr(optimizer._opt, "n_initial_points_"):
                optimizer._opt.n_initial_points_ = 0
    except Exception:
        pass

# ----------------------------- Two-space helpers -----------------------------

def _unionize_bounds(curr_variables, seeds_df: pd.DataFrame | None):
    """
    Expand continuous bounds to include all seed values.
    For categoricals, unify category lists.
    Returns a new 'variables' list (ModelSpace).
    """
    if seeds_df is None or seeds_df.empty:
        return curr_variables

    new_vars = []
    for (name, v1, v2, unit, vtype) in curr_variables:
        if vtype == "continuous":
            col = seeds_df[name] if name in seeds_df.columns else pd.Series(dtype=float)
            col = pd.to_numeric(col, errors="coerce").dropna()
            lo = min([v1] + (col.tolist() if not col.empty else []))
            hi = max([v2] + (col.tolist() if not col.empty else []))
            new_vars.append((name, float(lo), float(hi), unit, "continuous"))
        else:
            col = seeds_df[name] if name in seeds_df.columns else pd.Series(dtype=object)
            cats = set(v1) | set(col.dropna().astype(str).unique().tolist())
            new_vars.append((name, sorted(list(cats)), None, unit, "categorical"))
    return new_vars

def _in_suggest_space(x, suggest_variables):
    for (val, (name, v1, v2, _unit, vtype)) in zip(x, suggest_variables):
        if vtype == "continuous":
            try:
                fv = float(val)
            except Exception:
                return False
            if not (v1 <= fv <= v2):
                return False
        else:
            if str(val) not in set(map(str, v1)):
                return False
    return True

def _project_to_suggest_space(x, suggest_variables):
    """Last-resort projection: clip continuous; map unknown categorical to first."""
    out = []
    for (val, (name, v1, v2, _unit, vtype)) in zip(x, suggest_variables):
        if vtype == "continuous":
            fv = float(val)
            out.append(min(max(fv, v1), v2))
        else:
            cats = list(v1)
            sval = str(val)
            out.append(sval if sval in set(map(str, cats)) else cats[0])
    return out

# ----------------------------- Rebuild / observe -----------------------------

def rebuild_optimizer_from_df(
    variables,
    df: pd.DataFrame,
    response_col: str,
    n_initial_points_remaining: int = 0,
    acq_func: str = "EI"
) -> StepBayesianOptimizer:
    """Build StepBayesianOptimizer on 'variables' (ModelSpace), and observe seeds once."""
    space = []
    for name, v1, v2, _unit, vtype in variables:
        if vtype == "continuous":
            space.append(Real(v1, v2, name=name))
        else:
            space.append(Categorical(v1, name=name))  # v1 is list

    opt = _safe_build_optimizer(space, n_initial_points_remaining, acq_func)

    df = df.copy()
    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in reused data.")
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    df = df.dropna(subset=[response_col])

    for _, row in df.iterrows():
        x = [row[name] for name, *_ in variables]
        try:
            y = float(row[response_col])
            if pd.notnull(y):
                opt.observe(x, -y)  # maximizing
        except (ValueError, TypeError):
            continue

    if n_initial_points_remaining == 0:
        force_model_based(opt)
    return opt

# ----------------------------- Suggestion utils -----------------------------

def _existing_points_set(manual_variables, manual_data):
    """Build a set of tuples representing existing X (for duplicate checking)."""
    cols = [name for name, *_ in manual_variables]
    s = set()
    for row in manual_data:
        tup = tuple(row.get(c) for c in cols)
        s.add(tup)
    return s

def next_unique_suggestion(optimizer, manual_variables, manual_data, max_tries=120):
    """
    Generate next suggestion that is:
      (1) inside SuggestSpace (manual_variables),
      (2) not duplicate of existing X.
    Uses rejection sampling; projection fallback on last draw.
    """
    suggest_variables = manual_variables  # the user's current bounds
    seen = _existing_points_set(suggest_variables, manual_data)

    last_x = None
    for _ in range(max_tries):
        x = optimizer.suggest()
        last_x = x
        # keep only in-bounds suggestions
        if not _in_suggest_space(x, suggest_variables):
            continue
        tup = tuple(xi for xi in x)
        if tup not in seen:
            return x

    # Fallback: project last suggestion into bounds
    if last_x is not None:
        x_proj = _project_to_suggest_space(last_x, suggest_variables)
        if tuple(x_proj) not in seen:
            return x_proj

    # Last resort: return first valid corner point
    out = []
    for (name, v1, v2, unit, vtype) in suggest_variables:
        if vtype == "continuous":
            out.append(float(v1))
        else:
            out.append(v1[0])
    if tuple(out) in seen and len(suggest_variables) > 0:
        # small jitter for continuous dims
        out2 = []
        for (val, (name, v1, v2, unit, vtype)) in zip(out, suggest_variables):
            if vtype == "continuous":
                eps = (v2 - v1) * 1e-6
                out2.append(min(max(val + eps, v1), v2))
            else:
                out2.append(val)
        return out2
    return out

# =========================================================
# Session State Initialization
# =========================================================
defaults = {
    "manual_variables": [],       # SuggestSpace
    "model_variables": None,      # ModelSpace (union when reusing; else same as manual_variables)
    "manual_data": [],
    "manual_optimizer": None,
    "manual_initialized": False,
    "suggestions": [],
    "iteration": 0,
    "initial_results_submitted": False,
    "next_suggestion_cached": None,
    "submitted_initial": False,
    "edited_initial_df": None,
    "n_init": 1,
    "total_iters": 1,
    "edit_mode": False,
    "recalc_needed": False,
    "response": "Yield",
    "var_type": "Continuous",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# Sidebar: Resume (load exact previous run of this user)
# =========================================================
user_email = st.session_state.get("user_email", "default_user")
user_save_dir = os.path.join(SAVE_DIR, user_email)
os.makedirs(user_save_dir, exist_ok=True)

st.sidebar.markdown("---")
_resume_options = ["None"] + _list_valid_campaigns(user_save_dir)
resume_file = st.sidebar.selectbox("🔄 Resume from Previous Manual Campaign", options=_resume_options)

if resume_file != "None" and st.sidebar.button("Load Previous Manual Campaign"):
    run_path = os.path.join(user_save_dir, resume_file)
    try:
        with open(os.path.join(run_path, "optimizer.pkl"), "rb") as f:
            st.session_state.manual_optimizer = pickle.load(f)
    except FileNotFoundError:
        st.warning("optimizer.pkl not found. The optimizer will be rebuilt from data when needed.")
        st.session_state.manual_optimizer = None

    try:
        df_loaded = pd.read_csv(os.path.join(run_path, "manual_data.csv"))
    except pd.errors.EmptyDataError:
        df_loaded = pd.DataFrame()
        st.warning("manual_data.csv is empty. Starting with an empty dataset.")
    except FileNotFoundError:
        df_loaded = pd.DataFrame()
        st.warning("manual_data.csv not found. Starting with an empty dataset.")

    with open(os.path.join(run_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    st.session_state.manual_data = df_loaded.to_dict("records")
    st.session_state.manual_variables = metadata.get("variables", [])
    # if missing in metadata, align model_variables with manual_variables for now
    st.session_state.model_variables = metadata.get("model_variables", st.session_state.manual_variables)

    st.session_state.iteration = metadata.get("iteration", len(df_loaded))
    st.session_state.campaign_name = resume_file
    st.session_state.n_init = metadata.get("n_init", 1)
    st.session_state.total_iters = metadata.get("total_iters", 1)
    st.session_state.response = metadata.get("response", st.session_state.get("response", "Yield"))
    st.session_state.manual_initialized = True
    st.session_state.initial_results_submitted = metadata.get("initialization_complete", False)
    st.session_state.experiment_name = metadata.get("experiment_name", "")
    st.session_state.experiment_notes = metadata.get("experiment_notes", "")

    # If optimizer exists, (re)observe data using current ModelSpace (or fall back)
    if st.session_state.manual_variables and len(st.session_state.manual_data) > 0:
        model_vars = st.session_state.model_variables or st.session_state.manual_variables
        # rebuild opt fresh so space is correct
        opt_vars = []
        for name, v1, v2, _, vtype in model_vars:
            if vtype == "continuous":
                opt_vars.append(Real(v1, v2, name=name))
            else:
                opt_vars.append(Categorical(v1, name=name))

        optimizer = _safe_build_optimizer(opt_vars, n_initial_points_remaining=0, acq_func="EI")

        df_tmp = pd.DataFrame(st.session_state.manual_data)
        resp = st.session_state.response
        if resp in df_tmp.columns:
            df_tmp[resp] = pd.to_numeric(df_tmp[resp], errors="coerce")
            for _, row in df_tmp.iterrows():
                try:
                    y_val = float(row.get(resp, float("nan")))
                    if pd.notnull(y_val):
                        x = [row.get(name) for name, *_ in model_vars]
                        optimizer.observe(x, -y_val)
                except (ValueError, TypeError):
                    continue

        force_model_based(optimizer)
        st.session_state.manual_optimizer = optimizer

    st.success(f"Loaded campaign: {resume_file}")

# =========================================================
# Title & Reset
# =========================================================
st.title("🧰 Manual Optimization Campaign")

if st.button("🔄 Reset Campaign"):
    for key in list(st.session_state.keys()):
        if key not in ("user_email",):
            del st.session_state[key]
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()

st.markdown("Reuse past experiments as seeds, fit on a UNION space, and keep new suggestions inside your current bounds.")

# =========================================================
# Chart Functions (main area only)
# =========================================================
def show_progress_chart(data: list, response_name: str):
    if len(data) == 0:
        return
    df_results = pd.DataFrame(data)
    if df_results.empty or response_name not in df_results.columns:
        return
    df_results["Iteration"] = range(1, len(df_results) + 1)
    df_results[response_name] = pd.to_numeric(df_results[response_name], errors="coerce")

    st.markdown("### 📈 Optimization Progress")
    chart = alt.Chart(df_results).mark_line(point=True).encode(
        x=alt.X("Iteration", title="Experiment Number"),
        y=alt.Y(response_name, title=response_name),
        tooltip=["Iteration", response_name]
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

    if df_results[response_name].notna().any():
        best_val = df_results[response_name].max()
        st.markdown(f"**Current Best {response_name}:** {best_val:.4g}")

def show_parallel_coordinates(data: list, response_name: str):
    if len(data) == 0:
        return
    df = pd.DataFrame(data).copy()
    if df.empty or response_name not in df.columns:
        return

    df[response_name] = pd.to_numeric(df[response_name], errors="coerce")

    input_vars = [name for name, *_ in st.session_state.manual_variables]
    cols_to_plot = [c for c in (input_vars + [response_name]) if c in df.columns]
    if not cols_to_plot:
        return
    df = df[cols_to_plot]

    st.markdown("### 🔀 Parallel Coordinates Plot")

    legend_entries = []
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col].astype(str))
                legend_entries.append((col, dict(enumerate(le.classes_))))
            except Exception:
                continue

    fig = px.parallel_coordinates(
        df,
        color=response_name,
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={c: c for c in df.columns}
    )
    fig.update_layout(
        font=dict(size=20, color='black'),
        height=500,
        margin=dict(l=50, r=50, t=50, b=40),
        coloraxis_colorbar=dict(
            title=dict(text=response_name, font=dict(size=20, color='black')),
            tickfont=dict(size=20, color='black'),
            len=0.8,
            thickness=40,
            tickprefix=" ",
            xpad=5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    if legend_entries:
        st.markdown("### 🏷️ Categorical Legends")
        for col, mapping in legend_entries:
            st.markdown(f"**{col}**:")
            for code, label in mapping.items():
                st.markdown(f"- `{code}` → `{label}`")

# =========================================================
# Experiment Header
# =========================================================
experiment_name = st.text_input("Experiment Name", value=st.session_state.get("experiment_name", ""))
experiment_notes = st.text_area("Notes (optional)", value=st.session_state.get("experiment_notes", ""))
experiment_date = st.date_input("Experiment date")

run_name = sanitize_name(experiment_name)
st.session_state["campaign_name"] = run_name
run_path = os.path.join(user_save_dir, run_name)
os.makedirs(run_path, exist_ok=True)

# =========================================================
# Save Campaign (Sidebar)
# =========================================================
if st.sidebar.button("💾 Save Campaign"):
    try:
        with open(os.path.join(run_path, "optimizer.pkl"), "wb") as f:
            pickle.dump(st.session_state.manual_optimizer, f)
    except Exception:
        st.sidebar.warning("Could not save optimizer.pkl (optimizer may be None). Proceeding with data and metadata.")

    if st.session_state.manual_data:
        df_save = pd.DataFrame(st.session_state.manual_data)
    else:
        base_cols = [name for name, *_ in st.session_state.manual_variables]
        if st.session_state.get("response"):
            base_cols.append(st.session_state["response"])
        df_save = pd.DataFrame(columns=base_cols)
    df_save.to_csv(os.path.join(run_path, "manual_data.csv"), index=False)

    metadata = {
        "variables": st.session_state.manual_variables,         # SuggestSpace
        "model_variables": st.session_state.get("model_variables", st.session_state.manual_variables),  # ModelSpace
        "iteration": st.session_state.get("iteration", len(df_save)),
        "n_init": st.session_state.n_init,
        "total_iters": st.session_state.total_iters,
        "response": st.session_state.get("response", "Yield"),
        "experiment_name": experiment_name,
        "experiment_notes": experiment_notes,
        "initialization_complete": st.session_state.get("initial_results_submitted", False)
    }
    with open(os.path.join(run_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    st.sidebar.success(f"Campaign '{run_name}' saved successfully!")

# =========================================================
# Variable Definition / Editing
# =========================================================
st.subheader("🔧 Define and Edit Variables")
st.session_state.var_type = st.selectbox("Variable Type", ["Continuous", "Categorical"], key="var_type_select")

with st.form("manual_var_form"):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        var_name = st.text_input("Variable Name")
    with col2:
        if st.session_state.var_type == "Continuous":
            lower = st.number_input("Lower Bound", value=0.0, format="%.4f")
            upper = st.number_input("Upper Bound", value=1.0, format="%.4f")
        else:
            categories = st.text_input("Categories (comma-separated)", value="Type")
    with col3:
        unit = st.text_input("Unit")

    add_var = st.form_submit_button("Add Variable")
    if add_var and var_name:
        if st.session_state.var_type == "Continuous" and lower < upper:
            st.session_state.manual_variables.append((var_name, lower, upper, unit, "continuous"))
        elif st.session_state.var_type == "Categorical" and categories:
            values = [x.strip() for x in categories.split(",") if x.strip()]
            st.session_state.manual_variables.append((var_name, values, None, unit, "categorical"))

if st.session_state.manual_variables:
    st.markdown("### ✏️ Edit Variables")
    variables_df = pd.DataFrame(
        [
            {
                "Name": name,
                "Type": vtype,
                "Value 1": val1,
                "Value 2": val2 if vtype == "continuous" else None,
                "Unit": unit
            }
            for name, val1, val2, unit, vtype in st.session_state.manual_variables
        ]
    )
    edited_df = st.data_editor(variables_df, key="edit_variables_editor")
    if st.button("💾 Save Variable Changes"):
        updated_variables = []
        for _, row in edited_df.iterrows():
            if row["Type"] == "continuous" and pd.notnull(row["Value 1"]) and pd.notnull(row["Value 2"]) and row["Value 1"] < row["Value 2"]:
                updated_variables.append((row["Name"], float(row["Value 1"]), float(row["Value 2"]), row["Unit"], "continuous"))
            elif row["Type"] == "categorical":
                v1 = row["Value 1"]
                if isinstance(v1, list):
                    values = v1
                elif isinstance(v1, str):
                    values = [x.strip() for x in v1.split(",") if x.strip()]
                else:
                    values = []
                if values:
                    updated_variables.append((row["Name"], values, None, row["Unit"], "categorical"))
        st.session_state.manual_variables = updated_variables
        # If user edits bounds and we don't have a model space yet, align it
        if st.session_state.get("model_variables") is None:
            st.session_state.model_variables = st.session_state.manual_variables
        st.success("Variables updated successfully!")

    delete_var = st.selectbox("Select a Variable to Delete", options=["None"] + [v[0] for v in st.session_state.manual_variables])
    if delete_var != "None" and st.button("🗑️ Delete Variable"):
        st.session_state.manual_variables = [v for v in st.session_state.manual_variables if v[0] != delete_var]
        if st.session_state.get("model_variables") is None:
            st.session_state.model_variables = st.session_state.manual_variables
        st.success(f"Variable '{delete_var}' deleted successfully!")

# =========================================================
# Experiment Setup
# =========================================================
st.subheader("⚙️ Experiment Setup")
col5, col6 = st.columns(2)
with col5:
    response = st.selectbox(
        "Response to Optimize",
        ["Yield", "Conversion", "Transformation", "Productivity"],
        index=["Yield", "Conversion", "Transformation", "Productivity"].index(st.session_state.get("response", "Yield"))
    )
    st.session_state.response = response
with col6:
    n_init = st.number_input("# Initial Experiments", min_value=1, max_value=50, value=st.session_state.n_init, key="n_init")
    total_iters = st.number_input("Total Iterations", min_value=1, max_value=100, value=st.session_state.total_iters, key="total_iters")

# =========================================================
# Suggest Initial Experiments (fresh run only)
# =========================================================
if st.button("🚀 Suggest Initial Experiments"):
    if st.session_state.manual_initialized and st.session_state.manual_data:
        st.info("Already initialized (possibly via reuse). Use 📎 Get Next Suggestion to continue.")
    elif not st.session_state.manual_variables:
        st.warning("Please define at least one variable first.")
    else:
        # Fresh run: ModelSpace == SuggestSpace
        st.session_state.model_variables = st.session_state.manual_variables

        opt_vars = []
        for name, val1, val2, _, vtype in st.session_state.model_variables:
            if vtype == "continuous":
                opt_vars.append(Real(val1, val2, name=name))
            else:
                opt_vars.append(Categorical(val1, name=name))

        optimizer = _safe_build_optimizer(opt_vars, n_initial_points_remaining=st.session_state.n_init, acq_func="EI")
        st.session_state.manual_optimizer = optimizer
        st.session_state.manual_data = []
        st.session_state.manual_initialized = True
        st.session_state.iteration = 0
        st.session_state.initial_results_submitted = False
        st.session_state.next_suggestion_cached = None
        st.session_state.suggestions = [optimizer.suggest() for _ in range(st.session_state.n_init)]
        st.success("Initial experiments suggested successfully!")

# =========================================================
# Initial Results Input (only shown once)
# =========================================================
if not st.session_state.initial_results_submitted and st.session_state.suggestions:
    st.markdown("### 🧪 Initial Experiments (User Input Required)")
    st.caption("Press Enter or click outside each cell to confirm your entry before submitting.")

    default_data = []
    for vals in st.session_state.suggestions:
        row = {name: val for (name, *_), val in zip(st.session_state.manual_variables, vals)}
        row[response] = None
        default_data.append(row)

    edited_df_init = st.data_editor(pd.DataFrame(default_data), num_rows="fixed", key="initial_results_editor")

    if st.button("✅ Submit Initial Results"):
        if edited_df_init is not None:
            st.session_state.edited_initial_df = edited_df_init.copy()
            st.session_state.submitted_initial = True
        else:
            st.error("Please fill in the table before submitting.")

if st.session_state.submitted_initial and st.session_state.edited_initial_df is not None:
    valid_rows = 0
    for _, row in st.session_state.edited_initial_df.iterrows():
        value = row.get(response)
        if value is None or str(value).strip() == "":
            continue
        try:
            y_val = float(value)
            x = [row[name] for name, *_ in st.session_state.manual_variables]
            st.session_state.manual_optimizer.observe(x, -y_val)
            row_data = row.to_dict()
            row_data[response] = y_val
            row_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.manual_data.append(row_data)
            valid_rows += 1
        except ValueError:
            st.error(f"Invalid number format: {value}")
            st.stop()

    if valid_rows < len(st.session_state.edited_initial_df):
        st.warning(f"Only {valid_rows} of {len(st.session_state.edited_initial_df)} experiments had valid results.")
        st.stop()

    st.session_state.iteration += valid_rows
    st.session_state.suggestions = []
    st.session_state.initial_results_submitted = True
    st.session_state.submitted_initial = False
    force_model_based(st.session_state.manual_optimizer)

# =========================================================
# Sidebar: Reuse Previous Campaign (edit + select; with Select/Clear all)
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Reuse Previous Campaign as Seeds")

_reuse_options = ["None"] + _list_valid_campaigns(user_save_dir)
reuse_campaign = st.sidebar.selectbox("Select a Previous Campaign to Reuse", options=_reuse_options)

if reuse_campaign != "None":
    reuse_path = os.path.join(user_save_dir, reuse_campaign)
    try:
        prev_df_raw = pd.read_csv(os.path.join(reuse_path, "manual_data.csv"))
        with open(os.path.join(reuse_path, "metadata.json"), "r") as f:
            prev_meta = json.load(f)

        prev_variables = prev_meta.get("variables", [])
        curr_variables = st.session_state.manual_variables

        if len(prev_variables) != len(curr_variables):
            st.error("Variable count mismatch between previous and current campaign.")
        elif not all(p[0] == c[0] for p, c in zip(prev_variables, curr_variables)):
            st.error("Variable names do not match between campaigns.")
        else:
            resp = st.session_state.get("response", "Yield")
            if resp not in prev_df_raw.columns:
                candidates = ["Yield", "Conversion", "Transformation", "Productivity"]
                fallback = next((c for c in candidates if c in prev_df_raw.columns), None)
                if fallback is None:
                    st.error("No valid response column found in previous data.")
                else:
                    resp = fallback
                    st.session_state.response = resp
                    st.info(f"Using '{resp}' as response column from previous data.")

            if st.session_state.get("response") in prev_df_raw.columns or resp in prev_df_raw.columns:
                required_cols = [name for name, *_ in curr_variables]
                missing = [c for c in required_cols if c not in prev_df_raw.columns]
                if missing:
                    st.error(f"Missing variable columns in previous data: {missing}")
                else:
                    if "prev_df_editor_cache" not in st.session_state or st.session_state.get("prev_df_source_campaign") != reuse_campaign:
                        df_for_editor = prev_df_raw.copy()
                        df_for_editor.insert(0, "Use", True)
                        st.session_state.prev_df_editor_cache = df_for_editor
                        st.session_state.prev_df_source_campaign = reuse_campaign

                    st.markdown("### 📋 Previous Experiments (edit + select)")
                    default_cols = ["Use"] + required_cols + [resp]
                    extra_cols = [c for c in st.session_state.prev_df_editor_cache.columns if c not in default_cols]
                    show_cols = st.multiselect(
                        "Columns to display",
                        options=list(st.session_state.prev_df_editor_cache.columns),
                        default=default_cols + ([c for c in extra_cols if c.lower() in ["timestamp"]]),
                        key="reuse_cols_multiselect",
                    )
                    if not show_cols:
                        show_cols = list(st.session_state.prev_df_editor_cache.columns)

                    edited_prev_df = st.data_editor(
                        st.session_state.prev_df_editor_cache[show_cols],
                        key=f"reuse_editor_{reuse_campaign}",
                        use_container_width=True,
                        column_config={
                            "Use": st.column_config.CheckboxColumn("Use", help="Tick rows you want to include", default=True)
                        }
                    )
                    st.session_state.prev_df_editor_cache.loc[:, show_cols] = edited_prev_df

                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        if st.button("Select all"):
                            st.session_state.prev_df_editor_cache["Use"] = True
                            st.rerun()
                    with c2:
                        if st.button("Clear all"):
                            st.session_state.prev_df_editor_cache["Use"] = False
                            st.rerun()
                    with c3:
                        skip_random = st.checkbox(
                            "Skip additional random initial points (start BO suggestions immediately)",
                            value=True,
                            key="reuse_skip_random"
                        )

                    if st.button("Use selected experiments"):
                        selected_df = st.session_state.prev_df_editor_cache.copy()
                        selected_df[resp] = pd.to_numeric(selected_df[resp], errors="coerce")
                        selected_df = selected_df[selected_df["Use"] & selected_df[resp].notna()]

                        keep_cols = required_cols + [resp]
                        extra_keep = [c for c in ["Timestamp"] if c in selected_df.columns]
                        selected_df = selected_df[keep_cols + extra_keep].copy()

                        if selected_df.empty:
                            st.error("Select at least one valid row (with numeric response).")
                        else:
                            # Build ModelSpace as union(SuggestSpace, seeds)
                            model_variables = _unionize_bounds(st.session_state.manual_variables, selected_df)

                            seed_count = len(selected_df)
                            remaining_init = 0 if skip_random else max(0, int(st.session_state.n_init) - seed_count)

                            optimizer = rebuild_optimizer_from_df(
                                model_variables,     # fit on union bounds
                                selected_df,
                                resp,
                                n_initial_points_remaining=remaining_init,
                                acq_func="EI"
                            )

                            st.session_state.model_variables = model_variables
                            st.session_state.manual_optimizer = optimizer
                            st.session_state.manual_initialized = True
                            st.session_state.manual_data = selected_df.to_dict("records")
                            st.session_state.iteration = seed_count
                            st.session_state.initial_results_submitted = True
                            st.session_state.submitted_initial = False

                            # reflect "no initials left" in UI and caches
                            st.session_state.suggestions = []
                            st.session_state.next_suggestion_cached = None

                            msg = f"Reused {seed_count} experiment(s) from '{reuse_campaign}'. "
                            msg += "Starting BO now." if remaining_init == 0 else f"{remaining_init} initial random(s) remain."
                            st.success(msg)

    except FileNotFoundError as e:
        st.error(f"Missing file in selected campaign: {e.filename}")
    except pd.errors.EmptyDataError:
        st.error("The manual_data.csv in the selected campaign is empty.")
    except Exception as ex:
        st.error(f"Could not reuse campaign: {ex}")

# =========================================================
# Always show charts if data exists (MAIN area only)
# =========================================================
if len(st.session_state.manual_data) > 0:
    show_progress_chart(st.session_state.manual_data, st.session_state.response)
    show_parallel_coordinates(st.session_state.manual_data, st.session_state.response)

# =========================================================
# Edit Previous Results
# =========================================================
if len(st.session_state.manual_data) > 0:
    st.markdown("### ✏️ Edit Previous Results")
    if st.button("Enable Edit Mode"):
        st.session_state.edit_mode = True

    if st.session_state.edit_mode:
        edited_df = st.data_editor(pd.DataFrame(st.session_state.manual_data), key="edit_results_editor")
        if st.button("Save Edits"):
            st.session_state.manual_data = edited_df.to_dict("records")
            st.session_state.edit_mode = False
            st.session_state.recalc_needed = True
            st.success("Edits saved! The optimizer will be recalculated.")
            st.rerun()

# =========================================================
# Truncate to a specific experiment
# =========================================================
if len(st.session_state.manual_data) > 0:
    st.markdown("###  Return to a Previous Experiment")
    max_idx = len(st.session_state.manual_data)
    trunc_idx = st.number_input("Keep experiments up to (inclusive):", min_value=1, max_value=max_idx, value=max_idx, step=1)
    if st.button("Return and Restart From Here"):
        st.session_state.manual_data = st.session_state.manual_data[:trunc_idx]
        st.session_state.recalc_needed = True
        st.success(f"Returned to experiment {trunc_idx}. The optimizer will be recalculated.")
        st.rerun()

# =========================================================
# Recalculate optimizer after edits or truncation
# =========================================================
if st.session_state.recalc_needed:
    if st.session_state.manual_variables and st.session_state.manual_data:
        # Use stored ModelSpace if present; else fall back to SuggestSpace
        model_vars = st.session_state.get("model_variables", st.session_state.manual_variables)

        opt_vars = []
        for name, val1, val2, _, vtype in model_vars:
            if vtype == "continuous":
                opt_vars.append(Real(val1, val2, name=name))
            else:
                opt_vars.append(Categorical(val1, name=name))

        optimizer = _safe_build_optimizer(opt_vars, n_initial_points_remaining=0, acq_func="EI")

        df_tmp = pd.DataFrame(st.session_state.manual_data)
        resp = st.session_state.response
        if resp in df_tmp.columns:
            df_tmp[resp] = pd.to_numeric(df_tmp[resp], errors="coerce")
            for _, row in df_tmp.iterrows():
                try:
                    y_val = float(row.get(resp, float("nan")))
                    if pd.notnull(y_val):
                        x = [row.get(name) for name, *_ in model_vars]
                        optimizer.observe(x, -y_val)
                except (ValueError, TypeError):
                    continue

        force_model_based(optimizer)
        st.session_state.manual_optimizer = optimizer
        st.session_state.iteration = len(st.session_state.manual_data)
    st.session_state.recalc_needed = False

# =========================================================
# Get next suggestion (step-by-step) — constrained to SuggestSpace
# =========================================================
if (
    st.session_state.manual_initialized
    and st.session_state.manual_optimizer is not None
    and st.session_state.iteration < st.session_state.total_iters
    and st.session_state.initial_results_submitted
):
    if st.button("📎 Get Next Suggestion"):
        st.session_state.next_suggestion_cached = next_unique_suggestion(
            st.session_state.manual_optimizer,
            st.session_state.manual_variables,  # SuggestSpace
            st.session_state.manual_data,
            max_tries=120
        )

# Show cached suggestion + capture result
if st.session_state.next_suggestion_cached is not None:
    st.markdown("### ▶️ Next Experiment Suggestion")
    next_row = {
        name: val
        for (name, *_), val in zip(st.session_state.manual_variables, st.session_state.next_suggestion_cached)
    }
    st.dataframe(pd.DataFrame([next_row]), use_container_width=True)

    result = st.number_input(
        f"Result for {st.session_state.response} (Experiment {st.session_state.iteration + 1})",
        key=f"next_result_{st.session_state.iteration}"
    )

    if st.button("➕ Submit Result"):
        st.success("Result submitted. Press '**📎 Get Next Suggestion**' for the next point—charts update automatically.")
        if pd.notnull(result):
            x = [next_row[name] for name, *_ in st.session_state.manual_variables]
            y_val = float(result)
            st.session_state.manual_optimizer.observe(x, -y_val)

            row_data = {**next_row}
            row_data[st.session_state.response] = y_val
            row_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.manual_data.append(row_data)

            st.session_state.iteration += 1
            st.session_state.next_suggestion_cached = None  # clear

# =========================================================
# Completed state
# =========================================================
if st.session_state.iteration >= st.session_state.total_iters and st.session_state.total_iters > 0:
    st.markdown("### ✅ Optimization Completed")
    st.success("All iterations are completed! You can export the data or review the results.")

    df_results = pd.DataFrame(st.session_state.manual_data)
    st.dataframe(df_results, use_container_width=True)

    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", data=csv, file_name="manual_optimization_results.csv", mime="text/csv")

    if st.button("💾 Save to Database"):
        if st.session_state.response in df_results.columns and df_results[st.session_state.response].notna().any():
            best_row = df_results.loc[df_results[st.session_state.response].idxmax()].to_dict()
        else:
            best_row = {}

        optimization_settings = {
            "initial_experiments": st.session_state.n_init,
            "total_iterations": st.session_state.total_iters,
            "objective": st.session_state.response,
            "method": "Manual Bayesian Optimization"
        }

        db_handler.save_experiment(
            user_email=st.session_state.get("user_email", "default_user"),
            name=experiment_name,
            notes=experiment_notes,
            variables=st.session_state.manual_variables,
            df_results=df_results,
            best_result=best_row,
            settings=optimization_settings
        )

        run_path = os.path.join(user_save_dir, run_name)
        os.makedirs(run_path, exist_ok=True)
        df_results.to_csv(os.path.join(run_path, "manual_data.csv"), index=False)

        metadata = {
            "variables": st.session_state.manual_variables,  # SuggestSpace
            "model_variables": st.session_state.get("model_variables", st.session_state.manual_variables),  # ModelSpace
            "iteration": st.session_state.get("iteration", len(df_results)),
            "n_init": st.session_state.n_init,
            "total_iters": st.session_state.total_iters,
            "response": st.session_state.response,
            "experiment_name": experiment_name,
            "experiment_notes": experiment_notes,
            "initialization_complete": st.session_state.get("initial_results_submitted", False)
        }
        with open(os.path.join(run_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        st.success("✅ Experiment saved successfully! All campaign files have been generated.")
# =========================================================
# End of Manual Experimentation
# =========================================================
# =========================================================     













