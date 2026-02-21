# ui/charts.py
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =========================================================
# Chart Functions (main area only)
# =========================================================
class Charts:
    @staticmethod
    def show_progress_chart(data: list, response_name: str):
        """
        Display a line chart showing the progress of an optimization experiment.

        Parameters
        ----------
        data : list
            A list of dictionaries containing experimental results. Each dictionary should
            contain at least the key corresponding to `response_name`.
        response_name : str
            The name of the column in `data` to track as the optimization response.

        Behavior
        --------
        - Creates a DataFrame from the input data.
        - Adds an "Iteration" column for experiment numbering.
        - Converts the response column to numeric values (ignoring errors).
        - Plots an Altair line chart with points and a tooltip for each iteration.
        - Displays the current best value of the response if any valid numeric values exist.
        """
        if len(data) == 0:
            return
        df_results = pd.DataFrame(data)
        if df_results.empty or response_name not in df_results.columns:
            return
        df_results["Iteration"] = range(1, len(df_results) + 1)
        df_results[response_name] = pd.to_numeric(df_results[response_name], errors="coerce")

        with st.container(border=True):
            st.markdown("### Optimization Progress")
            chart = alt.Chart(df_results).mark_line(
                point=alt.OverlayMarkDef(size=110, filled=True)
            ).encode(
                x=alt.X(
                    "Iteration:Q",
                    title="Experiment Number",
                    axis=alt.Axis(
                        format="d",
                        tickMinStep=1,
                        labelColor="black",
                        titleColor="black",
                        labelFontSize=14,
                        titleFontSize=16,
                    ),
                ),
                y=alt.Y(
                    response_name,
                    title=response_name,
                    axis=alt.Axis(
                        labelColor="black",
                        titleColor="black",
                        labelFontSize=14,
                        titleFontSize=16,
                    ),
                ),
                tooltip=["Iteration", response_name]
            ).properties(width=700, height=400)
            st.altair_chart(chart, use_container_width=True)

            if df_results[response_name].notna().any():
                direction = st.session_state.get("response_direction", "Maximize")
                best_val = df_results[response_name].max() if direction == "Maximize" else df_results[response_name].min()
                label = "Best" if direction == "Maximize" else "Lowest"
                st.markdown(f"**Current {label} {response_name}:** {best_val:.4g}")


    @staticmethod
    def show_parallel_coordinates(data: list, response_name: str):
        """
        Display a parallel coordinates plot for the input experimental data.

        Parameters
        ----------
        data : list
            A list of dictionaries representing experimental results.
        response_name : str
            The column to use as the color mapping in the plot.

        Behavior
        --------
        - Converts the input list to a DataFrame.
        - Ensures the response column is numeric.
        - Retrieves input variables from `st.session_state.manual_variables`.
        - Encodes categorical columns to numeric using LabelEncoder.
        - Plots a Plotly parallel coordinates chart.
        - Displays legends for categorical variables, mapping encoded numbers back to original labels.
        """
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

        with st.container(border=True):
            st.markdown("### Parallel Coordinates Plot")

            legend_entries = []
            categorical_cols = set()
            for col in df.columns:
                if df[col].dtype == object:
                    le = LabelEncoder()
                    try:
                        df[col] = le.fit_transform(df[col].astype(str))
                        categorical_cols.add(col)
                        legend_entries.append((col, dict(enumerate(le.classes_))))
                    except Exception:
                        continue

            fig = px.parallel_coordinates(
                df,
                color=response_name,
                color_continuous_scale=px.colors.sequential.Viridis_r,
                labels={c: c for c in df.columns}
            )
            # Keep numeric axis tick labels readable for chemists.
            try:
                for dim in fig.data[0].dimensions:
                    dim_label = getattr(dim, "label", None)
                    if dim_label not in categorical_cols:
                        dim.tickformat = ".2f"
            except Exception:
                pass
            fig.update_layout(
                font=dict(size=20, color='black'),
                height=500,
                margin=dict(l=70, r=50, t=50, b=40),
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
                st.markdown("### Categorical Legends")
                for col, mapping in legend_entries:
                    st.markdown(f"**{col}**:")
                    for code, label in mapping.items():
                        st.markdown(f"- `{code}` -> `{label}`")

    @staticmethod
    def show_initial_design(points: list[list], variables: list[tuple]):
        """
        Visualize initial design with one line per continuous variable and
        categorical level counts.
        """
        if not points or not variables:
            return
        names = [name for name, *_ in variables]
        df = pd.DataFrame(points, columns=names)
        if df.empty:
            return

        st.markdown("### Initial Design Preview")
        continuous_vars = []
        categorical_vars = []
        for name, val1, val2, _unit, vtype in variables:
            if str(vtype).lower() == "continuous":
                continuous_vars.append((name, val1, val2))
            else:
                categorical_vars.append((name, val1))

        with st.container(border=True):
            if continuous_vars:
                fig = go.Figure()
                n_exp = len(df)
                exp_idx = np.arange(1, n_exp + 1)
                y_positions = np.arange(len(continuous_vars))

                for yi, (name, low_def, high_def) in enumerate(continuous_vars):
                    raw_vals = pd.to_numeric(df[name], errors="coerce")
                    if raw_vals.notna().any():
                        data_min = float(raw_vals.min())
                        data_max = float(raw_vals.max())
                    else:
                        data_min, data_max = 0.0, 1.0

                    try:
                        low = float(low_def)
                        high = float(high_def)
                        if not high > low:
                            low, high = data_min, data_max
                    except Exception:
                        low, high = data_min, data_max

                    fill_val = (low + high) / 2.0
                    vals = raw_vals.fillna(fill_val)
                    if high > low:
                        pos = (vals - low) / (high - low)
                    else:
                        pos = vals * 0.0 + 0.5
                    pos = pos.clip(0.0, 1.0)

                    fig.add_trace(
                        go.Scatter(
                            x=[0.0, 1.0],
                            y=[yi, yi],
                            mode="lines",
                            line=dict(color="#9CA3AF", width=3),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[0.0, 1.0],
                            y=[yi, yi],
                            mode="text",
                            text=[f"{low:.3g}", f"{high:.3g}"],
                            textposition=["middle left", "middle right"],
                            textfont=dict(size=11, color="#6B7280"),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=pos.to_numpy(dtype=float),
                            y=np.full(n_exp, yi, dtype=float),
                            mode="markers",
                            marker=dict(
                                size=10,
                                color=exp_idx,
                                colorscale="Viridis",
                                showscale=(yi == 0),
                                colorbar=dict(title="Experiment", len=0.9),
                                line=dict(color="white", width=1),
                            ),
                            customdata=np.column_stack([exp_idx, vals.to_numpy(dtype=float)]),
                            hovertemplate=(
                                "Experiment: %{customdata[0]:.0f}<br>"
                                f"{name}: "
                                "%{customdata[1]:.5g}<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                fig.update_layout(
                    height=max(280, 90 * len(continuous_vars)),
                    margin=dict(l=50, r=30, t=20, b=30),
                    xaxis=dict(
                        title="Position within variable range (normalized)",
                        range=[-0.03, 1.03],
                        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                        showgrid=True,
                        gridcolor="#E5E7EB",
                    ),
                    yaxis=dict(
                        tickmode="array",
                        tickvals=y_positions,
                        ticktext=[v[0] for v in continuous_vars],
                        range=[-0.6, len(continuous_vars) - 0.4],
                        showgrid=False,
                        zeroline=False,
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

            if categorical_vars:
                st.markdown("#### Categorical Level Counts")
                for name, _levels in categorical_vars:
                    if name not in df.columns:
                        continue
                    counts = df[name].astype(str).value_counts(dropna=False).reset_index()
                    counts.columns = [name, "Experiments"]
                    counts["Level"] = (
                        counts[name]
                        .astype(str)
                        .str.strip()
                        .str.strip("'")
                        .str.strip('"')
                        .str.replace(r"^\[(.*)\]$", r"\1", regex=True)
                    )
                    counts = (
                        counts.groupby("Level", as_index=False)["Experiments"]
                        .sum()
                        .sort_values("Experiments", ascending=True)
                    )
                    counts["Level"] = counts["Level"].astype(str)
                    level_order = counts["Level"].tolist()

                    fig_cat = px.bar(
                        counts,
                        x="Experiments",
                        y="Level",
                        orientation="h",
                        color="Experiments",
                        color_continuous_scale=px.colors.sequential.GnBu,
                        category_orders={"Level": level_order},
                    )
                    fig_cat.update_traces(
                        text=counts["Experiments"],
                        textposition="outside",
                        marker_line_color="rgba(15,23,42,0.15)",
                        marker_line_width=0.8,
                        hovertemplate=f"{name}: %{{y}}<br>Experiments: %{{x}}<extra></extra>",
                    )
                    fig_cat.update_layout(
                        height=min(260, max(130, 36 * len(counts))),
                        margin=dict(l=10, r=30, t=24, b=10),
                        title=dict(text=name, x=0.01, y=0.95, xanchor="left", yanchor="top", font=dict(size=14)),
                        xaxis=dict(title=None, showgrid=True, gridcolor="#E2E8F0", zeroline=False),
                        yaxis=dict(
                            title=None,
                            type="category",
                            categoryorder="array",
                            categoryarray=level_order,
                            showgrid=False,
                        ),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
