# ui/charts.py
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
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
