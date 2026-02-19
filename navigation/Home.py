import streamlit as st

from core.utils.path_utils import resource_path

st.image(str(resource_path("images/image3.png")), width=900)

st.markdown("---")

st.markdown(
    """
Welcome to **BO Studio** - a user-friendly space for experimenting with Bayesian Optimization manually and interactively.

Whether you're exploring optimization for the first time or running structured manual experiments, BO Studio gives you the tools to design, simulate, and analyze your campaigns with clarity.

#### With BO Studio, you can:
- Run **manual Bayesian optimization campaigns** (simulated or real values)
- Visualize and track the optimization process live
- **Save, resume, and preview** past campaigns
- Learn the basics in the **Bayesian Optimization Classroom**
- Store results in a clean, searchable experiment **database**
- Get answers in the built-in **FAQ & Guidance** section
"""
)

st.info('"A clear and simple environment for exploring Bayesian Optimization - no automation required."')

st.markdown("")

st.markdown(
    "Have suggestions or found a bug? "
    "[Click here to give feedback](https://docs.google.com/forms/d/e/1FAIpQLSeVOxjUAOUZJ4T4fqF6i2Vuq7n854onoZAE7pFxSzPg9d_6lQ/viewform?usp=dialog)"
)

col1, col2 = st.columns([1, 2])

with col1:
    st.image(str(resource_path("images/image.png")), use_container_width=True)

with col2:
    st.markdown("### How to Get Started:")
    st.markdown(
        """
    1. Choose a module from the **sidebar**.
    2. Define your experiment variables and objectives.
    3. Run a manual Bayesian Optimization campaign or explore the database.
    4. Dive into learning materials or review saved runs.

    ---
    """
    )
    st.success("Ready to get started? Select a module from the sidebar!")
