import streamlit as st


st.markdown(
    """
<style>
.home-card {
    border-radius: 12px;
    padding: 14px 16px;
    border: 1px solid transparent;
    min-height: 128px;
    margin: 4px 0 10px 0;
}
.home-card h4 {
    margin: 0 0 6px 0;
    color: #111111;
    font-size: 1.1rem;
}
.home-card p {
    margin: 0;
    color: #1f2937;
    line-height: 1.5;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("BO Studio")
st.caption("A practical workspace for Bayesian Optimization in chemistry and process development.")

st.markdown(
    """
<div class="home-card" style="background:#f7f9fc;border-color:#d8e0ea;">
  <h4>What You Can Do Here</h4>
  <p>Design and run manual BO campaigns, analyze tradeoffs, store experiments, and learn BO fundamentals through a structured classroom.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Start Here")
st.markdown(
    """
<div class="home-card" style="background:#eaf4ff;border-color:#bfd8f6;min-height:unset;">
  <h4>Quick Start</h4>
  <p>1) Pick a section in the sidebar.</p>
  <p>2) Configure variables and campaign settings.</p>
  <p>3) Run optimization and inspect plots/metrics.</p>
  <p>4) Compare outcomes and save useful runs.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Section Guide")
c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
<div class="home-card" style="background:#eaf4ff;border-color:#bfd8f6;">
  <h4>Single Objective Optimization</h4>
  <p>Run campaigns to maximize one objective function with full control of settings and observations.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="home-card" style="background:#ecfdf5;border-color:#b7ebd0;">
  <h4>Multi Objective Optimization</h4>
  <p>Explore Pareto tradeoffs when optimizing more than one objective at once.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="home-card" style="background:#fff7ed;border-color:#fed7aa;">
  <h4>Data Analysis</h4>
  <p>Review campaign behavior, compare runs, and extract practical insights from results.</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
<div class="home-card" style="background:#f5f3ff;border-color:#ddd6fe;">
  <h4>Bayesian Optimization Classroom</h4>
  <p>Follow a guided path from intuition to advanced practice with realistic chemistry scenarios.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="home-card" style="background:#fef3c7;border-color:#fcd34d;">
  <h4>Experiment Database</h4>
  <p>Store, browse, and reuse campaigns so progress is persistent and traceable.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="home-card" style="background:#f3f4f6;border-color:#d1d5db;">
  <h4>Feedback</h4>
  <p>Have suggestions or found a bug? Use the form link below to report it quickly.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    "Feedback form: "
    "[Open feedback form](https://docs.google.com/forms/d/e/1FAIpQLSeVOxjUAOUZJ4T4fqF6i2Vuq7n854onoZAE7pFxSzPg9d_6lQ/viewform?usp=dialog)"
)

st.markdown(
    "Documentation: "
    "[Open BO Studio docs](https://gono-cl.github.io/bo-studio/index.html)"
)
