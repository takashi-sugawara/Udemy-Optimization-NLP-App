import streamlit as st
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Page Config
st.set_page_config(page_title="NLP Optimization Simulator", layout="wide")

st.title("🚀 NLP Feasibility & Sensitivity Simulator")
st.markdown("""
This application is designed to visualize and analyze how **Initial Values**, **Tolerance**, and **Maximum Iterations** affect the results of Non-Linear Programming (NLP) optimization.
""")

# --- Solver Path Detection ---
def get_ipopt_path():
    import shutil
    import sys
    # 1. Check in PATH
    path = shutil.which("ipopt")
    if path: return path
    
    # 2. Check the directory of the current Python executable (Reliable for Conda)
    bin_dir = os.path.dirname(sys.executable)
    path_in_bin = os.path.join(bin_dir, "ipopt")
    if os.path.exists(path_in_bin): return path_in_bin

    # 3. Check common Conda/Linux paths
    for p in ["/home/adminuser/.conda/bin/ipopt", "/usr/bin/ipopt", "/opt/conda/bin/ipopt"]:
        if os.path.exists(p): return p
    return None

ipopt_bin = get_ipopt_path()


# --- Sidebar Settings ---
st.sidebar.header("🛠 Optimization Parameters")

# --- Scenario Data Initialization ---
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {
        "Scenario A: Standard": {"x": 0.0, "y": 0.0, "tol": 1e-8, "max_iter": 1000},
        "Scenario B: Boundary Area": {"x": 4.8, "y": 4.8, "tol": 1e-8, "max_iter": 1000},
        "Scenario C: Local Search": {"x": 3.0, "y": -3.0, "tol": 1e-8, "max_iter": 1000},
        "Scenario D: High Precision": {"x": 0.0, "y": 0.0, "tol": 1e-15, "max_iter": 5000}
    }

# Scenario Selection for Manual Testing
scenario_name = st.sidebar.selectbox(
    "Load Preset Parameters",
    ["Manual Adjustment"] + list(st.session_state.scenarios.keys())
)

# Reflect initial values based on scenario
if scenario_name in st.session_state.scenarios:
    s = st.session_state.scenarios[scenario_name]
    init_x, init_y, tol, max_iter = s["x"], s["y"], s["tol"], s["max_iter"]
else:
    init_x, init_y, tol, max_iter = 0.0, 0.0, 1e-8, 1000

# Input Widgets
st.sidebar.markdown("---")
x_init = st.sidebar.slider("Initial Value for x", -5.0, 5.0, float(init_x))
y_init = st.sidebar.slider("Initial Value for y", -5.0, 5.0, float(init_y))
tol_val = st.sidebar.number_input("Tolerance", value=float(tol), format="%.1e", key="main_tol")
max_iter_val = st.sidebar.number_input("Max Iterations", value=int(max_iter), key="main_iter")

# --- Optimization Logic ---
def solve_nlp(x_i, y_i, t, m):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-5, 5), initialize=x_i)
    model.y = pyo.Var(bounds=(-5, 5), initialize=y_i)
    
    model.obj = pyo.Objective(
        expr=pyo.cos(model.x + 1) + pyo.cos(model.x) * pyo.cos(model.y), 
        sense=pyo.maximize
    )
    
    # Use the detected binary path
    ipopt_exec = get_ipopt_path()
    if ipopt_exec:
        opt = SolverFactory('ipopt', executable=ipopt_exec)
    else:
        opt = SolverFactory('ipopt')

    opt.options['tol'] = t
    opt.options['max_iter'] = m
    
    start_time = time.time()
    results = opt.solve(model, tee=False)
    end_time = time.time()
    
    return model, end_time - start_time, results

# --- Tab Layout ---
tab1, tab2, tab3 = st.tabs(["🎯 Individual Test", "📊 Scenario Comparison", "📚 Knowledge Base"])

with tab1:
    if st.sidebar.button("Run Optimization"):
        with st.spinner("Solving..."):
            model, solve_time, results = solve_nlp(x_init, y_init, tol_val, max_iter_val)
            
            # Result Display
            col1, col2, col3 = st.columns(3)
            col1.metric("Objective Value", f"{pyo.value(model.obj):.6f}")
            col2.metric("Solve Time", f"{solve_time:.4f} sec")
            col3.metric("Solver Status", str(results.solver.termination_condition))
            
            st.write(f"**Optimal Solution:** x = `{pyo.value(model.x):.10f}`, y = `{pyo.value(model.y):.10f}`")

            # Visualization
            st.subheader("📊 Result Visualization")
            x_val = pyo.value(model.x)
            y_val = pyo.value(model.y)
            z_val = pyo.value(model.obj)

            x_range = np.linspace(-5, 5, 100)
            y_range = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.cos(X + 1) + np.cos(X) * np.cos(Y)

            fig = plt.figure(figsize=(12, 6))
            
            # 3D
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
            ax1.scatter(x_val, y_val, z_val, color='red', s=100, label='Optimal', edgecolors='white')
            ax1.set_title("3D Surface Plot")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            
            # 2D Contour
            ax2 = fig.add_subplot(122)
            cp = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(cp, ax=ax2)
            ax2.scatter(x_val, y_val, color='red', s=150, marker='*', label='Optimal', edgecolors='white')
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_title("Contour Map")
            
            st.pyplot(fig)
    else:
        st.info("Adjust parameters in the sidebar and click 'Run Optimization'.")

with tab2:
    st.subheader("🏁 Multi-Scenario Batch Comparison")
    st.markdown("Run all preset scenarios at once and compare the stability and quality of the solutions.")
    
    # Preset Editor
    with st.expander("⚙️ Edit Scenario Presets"):
        st.info("You can customize the batch scenario parameters here. Changes are saved for the current session.")
        for name in list(st.session_state.scenarios.keys()):
            st.markdown(f"##### {name}")
            col1, col2, col3, col4 = st.columns(4)
            st.session_state.scenarios[name]["x"] = col1.number_input(f"x", value=float(st.session_state.scenarios[name]["x"]), key=f"tab_x_{name}")
            st.session_state.scenarios[name]["y"] = col2.number_input(f"y", value=float(st.session_state.scenarios[name]["y"]), key=f"tab_y_{name}")
            st.session_state.scenarios[name]["tol"] = col3.number_input(f"tol", value=float(st.session_state.scenarios[name]["tol"]), format="%.1e", key=f"tab_tol_{name}")
            st.session_state.scenarios[name]["max_iter"] = col4.number_input(f"max_iter", value=int(st.session_state.scenarios[name]["max_iter"]), key=f"tab_iter_{name}")
    
    if st.button("Execute All Scenarios", type="primary"):
        all_results = []
        progress_bar = st.progress(0)
        
        scenarios_to_run = st.session_state.scenarios
        for i, (name, s) in enumerate(scenarios_to_run.items()):
            model, solve_time, results = solve_nlp(s["x"], s["y"], s["tol"], s["max_iter"])
            term_cond = str(results.solver.termination_condition)
            obj_raw = pyo.value(model.obj) if term_cond == "optimal" else None
            
            all_results.append({
                "Scenario Name": name,
                "Result Quality": "",
                "Obj Value": obj_raw,
                "Opt x": round(pyo.value(model.x), 6) if obj_raw is not None else "-",
                "Opt y": round(pyo.value(model.y), 6) if obj_raw is not None else "-",
                "Time(s)": round(solve_time, 4),
                "Status": term_cond
            })
            progress_bar.progress((i + 1) / len(scenarios_to_run))
        
        df = pd.DataFrame(all_results)
        
        # Quality Check Logic
        valid_objs = [res["Obj Value"] for res in all_results if res["Obj Value"] is not None]
        if valid_objs:
            max_obj = max(valid_objs)
            def judge_quality(row):
                if row["Obj Value"] is None: return "❌ Failed"
                if abs(row["Obj Value"] - max_obj) < 1e-4: return "🌟 Global Best"
                return "⛰️ Local Opt"
            df["Result Quality"] = df.apply(judge_quality, axis=1)
        
        st.dataframe(df.style.apply(lambda x: ['background-color: #d4edda' if 'Global' in str(v) else '' for v in x], axis=1))
        
        # Comparison Charts
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📈 Solve Time Comparison")
            st.bar_chart(df.set_index("Scenario Name")["Time(s)"])
        with col_g2:
            st.subheader("📍 Objective Value Comparison")
            plot_df = df.set_index("Scenario Name")["Obj Value"].fillna(0)
            st.bar_chart(plot_df)
        
        st.info("💡 **Interpretation**: Different Objective Values indicate that the initial values led to different 'peaks' (Local Optima).")

with tab3:
    st.subheader("� Knowledge Base: NLP Validation Checklist")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### 1. Initial Value Dependency
        - Compare Scenarios A and C.
        - Starting from different points can result in stopping at different peaks. This is the biggest challenge in NLP.
        
        ### 2. Numerical Precision (Tolerance)
        - Setting a very low tolerance (e.g., 1e-15) ensures a more precise vertex but increases computation time.
        """)

    with col_b:
        st.markdown("""
        ### 3. Scaling
        - If objective values differ by orders of magnitude (e.g., millions vs. 0.001), solvers may struggle. Normalize your variables for stability.
        
        ### 4. Feasibility & Constraints
        - Adding complex non-linear constraints can lead to situations where no possible solution exists. Start simple.
        """)
