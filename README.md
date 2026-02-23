# NLP Optimization Simulator

A Streamlit-based interactive simulator that visualizes how initial values, tolerance, and maximum iterations affect Non-Linear Programming (NLP) results using Pyomo and Ipopt.

## 🚀 Key Features
- **Individual Analysis**: Visualize the optimization path on 3D surfaces and 2D contour maps.
- **Sensitivity Testing**: Batch compare multiple scenarios with different initial settings.
- **Optimality Check**: Automatically identifies "Global Best" candidates vs "Local Optima."

## 🛠 Tech Stack
- **Python** (3.8+)
- **Streamlit** (UI)
- **Pyomo** (Modeling)
- **Ipopt** (Non-linear Solver)
- **Matplotlib & Plotly** (Visualization)

## 📦 Installation & Setup

1. **Install Ipopt Solver**:
   - Mac: `brew install ipopt`
   - Linux: `sudo apt-get install coinor-libipopt-dev`

2. **Clone and Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## 📖 Why this Simulator?
In Non-Linear Programming, the result often depends heavily on where the "walk" starts (Initial Values). This tool allows students and engineers to intuitively understand local vs. global optima through real-time visualization.
