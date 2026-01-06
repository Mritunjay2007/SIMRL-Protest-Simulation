# 🧠 SIRML – Modeling Protest Dynamics with Media & Leadership Effects

## 📌 Overview
**SIRML** is an interactive simulation framework that models the **rise, spread, and decline of large-scale protests** using an extended epidemiological approach.

Building on the classical **SIR (Susceptible–Infected–Recovered)** model, this project introduces two critical real-world drivers:
- **M (Media Influence)** – time-varying amplification of emotional spread
- **L (Leadership)** – organizational forces that shape protest persistence and stabilization

The model is implemented as a **Streamlit web application** with real-world datasets, numerical simulation, and parameter fitting.

---

## 🎯 Motivation
Protests often behave like **contagious social processes**:
- Emotions spread through social contact
- Media amplifies participation
- Leadership structures influence duration and intensity
- Events trigger sudden surges

SIRML provides a **mathematical and data-driven framework** to study these dynamics in a unified system.

---

## 🧩 Model Description (SIRML)

### State Variables
- **S(t)** – Calm / susceptible population  
- **I(t)** – Emotionally activated / protesting population  
- **R(t)** – Recovered / disengaged population  
- **L(t)** – Leadership / organizational strength  
- **M(t)** – Media intensity (external input)

### Core Equations
\[
\beta(t) = \beta_0 (1 + \kappa M(t))
\]

\[
\frac{dS}{dt} = -\beta(t)\frac{SI}{N} + \omega R
\]

\[
\frac{dI}{dt} = \beta(t)\frac{SI}{N} - \gamma I - \mu \frac{LI}{N}
\]

\[
\frac{dR}{dt} = \gamma I + \mu \frac{LI}{N} - \omega R
\]

\[
\frac{dL}{dt} = \lambda_L - \eta_L L + \sigma_L E(t)
\]

---

## 🖥️ Application Features

### 🔹 Simulation Mode
- Interactive sliders for all model parameters
- Time-varying media cycles and event shocks
- Numerical ODE solving using `solve_ivp`
- Dynamic visualization of S, I, R, L, and M
- Automatic peak-intensity detection

### 🔹 Case Study Mode
Includes **10 real-world protest movements**, such as:
- George Floyd / BLM (2020)
- Climate Strikes (2019)
- Hong Kong Protests (2014, 2019)
- Egypt Revolution (2011)
- Yellow Vests (France)
- Chile Protests (2019)

Each case includes:
- Time-series data for S, I, R, L, M
- Dual-axis visualizations
- Normalized comparisons across cases

### 🔹 Parameter Fitting
- Fits model parameters to real data using **nonlinear least squares**
- Reconstructs best-fit protest dynamics
- Enables theory-to-data validation

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit  
- **Numerics:** NumPy, SciPy (`solve_ivp`, `least_squares`)  
- **Data:** pandas  
- **Visualization:** Plotly  

---

## 🚀 How to Run
```bash
pip install streamlit numpy scipy pandas plotly
streamlit run app.py
