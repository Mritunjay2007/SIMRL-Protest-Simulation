import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from scipy.optimize import least_squares
import plotly.graph_objects as go

st.set_page_config(page_title="SIRML Protest Model", layout="wide")
st.title("SIRML Model: Simulating Public Emotion Spread and Real-World Protests")

# Helper Functions
def sir_lm_rhs(t, y, p):
    S, I, R, L = y
    N, beta0, kappa, gamma, mu, omega, lambda_L, eta_L, sigma_L, M_fun, E_fun = p
    M = M_fun(t)
    E = E_fun(t)
    beta = beta0 * (1 + kappa * M)
    dS = -beta * S * I / N + omega * R
    dI = beta * S * I / N - gamma * I - mu * L * I / N
    dR = gamma * I + mu * L * I / N - omega * R
    dL = lambda_L - eta_L * L + sigma_L * E
    return [dS, dI, dR, dL]

def make_media_function(kind, const, amp, freq):
    if kind == "Constant":
        return lambda t: const
    elif kind == "Sinusoid":
        return lambda t: max(0.0, const + amp * np.sin(2 * np.pi * freq * t))
    else:
        return lambda t: const

def make_event_function(times, amps, decay=0.7):
    if len(times) == 0:
        return lambda t: 0.0
    def E(t):
        total = 0
        for ti, ai in zip(times, amps):
            if t >= ti:
                total += ai * np.exp(-decay * (t - ti))
        return total
    return E


# 1. George Floyd / BLM (USA, 2020)
george_floyd_data = {
    "Date": ["2020-05-26", "2020-05-27", "2020-05-28", "2020-05-31", "2020-06-06", 
             "2020-06-14", "2020-06-20", "2020-06-30", "2020-07-15", "2020-08-22"],
    "S":    [330_970_000, 330_800_000, 330_000_000, 329_500_000, 327_000_000, 
             327_500_000, 328_000_000, 326_000_000, 327_500_000, 329_000_000],
    "I":    [     10_000,    150_000,    500_000,  1_500_000,  4_000_000, 
              3_500_000,  3_200_000,  3_000_000,  2_000_000,  1_000_000],
    "R":    [         0,     15_000,     50_000,    200_000,    500_000, 
            1_000_000,  1_500_000,  2_000_000,  2_500_000,  3_000_000],
    "L":    [3, 4, 5, 6, 8, 8, 7, 7, 6, 4],
    "M":    [ 0.30,  0.50,  0.80,  0.90, 1.00,  0.90,  0.80,  0.70,  0.60,  0.40],
}
george_floyd_params = {"N": 331_000_000, "beta0": 0.15, "kappa": 0.40, "gamma": 0.05, "mu": 0.10}

# 2. Climate Strikes (Global, 2019)
climate_strike_data = {
    "Date": ["2019-03-15", "2019-05-24", "2019-08-01", "2019-09-15", "2019-09-20", "2019-09-27", 
             "2019-10-15", "2019-11-29", "2019-12-15", "2020-01-31"],
    "S":    [7_699_000_000, 7_699_200_000, 7_699_100_000, 7_697_500_000, 7_696_000_000, 7_698_000_000,
             7_698_500_000, 7_698_000_000, 7_698_800_000, 7_699_500_000],
    "I":    [  1_600_000,    800_000,  1_200_000, 2_500_000, 4_000_000,  2_000_000,
               1_500_000,  2_000_000,  1_000_000,    500_000],
    "R":    [         0,    500_000,    800_000, 1_000_000, 1_000_000,  1_500_000,
              2_000_000,  1_800_000,  2_500_000,  3_000_000],
    "L":    [5, 5, 6, 7, 8, 7, 6, 6, 5, 3],
    "M":    [ 0.50,  0.40,  0.60, 0.80, 1.00,  0.90,  0.80,  0.70,  0.50,  0.30],
}
climate_strike_params = {"N": 7_700_000_000, "beta0": 0.05, "kappa": 0.50, "gamma": 0.10, "mu": 0.05}

# 3. Yellow Vests (France, 2018–19)
yellow_vests_data = {
    "Date": ["2018-11-17", "2018-11-24", "2018-12-01", "2018-12-08", "2018-12-15", 
             "2019-01-12", "2019-02-02", "2019-03-09", "2019-03-29", "2019-06-01"],
    "S":    [ 66_700_000,  66_800_000,  66_850_000,  66_870_000,  66_880_000,
              66_910_000,  66_920_000,  66_930_000,  66_935_000,  66_950_000],
    "I":    [    282_000,    166_000,    136_000,    125_000,     66_000,
                 84_000,     50_000,     33_700,     30_000,     15_000],
    "R":    [         0,     20_000,     40_000,     50_000,     60_000,
                 60_000,     70_000,     75_000,     80_000,     85_000],
    "L":    [4, 5, 6, 6, 5, 5, 4, 3, 3, 2],
    "M":    [ 0.60,  0.80, 1.00,  0.90,  0.80,  0.70,  0.60,  0.40,  0.30,  0.20],
}
yellow_vests_params = {"N": 67_000_000, "beta0": 0.10, "kappa": 0.25, "gamma": 0.15, "mu": 0.20}

# 4. Ferguson BLM (USA, 2014)
ferguson_data = {
    "Date": ["2014-08-10", "2014-08-15", "2014-08-18", "2014-09-01", "2014-10-11", 
             "2014-10-13", "2014-11-15", "2014-11-25", "2014-12-13", "2014-12-31"],
    "S":    [317_980_000, 317_500_000, 317_200_000, 317_350_000, 317_200_000,
             317_150_000, 317_300_000, 317_500_000, 317_750_000, 317_800_000],
    "I":    [     15_000,    100_000,    200_000,    120_000,    150_000,
                180_000,    100_000,     90_000,     80_000,     50_000],
    "R":    [         0,     50_000,    150_000,    200_000,    180_000,
                200_000,    220_000,    240_000,    170_000,    250_000],
    "L":    [3, 5, 6, 5, 5, 5, 4, 4, 4, 3],
    "M":    [ 0.40,  0.80,  0.90,  0.70,  0.70,  0.70,  0.60,  0.60,  0.60,  0.30],
}
ferguson_params = {"N": 318_000_000, "beta0": 0.12, "kappa": 0.35, "gamma": 0.08, "mu": 0.12}

# 5. Hong Kong Umbrella (2014)
hongkong_data = {
    "Date": ["2014-09-28", "2014-10-01", "2014-10-08", "2014-10-15", "2014-11-01", 
             "2014-11-10", "2014-11-20", "2014-12-01", "2014-12-06", "2014-12-11"],
    "S":    [ 7_290_000,  7_250_000,  7_220_000,  7_200_000,  7_170_000,
              7_150_000,  7_160_000,  7_175_000,  7_180_000,  7_180_000],
    "I":    [     40_000,    150_000,    180_000,    200_000,    170_000,
                150_000,    130_000,    110_000,    105_000,    100_000],
    "R":    [         0,     50_000,    100_000,    150_000,    180_000,
                200_000,    220_000,    240_000,    250_000,    250_000],
    "L":    [5, 7, 8, 8, 7, 7, 6, 6, 6, 6],
    "M":    [ 0.70,  0.90, 1.00, 1.00,  0.90,  0.80,  0.70,  0.70,  0.60,  0.60],
}
hongkong_params = {"N": 7_300_000, "beta0": 0.14, "kappa": 0.30, "gamma": 0.04, "mu": 0.11}

# 6. Women's March (USA, 2017)
womens_march_data = {
    "Date": ["2017-01-20", "2017-01-21", "2017-01-22", "2017-01-25", "2017-01-28", 
             "2017-02-01", "2017-02-04", "2017-02-10", "2017-02-20", "2017-02-28"],
    "S":    [325_500_000, 321_800_000, 323_000_000, 323_500_000, 324_000_000,
             324_300_000, 325_000_000, 325_200_000, 325_400_000, 325_600_000],
    "I":    [    100_000,    500_000,  2_500_000,  1_500_000,    800_000,
                600_000,    500_000,    400_000,    300_000,    200_000],
    "R":    [         0,    100_000,  1_000_000,  1_200_000,  1_500_000,
              1_700_000,  1_800_000,  1_900_000,  2_000_000,  2_100_000],
    "L":    [6, 8, 8, 7, 7, 6, 6, 5, 4, 3],
    "M":    [ 0.80, 1.00,  0.90,  0.80,  0.70,  0.60,  0.60,  0.50,  0.40,  0.30],
}
womens_march_params = {"N": 326_000_000, "beta0": 0.20, "kappa": 0.45, "gamma": 0.12, "mu": 0.08}

# 7. Egypt Revolution (2011)
egypt_2011_data = {
    "Date": [
        "2011-01-01","2011-01-15","2011-02-01","2011-02-11","2011-03-01",
        "2011-04-01","2011-05-01","2011-06-01","2011-07-01","2011-08-01"
    ],
    "S": [80000000,79700000,79100000,79200000,79400000,79550000,79650000,79720000,79780000,79820000],
    "I": [800,2500,8800,8000,6000,4000,3000,2200,1500,1200],
    "R": [50000,100000,200000,280000,350000,380000,390000,395000,398000,400000],
    "L": [2,3,8,9,7,5,3,2,2,1],
    "M": [0.15,0.35,0.92,0.85,0.68,0.45,0.35,0.28,0.22,0.18]
}
egypt_2011_params = {"N": 80_000_000, "beta0": 0.20, "kappa": 0.40, "gamma": 0.07, "mu": 0.15}

# 8. Hong Kong Protests (2019)
hongkong_2019_data = {
    "Date": [
        "2019-06-09","2019-06-16","2019-06-23","2019-07-01","2019-07-15",
        "2019-08-01","2019-08-18","2019-09-15","2019-10-01","2019-11-15"
    ],
    "S": [6500000,5500000,6200000,6600000,6800000,6850000,6900000,6950000,7000000,7050000],
    "I": [1030000,2000000,850000,550000,420000,380000,350000,320000,280000,200000],
    "R": [500000,1000000,1500000,1800000,2000000,2100000,2150000,2180000,2200000,2220000],
    "L": [8,9,8,7,7,6,6,5,5,4],
    "M": [0.95,1.0,0.85,0.78,0.72,0.70,0.68,0.65,0.60,0.50]
}
hongkong_2019_params = {"N": 7_300_000, "beta0": 0.18, "kappa": 0.40, "gamma": 0.05, "mu": 0.12}

# 9. Chile Protests (2019–20)
chile_2019_data = {
    "Date": [
        "2019-10-19","2019-10-25","2019-11-01","2019-11-15","2019-12-01",
        "2019-12-15","2020-01-15","2020-02-15","2020-03-01","2020-03-15"
    ],
    "S": [66300000,66100000,65900000,66000000,66200000,66350000,66500000,66600000,66700000,66800000],
    "I": [450,1800,2100,1900,1400,1200,950,800,650,500],
    "R": [200,600,1200,1800,2200,2600,2900,3100,3250,3350],
    "L": [4,7,8,8,7,6,5,5,4,3],
    "M": [0.50,0.95,1.0,0.90,0.75,0.70,0.60,0.55,0.48,0.40]
}
chile_2019_params = {"N": 67_000_000, "beta0": 0.10, "kappa": 0.35, "gamma": 0.04, "mu": 0.10}

# 10. Brazil Protests (2013)
brazil_2013_hard_data = {
    "Date": [
        "2013-06-06","2013-06-13","2013-06-20","2013-06-27","2013-07-03",
        "2013-07-10","2013-07-17","2013-07-24","2013-07-31","2013-08-07"
    ],
    "S": [200900000,200700000,200000000,200250000,200450000,
          200600000,200700000,200800000,200850000,200900000],
    "I": [50000,200000,1000000,750000,500000,350000,250000,200000,150000,100000],
    "R": [100000,300000,700000,1000000,1200000,
          1300000,1350000,1370000,1380000,1385000],
    "L": [3,5,8,7,6,5,4,3,3,2],
    "M": [0.35,0.75,1.00,0.85,0.70,0.60,0.50,0.42,0.35,0.28]
}
brazil_2013_hard_params = {"N": 202_000_000, "beta0": 0.18, "kappa": 0.40, "gamma": 0.05, "mu": 0.08}

CASE_STUDIES = {
    "1. George Floyd / BLM (USA, 2020)": (george_floyd_data, george_floyd_params),
    "2. Climate Strikes (Global, 2019)": (climate_strike_data, climate_strike_params),
    "3. Yellow Vests (France, 2018–19)": (yellow_vests_data, yellow_vests_params),
    "4. Ferguson BLM (USA, 2014)": (ferguson_data, ferguson_params),
    "5. Hong Kong Umbrella (2014)": (hongkong_data, hongkong_params),
    "6. Women's March (USA, 2017)": (womens_march_data, womens_march_params),
    "7. Egypt Revolution (2011)": (egypt_2011_data, egypt_2011_params),
    "8. Hong Kong Protests (2019)": (hongkong_2019_data, hongkong_2019_params),
    "9. Chile Protests (2019)": (chile_2019_data, chile_2019_params),
    "10. Brazil Protests (2013 – Hardcoded)": (brazil_2013_hard_data, brazil_2013_hard_params),

}

# Mode Selection
mode = st.radio("Choose Mode:", ["Simulation", "Case Studies"])

# SIMULATION MODE
if mode == "Simulation":
    with st.sidebar:
        st.header("Simulation Controls")
        N = st.number_input("Total Population N", min_value=1000, value=1_000_000, step=10_000)
        S0 = st.number_input("S(0)", 0.0, N*1.0, 990_000.0, 1_000.0)
        I0 = st.number_input("I(0)", 0.0, N*1.0, 10_000.0, 1_000.0)
        R0 = st.number_input("R(0)", 0.0, N*1.0, 0.0, 1_000.0)
        L0 = st.number_input("L(0)", 0.0, 100.0, 5.0, 1.0)

        beta0 = st.slider("β₀ (Base Spread Rate)", 0.0, 1.0, 0.2, 0.01)
        st.caption("Low β₀ → Emotions spread slowly.\nHigh β₀ → Faster contagion.")

        kappa = st.slider("κ (Media Amplification)", 0.0, 1.0, 0.3, 0.01)
        st.caption("κ = 0 → Media has no effect.\nHigh κ → Media strongly amplifies emotions.")

        gamma = st.slider("γ (Recovery Rate)", 0.01, 1.0, 0.1, 0.01)
        st.caption("Low γ → Emotions linger longer.\nHigh γ → Faster calming down.")

        mu = st.slider("μ (Leadership Calming Effect)", 0.0, 1.0, 0.2, 0.01)
        st.caption("μ = 0 → Leaders have no calming power.\nHigh μ → Strong leadership influence.")

        omega = st.slider("ω (Loss of calm)", 0.0, 0.1, 0.01, 0.005)
        st.caption("Low ω → People remain calm longer.\nHigh ω → Quicker relapse to emotional state.")

        lambda_L = st.slider("λ_L (Leader Recruitment)", 0.0, 0.2, 0.01, 0.005)
        eta_L = st.slider("η_L (Leader Decay)", 0.0, 0.5, 0.05, 0.01)
        sigma_L = st.slider("σ_L (Event → Leadership)", 0.0, 0.2, 0.02, 0.005)

        M_const = st.slider("M baseline", 0.0, 1.0, 0.5, 0.05)
        M_amp = st.slider("M amplitude", 0.0, 1.0, 0.3, 0.05)
        M_freq = st.slider("M frequency", 0.0, 1.0, 0.2, 0.01)

        event_times = [15.0, 45.0]
        event_amps = [1.0, 1.0]
        t_end = st.slider("Simulation Duration", 10.0, 120.0, 60.0, 1.0)

    M_fun = make_media_function("Sinusoid", M_const, M_amp, M_freq)
    E_fun = make_event_function(event_times, event_amps)
    params = (N, beta0, kappa, gamma, mu, omega, lambda_L, eta_L, sigma_L, M_fun, E_fun)
    y0 = [S0, I0, R0, L0]
    t_eval = np.linspace(0, t_end, 600)
    sol = solve_ivp(lambda t, y: sir_lm_rhs(t, y, params), (0, t_end), y0, t_eval=t_eval)

    t = sol.t
    S, I, R, L = sol.y
    M_vals = np.array([M_fun(tt) for tt in t])

    st.subheader("Select factors to plot vs time")
    options = st.multiselect(
        "Choose which variables to display:",
        ["S (Calm)", "I (Emotional)", "R (Recovered)", "L (Leadership)", "M (Media)"],
        default=["S (Calm)", "I (Emotional)", "R (Recovered)", "L (Leadership)", "M (Media)"]
    )

    fig = go.Figure()
    if "S (Calm)" in options:
        fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='S (Calm)'))
    if "I (Emotional)" in options:
        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='I (Emotional)'))
    if "R (Recovered)" in options:
        fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='R (Recovered)'))
    if "L (Leadership)" in options:
        fig.add_trace(go.Scatter(x=t, y=L, mode='lines', name='L (Leadership)'))
    if "M (Media)" in options:
        fig.add_trace(go.Scatter(x=t, y=M_vals, mode='lines', name='M (Media)', line=dict(dash='dot')))

    fig.update_layout(title='Selected Variables vs Time', xaxis_title='Time (t)', yaxis_title='Level / Fraction', height=500)

    st.plotly_chart(fig, use_container_width=True)

    I_peak = float(np.max(I))
    t_peak = float(t[np.argmax(I)])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak Emotional Level (I)", f"{I_peak:.3f}")
    with col2:
        st.metric("Time of Peak Emotion", f"t = {t_peak:.2f}")

# CASE STUDIES MODE
elif mode == "Case Studies":
    st.subheader("📊 10 Real-World Protest Case Studies (Full 10 Data Points)")

    case_name = st.selectbox(
        "Select a Case Study:",
        list(CASE_STUDIES.keys())
    )

    data_dict, param_dict = CASE_STUDIES[case_name]
    df = pd.DataFrame(data_dict)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Normalised / scaled versions for plotting
    N_case = param_dict["N"]
    df["S_frac"] = df["S"] / N_case
    df["I_frac"] = df["I"] / N_case
    df["R_frac"] = df["R"] / N_case
    df["L_scaled"] = df["L"] / df["L"].max()   # 0–1 range, comparable to M


    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"**Selected Case:** {case_name}")

    with colB:
        fig = go.Figure()

        # Left axis: S, I, R (people)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["S"],
        name="S (Susceptible)",
        mode="lines+markers",
        line=dict(color="green"),
        marker=dict(size=6),
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["I"],
        name="I (Infected/Emotional)",
        mode="lines+markers",
        line=dict(color="red"),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["R"],
        name="R (Recovered)",
        mode="lines+markers",
        line=dict(color="blue"),
        marker=dict(size=6),
    ))

# Right axis: L, M (indices)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["L"],
        name="L (Leadership index)",
        mode="lines+markers",
        line=dict(color="orange"),
        marker=dict(size=6),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["M"],
        name="M (Media index)",
        mode="lines+markers",
        line=dict(color="purple", dash="dot"),
        marker=dict(size=6),
        yaxis="y2",
    ))    

    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="People (S, I, R)"),
        yaxis2=dict(
            title="Leadership / Media index",
            overlaying="y",
            side="right"
        ),
        height=500,
        hovermode="closest",  # default; can even omit this line
    )


    st.subheader("🔍 Fit Model to Data")

    if st.button("Best fit parameters"):

        st.info("Fitting all parameters... this may take longer.")

        t_data = (df.index - df.index[0]).days.values

        I_true = df["I"].values / N_case
        R_true = df["R"].values / N_case
        L_true = df["L"].values / df["L"].max()

        S0, I0, R0, L0 = df.iloc[0][["S","I","R","L"]].values
        y0 = [S0, I0, R0, L0]

        M_vals = df["M"].values
        M_fun = lambda t: np.interp(t, t_data, M_vals)
        E_fun = lambda t: 0.0

        def simulate(params):
            beta0, kappa, gamma, mu, omega, lambda_L, eta_L, sigma_L = params

            model_params = (N_case, beta0, kappa, gamma, mu,
                            omega, lambda_L, eta_L, sigma_L,
                            M_fun, E_fun)

            sol = solve_ivp(lambda t, y: sir_lm_rhs(t, y, model_params),
                            (0, max(t_data)), y0,
                            t_eval=t_data)

            S, I, R, L = sol.y
            L_frac = L / np.max(L) if np.max(L) > 0 else L

            return I / N_case, R / N_case, L_frac

        def loss(params):
            I_model, R_model, L_model = simulate(params)

            return np.concatenate([
                I_model - I_true,
                R_model - R_true,
                L_model - L_true
        ])

        start = [
            param_dict["beta0"],
            param_dict["kappa"],
            param_dict["gamma"],
            param_dict["mu"],
            0.01,   # omega
            0.01,   # lambda_L
            0.05,   # eta_L
            0.05    # sigma_L
        ]

        #Bounds to prevent insane behavior
        bounds = (
            [0.001, 0.0, 0.001, 0.0, 0.0,   0.0,   0.001, 0.0],
            [2.0,   3.0,  1.0,   3.0,  1.0,   1.0,   2.0,   2.0]
        )

        result = least_squares(loss, start, bounds=bounds)

        beta0_f, kappa_f, gamma_f, mu_f, omega_f, lambda_L_f, eta_L_f, sigma_L_f = result.x

        st.success("Fitting complete!")

        #Display best-fit values
        st.write("### Best-fit parameters:")
        st.write(f"β₀ (Base emotional contagion rate) = {beta0_f:.4f}")
        st.write(f"κ (Media amplification factor) = {kappa_f:.4f}")
        st.write(f"γ (Recovery rate) = {gamma_f:.4f}")
        st.write(f"μ (Leadership moderation rate) = {mu_f:.4f}")
        st.write(f"ω (Calm relapse rate) = {omega_f:.4f}")
        st.write(f"λₗ (Leadership formation rate) = {lambda_L_f:.4f}")
        st.write(f"ηₗ (Leadership decay) = {eta_L_f:.4f}")
        st.write(f"σₗ (Even-based leadership spike) = {sigma_L_f:.4f}")

        t_dense = np.linspace(0, max(t_data), 300)

        model_params = (N_case, beta0_f, kappa_f, gamma_f, mu_f,
                        omega_f, lambda_L_f, eta_L_f, sigma_L_f,
                        M_fun, E_fun)

        sol = solve_ivp(lambda t, y: sir_lm_rhs(t, y, model_params),
                        (0, max(t_data)), y0,
                        t_eval=t_dense)

        S_sim, I_sim, R_sim, L_sim = sol.y
        M_sim = np.array([M_fun(t) for t in t_dense])

        fig_fit = go.Figure()

        fig_fit.add_trace(go.Scatter(
            x=df.index[0] + pd.to_timedelta(t_dense, unit="D"),
            y=sol.y[0],
            name="Fitted S",
            mode="lines",
            line=dict(color="green")
        ))

        fig_fit.add_trace(go.Scatter(
            x=df.index[0] + pd.to_timedelta(t_dense, unit="D"),
            y=sol.y[1],
            name="Fitted I",
            mode="lines",
            line=dict(color="red")
        ))

        fig_fit.add_trace(go.Scatter(
            x=df.index[0] + pd.to_timedelta(t_dense, unit="D"),
            y=sol.y[2],
            name="Fitted R",
            mode="lines",
            line=dict(color="blue")
        ))

        fig_fit.add_trace(go.Scatter(
            x=df.index[0] + pd.to_timedelta(t_dense, unit="D"),
            y=sol.y[3],
            name="Fitted L",
            mode="lines",
            line=dict(color="orange"),
            yaxis="y2"
        ))

        fig_fit.add_trace(go.Scatter(
            x=df.index[0] + pd.to_timedelta(t_dense, unit="D"),
            y=M_sim,
            name="Fitted M",
            mode="lines",
            line=dict(color="purple", dash="dot"),
            yaxis="y2"
        ))

        fig_fit.update_layout(
            title="Fitted SIRLM Model Output",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Population (S, I, R)"),
            yaxis2=dict(title="Leadership / Media (L, M)", overlaying="y", side="right"),
            height=500
        )        

        st.plotly_chart(fig_fit, use_container_width=True)


    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

