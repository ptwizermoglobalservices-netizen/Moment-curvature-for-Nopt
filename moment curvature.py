import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# STREAMLIT PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="Moment-Curvature Analysis",
    layout="wide"
)

st.title("Moment–Curvature Analysis of RC Section")

st.write("""
This app performs curvature-controlled moment–curvature analysis
for a rectangular reinforced concrete section under different axial loads.
""")

# ==========================================================
# USER INPUTS
# ==========================================================

st.sidebar.header("Section Inputs")

b = st.sidebar.number_input("Section width b (mm)", value=300.0, min_value=1.0)
H = st.sidebar.number_input("Section height H (mm)", value=500.0, min_value=1.0)

As = st.sidebar.number_input("Tension reinforcement As (mm²)", value=2000.0, min_value=0.0)
Asp = st.sidebar.number_input("Compression reinforcement As' (mm²)", value=1000.0, min_value=0.0)

cover = st.sidebar.number_input("Concrete cover (mm)", value=50.0, min_value=0.0)

st.sidebar.header("Material Properties")

fck = st.sidebar.number_input("Concrete strength fck (MPa)", value=30.0, min_value=1.0)
alpha_cc = st.sidebar.number_input("alpha_cc", value=0.85, min_value=0.0)
gamma_c = st.sidebar.number_input("gamma_c", value=1.5, min_value=0.1)

fyk = st.sidebar.number_input("Steel yield strength fyk (MPa)", value=450.0, min_value=1.0)
gamma_s = st.sidebar.number_input("gamma_s", value=1.15, min_value=0.1)
Es = st.sidebar.number_input("Steel modulus Es (MPa)", value=200000.0, min_value=1.0)

ecu = st.sidebar.number_input(
    "Ultimate concrete strain εcu",
    value=0.0035,
    format="%.5f"
)

st.sidebar.header("Analysis Settings")

max_phi = st.sidebar.number_input(
    "Maximum curvature φmax (1/mm)",
    value=5e-5,
    format="%.8f"
)

num_steps = st.sidebar.number_input(
    "Number of curvature steps",
    value=600,
    min_value=50,
    step=50
)

run_analysis = st.sidebar.button("Run Analysis")

# ==========================================================
# DERIVED VALUES
# ==========================================================

fcd = alpha_cc * fck / gamma_c
fyd = fyk / gamma_s
ey = fyd / Es

delta = cover
d = H - cover
Yg = H / 2.0

# ==========================================================
# MATERIAL MODELS
# ==========================================================

def steel_stress(eps):
    if eps >= ey:
        return fyd
    elif eps <= -ey:
        return -fyd
    else:
        return Es * eps


def concrete_stress(eps):
    if eps <= 0:
        return 0.0
    elif eps <= 0.002:
        return 1000.0 * eps * fcd * (1.0 - 250.0 * eps)
    elif eps <= ecu:
        return fcd
    else:
        return 0.0


# ==========================================================
# XI AND LAMBDA FUNCTIONS
# ==========================================================

def xi_function(eps_top, n=300):
    eps = np.linspace(0.0, eps_top, n)
    sigma = np.array([concrete_stress(e) for e in eps])

    area = np.trapezoid(sigma, eps)

    if fcd * eps_top == 0:
        return 0.0

    return area / (fcd * eps_top)


def lambda_function(eps_top, xi, n=300):
    eps = np.linspace(0.0, eps_top, n)
    sigma = np.array([concrete_stress(e) for e in eps])

    integral = np.trapezoid(
        sigma * (eps_top - eps),
        eps
    )

    denominator = eps_top**2 * fcd * xi

    if denominator == 0:
        return 0.0

    return integral / denominator


# ==========================================================
# EQUILIBRIUM FUNCTIONS
# ==========================================================

def axial_equilibrium(Xc, phi, N0):
    eps_top = phi * Xc

    if eps_top <= 0 or eps_top > ecu:
        return None

    xi = xi_function(eps_top)

    Cc = xi * fcd * b * Xc

    eps_sp = phi * (Xc - delta)
    eps_s = phi * (Xc - d)

    Fsp = steel_stress(eps_sp) * Asp
    Fs = steel_stress(eps_s) * As

    N = Cc + Fsp + Fs

    return N - N0


def solve_Xc(phi, N0, tol=1e-3, max_iter=100):
    X_low = 1e-6
    X_high = H

    f_low = axial_equilibrium(X_low, phi, N0)
    f_high = axial_equilibrium(X_high, phi, N0)

    if f_low is None or f_high is None:
        return None

    if f_low * f_high > 0:
        return None

    for _ in range(max_iter):
        X_mid = 0.5 * (X_low + X_high)
        f_mid = axial_equilibrium(X_mid, phi, N0)

        if f_mid is None:
            X_high = X_mid
            continue

        if abs(f_mid) < tol:
            return X_mid

        if f_low * f_mid < 0:
            X_high = X_mid
            f_high = f_mid
        else:
            X_low = X_mid
            f_low = f_mid

    return X_mid


# ==========================================================
# MOMENT-CURVATURE ANALYSIS
# ==========================================================

def moment_curvature_analysis(N0):
    results = []

    results.append({
        "Curvature (1/mm)": 0.0,
        "Moment (kN.m)": 0.0,
        "Xc (mm)": np.nan,
        "eps_top": 0.0
    })

    phi_values = np.linspace(1e-8, max_phi, int(num_steps))

    for phi in phi_values:

        Xc = solve_Xc(phi, N0)

        if Xc is None:
            continue

        eps_top = phi * Xc

        if eps_top >= ecu:
            break

        xi = xi_function(eps_top)
        lam = lambda_function(eps_top, xi)

        Cc = xi * fcd * b * Xc
        Mc = Cc * (Yg - lam * Xc)

        eps_sp = phi * (Xc - delta)
        Fsp = steel_stress(eps_sp) * Asp
        Msp = Fsp * (Yg - delta)

        eps_s = phi * (Xc - d)
        Fs = steel_stress(eps_s) * As
        Ms = Fs * (Yg - d)

        M = Mc + Msp + Ms
        M = M / 1e6

        results.append({
            "Curvature (1/mm)": phi,
            "Moment (kN.m)": M,
            "Xc (mm)": Xc,
            "eps_top": eps_top
        })

    return pd.DataFrame(results)


# ==========================================================
# RUN STREAMLIT APP
# ==========================================================

if run_analysis:

    if cover >= H:
        st.error("Cover must be smaller than section height H.")
        st.stop()

    X_bal = ecu * d / (ecu + ey)
    xi_bal = xi_function(ecu)

    Nopt = b * X_bal * fcd * xi_bal

    N_values = {
        "N = 0": 0.0,
        "N = 0.8 Nopt": 0.8 * Nopt,
        "N = Nopt": Nopt,
        "N = 1.2 Nopt": 1.2 * Nopt
    }

    st.subheader("Calculated Section Parameters")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("fcd", f"{fcd:.2f} MPa")
    col2.metric("fyd", f"{fyd:.2f} MPa")
    col3.metric("Yield strain εy", f"{ey:.6f}")
    col4.metric("Nopt", f"{Nopt/1000:.2f} kN")

    st.write(f"Balanced neutral axis depth Xbal = **{X_bal:.2f} mm**")
    st.write(f"ξ at balanced strain = **{xi_bal:.4f}**")

    all_results = {}

    fig, ax = plt.subplots(figsize=(9, 6))

    for label, N0 in N_values.items():
        df = moment_curvature_analysis(N0)
        all_results[label] = df

        ax.plot(
            df["Curvature (1/mm)"],
            df["Moment (kN.m)"],
            label=label
        )

    ax.set_xlabel("Curvature φ (1/mm)")
    ax.set_ylabel("Moment M (kN.m)")
    ax.set_title("Moment–Curvature Curves for Different Axial Loads")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    st.subheader("Analysis Results")

    selected_case = st.selectbox(
        "Select axial load case to view table",
        list(all_results.keys())
    )

    st.dataframe(all_results[selected_case])

    csv = all_results[selected_case].to_csv(index=False)

    st.download_button(
        label="Download selected results as CSV",
        data=csv,
        file_name=f"{selected_case.replace(' ', '_')}_moment_curvature.csv",
        mime="text/csv"
    )

else:
    st.info("Enter section details in the sidebar and click **Run Analysis**.")