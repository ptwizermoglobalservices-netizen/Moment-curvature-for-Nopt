import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Moment-Curvature Analysis", layout="wide")

st.title("Moment–Curvature Analysis of RC Section")

# ==========================================================
# USER INPUTS
# ==========================================================

st.sidebar.header("Section Inputs")

b = st.sidebar.number_input("Section width b (mm)", value=300.0)
H = st.sidebar.number_input("Section height H (mm)", value=500.0)
As = st.sidebar.number_input("Tension reinforcement As (mm²)", value=1200.0)
Asp = st.sidebar.number_input("Compression reinforcement As' (mm²)", value=600.0)

st.sidebar.header("Material Inputs")

fck = st.sidebar.number_input("Concrete strength fck (MPa)", value=30.0)
fyk = st.sidebar.number_input("Steel yield strength fyk (MPa)", value=450.0)
Es = st.sidebar.number_input("Steel modulus Es (MPa)", value=200000.0)

ecu = st.sidebar.number_input("Concrete ultimate strain εcu", value=0.01, format="%.5f")
esu = st.sidebar.number_input("Steel ultimate strain εsu", value=0.05, format="%.5f")

cover = st.sidebar.number_input("Cover (mm)", value=50.0)

phi_max = st.sidebar.number_input("Maximum curvature", value=3e-4, format="%.6f")
n_points = st.sidebar.number_input("Number of curvature points", value=1000, step=100)

# ==========================================================
# MATERIAL PROPERTIES
# ==========================================================

alpha_cc = 0.85
gamma_c = 1.5
fcd = alpha_cc * fck / gamma_c

gamma_s = 1.15
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


def xi_function(eps_top, n=300):
    if eps_top <= 0:
        return 0.0

    eps = np.linspace(0.0, eps_top, n)
    sigma = np.array([concrete_stress(e) for e in eps])
    area = np.trapezoid(sigma, eps)

    return area / (fcd * eps_top)


def lambda_function(eps_top, xi, n=300):
    if eps_top <= 0 or xi <= 0:
        return 0.0

    eps = np.linspace(0.0, eps_top, n)
    sigma = np.array([concrete_stress(e) for e in eps])

    integral = np.trapezoid(sigma * (eps_top - eps), eps)

    return integral / (eps_top**2 * fcd * xi)


# ==========================================================
# BALANCED FAILURE AND Nopt
# ==========================================================

X_bal = ecu * d / (ecu + esu)
xi_bal = xi_function(ecu)

eps_sp_bal = ecu * (X_bal - delta) / X_bal
eps_s_bal = ecu * (X_bal - d) / X_bal

Cc_bal = b * X_bal * fcd * xi_bal
Fsp_bal = steel_stress(eps_sp_bal) * Asp
Fs_bal = steel_stress(eps_s_bal) * As

Nopt = Cc_bal + Fsp_bal + Fs_bal

N_values = {
    "N = 0": 0.0,
    "N = 0.8 Nopt": 0.8 * Nopt,
    "N = Nopt": Nopt,
    "N = 1.2 Nopt": 1.2 * Nopt
}

# ==========================================================
# ANALYSIS FUNCTIONS
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
    X_high = min(H, ecu / phi)

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


def moment_curvature_analysis(N0):
    results = []

    results.append({
        "Curvature (1/mm)": 0.0,
        "Moment (kN.m)": 0.0,
        "Xc (mm)": np.nan,
        "eps_top": 0.0,
        "eps_s": 0.0,
        "eps_sp": 0.0
    })

    phi_values = np.linspace(1e-8, phi_max, int(n_points))

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

        if abs(eps_s) >= esu or abs(eps_sp) >= esu:
            break

        M = (Mc + Msp + Ms) / 1e6

        results.append({
            "Curvature (1/mm)": phi,
            "Moment (kN.m)": M,
            "Xc (mm)": Xc,
            "eps_top": eps_top,
            "eps_s": eps_s,
            "eps_sp": eps_sp
        })

    return pd.DataFrame(results)


# ==========================================================
# RUN BUTTON
# ==========================================================

if st.button("Run Analysis"):

    st.subheader("Balanced Section Values")

    col1, col2, col3 = st.columns(3)

    col1.metric("Balanced neutral axis X_bal", f"{X_bal:.3f} mm")
    col2.metric("ξ_bal", f"{xi_bal:.4f}")
    col3.metric("Nopt", f"{Nopt / 1000:.3f} kN")

    all_results = {}
    final_rows = []

    fig, ax = plt.subplots(figsize=(9, 6))

    for label, N0 in N_values.items():
        df = moment_curvature_analysis(N0)
        all_results[label] = df

        last = df.iloc[-1]

        final_rows.append({
            "Axial load case": label,
            "N0 (kN)": N0 / 1000,
            "Final curvature (1/mm)": last["Curvature (1/mm)"],
            "Final moment (kN.m)": last["Moment (kN.m)"],
            "Final concrete strain": last["eps_top"],
            "Final tension steel strain": last["eps_s"],
            "Final compression steel strain": last["eps_sp"],
            "Final Xc (mm)": last["Xc (mm)"]
        })

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

    st.subheader("Final Strain Values")
    final_df = pd.DataFrame(final_rows)
    st.dataframe(final_df, use_container_width=True)

    st.subheader("Detailed Results")

    selected_case = st.selectbox("Select axial load case", list(all_results.keys()))
    st.dataframe(all_results[selected_case], use_container_width=True)

    csv = all_results[selected_case].to_csv(index=False)

    st.download_button(
        label="Download selected results as CSV",
        data=csv,
        file_name=f"{selected_case.replace(' ', '_').replace('=', '')}_results.csv",
        mime="text/csv"
    )
