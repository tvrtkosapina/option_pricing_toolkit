import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from option_pricing import (
    StohasticOrdinaryDiffEquation,
    set_GBM,
    MonteCarloOptionPricer,
    BinomialTreeOptionPricer,
    BSM_option_pricing,
)

st.set_page_config(
    page_title="Option Pricing Toolkit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #071A33;
        --blue: #1478FF;
        --cyan: #55CCFF;
        --panel: rgba(10, 35, 67, 0.88);
        --border: rgba(91, 184, 255, 0.25);
        --text: #ECF6FF;
        --muted: #A9C8E7;
    }

    .stApp {
        background:
            radial-gradient(circle at 88% 4%, rgba(20,120,255,.23), transparent 31%),
            linear-gradient(145deg, #06162B 0%, #0A2445 52%, #0D3158 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061429, #081D36);
        border-right: 1px solid var(--border);
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3, h4, p, label, .stMarkdown {
        color: var(--text);
    }

    .hero {
        padding: 1.55rem 1.75rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(20,120,255,.22), rgba(85,204,255,.06));
        box-shadow: 0 20px 55px rgba(0,0,0,.24);
    }

    .hero-kicker {
        color: var(--cyan);
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: .25rem 0 .3rem;
        color: white;
        font-size: 2.35rem;
        font-weight: 780;
        letter-spacing: -.035em;
    }

    .hero-copy { color: var(--muted); margin: 0; }

    .result-card {
        min-height: 118px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: var(--panel);
        box-shadow: 0 12px 32px rgba(0,0,0,.17);
    }

    .result-label { color: #8EBDE8; font-size: .82rem; }
    .result-value { color: white; font-size: 1.9rem; font-weight: 760; margin-top: .15rem; }
    .result-note { color: var(--cyan); font-size: .78rem; }

    .info-box {
        padding: .9rem 1rem;
        border-left: 4px solid var(--blue);
        border-radius: 12px;
        background: rgba(16, 67, 120, .45);
        color: #CFE8FF;
    }

    .stButton > button {
        width: 100%;
        border: 0;
        border-radius: 14px;
        color: white;
        font-weight: 700;
        background: linear-gradient(90deg, #1478FF, #34B9FF);
        box-shadow: 0 9px 26px rgba(20,120,255,.32);
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        color: white !important;
        background: #0A203B !important;
    }

    [data-testid="stTabs"] button { color: #B8D7F5; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #5CCBFF; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Quantitative Finance Project</div>
        <div class="hero-title">Option Pricing Toolkit</div>
        <p class="hero-copy">
            A unified interface for Black-Scholes, CRR binomial-tree,
            Monte Carlo and stochastic differential-equation methods.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


def result_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
            <div class="result-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Market inputs")
    S = st.number_input("Spot price (S₀)", min_value=0.01, value=100.0, step=1.0)
    K = st.number_input("Strike price (K)", min_value=0.01, value=100.0, step=1.0)
    T = st.number_input("Time to maturity (years)", min_value=0.001, value=1.0, step=0.25)
    r = st.number_input("Risk-free rate", value=0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility", min_value=0.001, value=0.20, step=0.01, format="%.4f")
    q = st.number_input("Dividend yield", value=0.02, step=0.005, format="%.4f")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True)

pricing_tab, sde_tab, notes_tab = st.tabs(
    ["Option pricing", "SDE solver", "Project notes"]
)

with pricing_tab:
    model = st.selectbox(
        "Pricing method",
        ["Black-Scholes", "CRR binomial tree", "Monte Carlo"],
    )

    if model == "Black-Scholes":
        st.markdown(
            '<div class="info-box">European call and put pricing using Black-Scholes formula.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Calculate Black-Scholes price"):
            try:
                price = BSM_option_pricing(S, K, T, r, sigma, q, option_type)
                st.session_state["bs_result"] = float(price)
            except Exception as error:
                st.error(f"Your Black-Scholes function returned an error: {error}")

        if "bs_result" in st.session_state:
            c1, c2, c3 = st.columns(3)
            with c1:
                result_card("Option price", f"{st.session_state['bs_result']:.4f}", "Black-Scholes")
            with c2:
                result_card("Moneyness", f"{S / K:.3f}", "S₀ / K")
            with c3:
                result_card("Intrinsic value", f"{max(S-K, 0) if option_type == 'call' else max(K-S, 0):.4f}", option_type.title())

            spots = np.linspace(max(0.01, S * 0.5), S * 1.5, 80)
            values = [BSM_option_pricing(x, K, T, r, sigma, q, option_type) for x in spots]
            fig, ax = plt.subplots(figsize=(10, 4.7))
            fig.patch.set_facecolor("#071A33")
            ax.set_facecolor("#0A2342")
            ax.plot(spots, values, color="#55CCFF", linewidth=2.8)
            ax.fill_between(spots, values, color="#1478FF", alpha=0.15)
            ax.set_title("Option price across spot prices", color="white")
            ax.set_xlabel("Spot price", color="#CFE8FF")
            ax.set_ylabel("Option value", color="#CFE8FF")
            ax.tick_params(colors="#B8D7F5")
            ax.grid(alpha=0.15)
            for spine in ax.spines.values():
                spine.set_color("#28517A")
            st.pyplot(fig, use_container_width=True)

    elif model == "CRR binomial tree":
        c1, c2 = st.columns(2)
        with c1:
            n = st.number_input("Number of tree steps", min_value=1, value=300, step=50)
            exercise_type = st.selectbox(
                "Exercise style", ["european", "american", "bermudan"]
            )
        with c2:
            bermudan_steps = st.text_input(
                "Bermudan exercise steps",
                value="75, 150, 225",
                disabled=exercise_type != "bermudan",
                help="Enter comma-separated tree-step indices.",
            )

        if st.button("Calculate CRR price"):
            try:
                early_exercise = (
                    [int(value.strip()) for value in bermudan_steps.split(",") if value.strip()]
                    if exercise_type == "bermudan"
                    else [0]
                )

                # This directly instantiates and calls your original class.
                pricer = BinomialTreeOptionPricer(
                    S, K, T, r, sigma, q, int(n),
                    option_type,
                    exercise_type,
                    early_exercise,
                )
                price = pricer.binomial_CRR_pricing_model()
                st.session_state["crr_result"] = float(price)
            except Exception as error:
                st.error(
                    "Your original CRR class returned an error. "
                    f"Error: {error}"
                )

        if "crr_result" in st.session_state:
            c1, c2, c3 = st.columns(3)
            with c1:
                result_card("Option price", f"{st.session_state['crr_result']:.4f}", "CRR tree")
            with c2:
                result_card("Exercise style", exercise_type.title(), f"{int(n)} steps")
            with c3:
                result_card("Intrinsic value", f"{max(S-K, 0) if option_type == 'call' else max(K-S, 0):.4f}", option_type.title())

    else:
        c1, c2 = st.columns(2)
        with c1:
            n_steps = st.number_input("SDE time steps", min_value=1, value=252, step=21)
        with c2:
            n_paths = st.number_input("Monte Carlo paths", min_value=100, value=10_000, step=1_000)

        st.caption(
            "The risk-neutral GBM drift passed to your SDE implementation is (r − q)."
        )

        if st.button("Run Monte Carlo pricing"):
            try:
                sde = set_GBM(
                    str(r - q),
                    str(sigma),
                    x0=0,
                    y0=S,
                    xn=T,
                    num_of_intervals=int(n_steps),
                )
                pricer = MonteCarloOptionPricer(sde, r, option_type)
                result = pricer.european(K, int(n_paths))
                if result is False:
                    raise ValueError("The original Monte Carlo method returned False.")
                price, standard_error = result
                st.session_state["mc_result"] = (float(price), float(standard_error))
            except Exception as error:
                st.error(
                    "Your original Monte Carlo class returned an error. "
                    f"Error: {error}"
                )

        if "mc_result" in st.session_state:
            price, standard_error = st.session_state["mc_result"]
            lower = price - 1.96 * standard_error
            upper = price + 1.96 * standard_error
            c1, c2, c3 = st.columns(3)
            with c1:
                result_card("Estimated price", f"{price:.4f}", "Monte Carlo")
            with c2:
                result_card("Standard error", f"{standard_error:.4f}", f"{int(n_paths):,} paths")
            with c3:
                result_card("95% confidence interval", f"{lower:.4f} – {upper:.4f}", "Normal approximation")

with sde_tab:
    st.subheader("Stochastic differential-equation solver")
    st.write("Solve SDEs using Euler-Maruyama, Tamed Euler and Milestein method.")

    c1, c2, c3 = st.columns(3)
    with c1:
        drift = st.text_input("Drift f(x, y)", value="0.05*y")
        y0 = st.number_input("Initial value y₀", value=100.0, key="sde_y0")
    with c2:
        diffusion = st.text_input("Diffusion g(x, y)", value="0.20*y")
        horizon = st.number_input("Final time", min_value=0.001, value=1.0, key="sde_T")
    with c3:
        method = st.selectbox("Numerical method", ["Euler-Maruyama", "Milstein", "Tamed Euler"])
        intervals = st.number_input("Number of intervals", min_value=1, value=252, step=21)

    if st.button("Solve SDE"):
        try:
            equation = StohasticOrdinaryDiffEquation(
                drift,
                diffusion,
                x0=0,
                y0=y0,
                xn=horizon,
                num_of_intervals=int(intervals),
            )

            if method == "Euler-Maruyama":
                path = equation.EulerMethod()
            elif method == "Milstein":
                path = equation.MilsteinMethod()
            else:
                path = equation.TamedEulerMethod()

            if path is False or path is None:
                raise ValueError("The original solver did not return a path.")

            times = np.linspace(0, horizon, int(intervals) + 1)
            st.session_state["sde_path"] = (times, np.asarray(path), method)
        except Exception as error:
            st.error(f"Your original SDE solver returned an error: {error}")

    if "sde_path" in st.session_state:
        times, path, chosen_method = st.session_state["sde_path"]
        c1, c2, c3 = st.columns(3)
        with c1:
            result_card("Terminal value", f"{path[-1]:.4f}", chosen_method)
        with c2:
            result_card("Path minimum", f"{np.min(path):.4f}", "Simulated path")
        with c3:
            result_card("Path maximum", f"{np.max(path):.4f}", "Simulated path")

        chart_data = pd.DataFrame({"Time": times, "State": path}).set_index("Time")
        st.line_chart(chart_data, color="#55CCFF", use_container_width=True)

with notes_tab:
    st.subheader("About")
    st.markdown(
        """
        - Thank you for using this option pricing tool!
        - This project was developed by Tvrtko Šapina
        - GitHub profile: https://github.com/tvrtkosapina
        """
    )
