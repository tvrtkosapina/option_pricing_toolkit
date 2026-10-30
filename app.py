import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SDE_solvers import StohasticOrdinaryDiffEquation, set_GBM
from MonteCarlo import MonteCarloOptionPricer
from BinomialTree import BinomialTreeOptionPricer
from BlackScholes import BlackScholesMertonOptionPricer


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CYAN, BLUE, ORANGE, MUTED = "#55CCFF", "#1478FF", "#FFB547", "#8EBDE8"
GREEK_ORDER = ["delta", "gamma", "vega", "theta", "rho"]
GREEK_INFO = {
    "delta": ("Delta", "∂V/∂S"),
    "gamma": ("Gamma", "∂²V/∂S²"),
    "vega": ("Vega", "∂V/∂σ"),
    "theta": ("Theta", "∂V/∂t, per year"),
    "rho": ("Rho", "∂V/∂r"),
}


def styled_figure(nrows=1, ncols=1, figsize=(10, 4.7)):
    """Matplotlib figure in the page's colour scheme."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("#071A33")
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("#0A2342")
        ax.tick_params(colors="#B8D7F5")
        ax.grid(alpha=0.15)
        for spine in ax.spines.values():
            spine.set_color("#28517A")
    return fig, axes


def show_figure(fig):
    """Colour titles/labels/legends, render, and free the figure's memory."""
    for ax in fig.axes:
        ax.title.set_color("white")
        ax.xaxis.label.set_color("#CFE8FF")
        ax.yaxis.label.set_color("#CFE8FF")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#0A2342")
            legend.get_frame().set_edgecolor("#28517A")
            for text in legend.get_texts():
                text.set_color("#CFE8FF")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def intrinsic(spot, strike, kind):
    return max(spot - strike, 0) if kind == "call" else max(strike - spot, 0)


def bs_pricer(spot, strike, maturity, rate, vol, div, kind):
    return BlackScholesMertonOptionPricer(spot, strike, maturity, rate, vol, div, kind)


def bs_greeks(pricer):
    return {
        "delta": pricer.BSMdelta(),
        "gamma": pricer.BSMgamma(),
        "vega": pricer.BSMvega(),
        "theta": pricer.BSMtheta(),
        "rho": pricer.BSMrho(),
    }


def greek_cards(greeks, reference=None):
    """One row of Greek cards. `reference` (optional) = Black-Scholes values shown for comparison."""
    columns = st.columns(len(GREEK_ORDER))
    for column, key in zip(columns, GREEK_ORDER):
        label, symbol = GREEK_INFO[key]
        value = greeks.get(key)
        with column:
            if value is None or not np.isfinite(value):
                result_card(label, "n/a", "needs n ≥ 2" if key in ("gamma", "theta") else "not available")
                continue
            if reference is not None:
                note = f"{symbol} · BS {reference[key]:.4f}"
            elif key == "vega":
                note = f"{symbol} · {value / 100:.4f} per 1% vol"
            elif key == "theta":
                note = f"{value / 365:.4f} per calendar day"
            elif key == "rho":
                note = f"{symbol} · {value / 100:.4f} per 1% rate"
            else:
                note = symbol
            result_card(label, f"{value:.4f}", note)


def stale_warning(stored_params, current_params):
    if stored_params != current_params:
        st.caption("⚠️ Inputs changed since the last calculation. Recalculate to update the results.")


def plot_binomial_tree(stock, value, exercised, steps, early_steps, exercise_kind, dt):
    """Draw a small CRR tree: every node shows the stock price S and the option value V."""
    fig, ax = styled_figure(figsize=(11, 1.6 + 1.1 * steps))
    for i in range(steps):
        for j in range(i + 1):
            for dj in (0, 1):
                ax.plot([i, i + 1], [2 * j - i, 2 * (j + dj) - (i + 1)],
                        color="#28517A", linewidth=1.4, zorder=1)

    if exercise_kind == "bermudan":
        for i in early_steps:
            if 0 <= i < steps:
                ax.axvspan(i - 0.32, i + 0.32, color=ORANGE, alpha=0.07, zorder=0)

    font = 10 if steps <= 3 else 8.5
    for i in range(steps + 1):
        for j in range(i + 1):
            is_exercise = exercised[i][j]
            ax.text(
                i, 2 * j - i,
                f"S = {stock[i][j]:.2f}\nV = {value[i][j]:.3f}",
                ha="center", va="center", fontsize=font, zorder=3,
                color="#071A33" if is_exercise else "white",
                bbox=dict(
                    boxstyle="round,pad=0.45",
                    facecolor=ORANGE if is_exercise else "#10335C",
                    edgecolor=ORANGE if is_exercise else CYAN,
                    linewidth=1.2,
                ),
            )

    ax.set_xticks(range(steps + 1))
    ax.set_xticklabels([f"step {i}\nt = {i * dt:.3f}" for i in range(steps + 1)])
    ax.set_yticks([])
    ax.set_xlim(-0.6, steps + 0.6)
    ax.set_ylim(-steps - 1, steps + 1)
    ax.grid(False)
    ax.set_title("CRR tree: stock price S and option value V at each node "
                 "(orange = early exercise is optimal)")
    show_figure(fig)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Market inputs")
    S = st.number_input("Spot price (S₀)", min_value=0.01, value=100.0, step=1.0)
    K = st.number_input("Strike price (K)", min_value=0.01, value=100.0, step=1.0)
    T = st.number_input("Time to maturity (years)", min_value=0.001, value=1.0, step=0.25)
    r = st.number_input("Risk-free rate", value=0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility", min_value=0.001, value=0.20, step=0.01, format="%.4f")
    q = st.number_input("Dividend yield", value=0.02, step=0.005, format="%.4f")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True)

market = (S, K, T, r, sigma, q, option_type)

pricing_tab, greeks_tab, sde_tab, notes_tab = st.tabs(
    ["Option pricing", "Greeks explorer", "SDE solver", "Project notes"]
)

# ---------------------------------------------------------------------------
# Option pricing
# ---------------------------------------------------------------------------

with pricing_tab:
    model = st.selectbox(
        "Pricing method",
        ["Black-Scholes", "CRR binomial tree", "Monte Carlo"],
    )

    if model == "Black-Scholes":
        st.markdown(
            '<div class="info-box">European call and put pricing using the Black-Scholes-Merton '
            'formula with continuous dividend yield, plus closed-form Greeks.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Calculate Black-Scholes price"):
            try:
                pricer = bs_pricer(*market)
                st.session_state["bs_result"] = {
                    "params": market,
                    "price": float(pricer.BSM_option_pricing()),
                    "greeks": {k: float(v) for k, v in bs_greeks(pricer).items()},
                }
            except Exception as error:
                st.error(f"Your Black-Scholes class returned an error: {error}")

        if "bs_result" in st.session_state:
            result = st.session_state["bs_result"]
            s0, k0, t0, r0, v0, q0, kind = result["params"]
            stale_warning(result["params"], market)

            c1, c2, c3 = st.columns(3)
            with c1:
                result_card("Option price", f"{result['price']:.4f}", "Black-Scholes")
            with c2:
                result_card("Moneyness", f"{s0 / k0:.3f}", "S₀ / K")
            with c3:
                result_card("Intrinsic value", f"{intrinsic(s0, k0, kind):.4f}", kind.title())

            st.markdown("#### Greeks")
            greek_cards(result["greeks"])

            spots = np.linspace(max(0.01, s0 * 0.5), s0 * 1.5, 80)
            values = [bs_pricer(x, k0, t0, r0, v0, q0, kind).BSM_option_pricing() for x in spots]
            payoff = [intrinsic(x, k0, kind) for x in spots]
            fig, ax = styled_figure()
            ax.plot(spots, values, color=CYAN, linewidth=2.8, label="Option value today")
            ax.fill_between(spots, values, color=BLUE, alpha=0.15)
            ax.plot(spots, payoff, color=MUTED, linewidth=1.4, linestyle="--", label="Payoff at maturity")
            ax.axvline(s0, color=ORANGE, linewidth=1, alpha=0.8, label="Current spot")
            ax.set_title("Option price across spot prices")
            ax.set_xlabel("Spot price")
            ax.set_ylabel("Option value")
            ax.legend()
            show_figure(fig)

    elif model == "CRR binomial tree":
        st.markdown(
            '<div class="info-box">Cox-Ross-Rubinstein tree for European, American and Bermudan options. '
            'Delta, gamma and theta are read from the first two levels of the tree; vega and rho are '
            'computed by re-pricing with bumped volatility and rate. Use 5 steps or fewer to see the '
            'whole tree.</div>',
            unsafe_allow_html=True,
        )
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
                help="Comma-separated tree-step indices between 0 and n − 1 at which early exercise "
                     "is allowed. Exercise at maturity (step n) is always allowed.",
            )
            show_convergence = st.checkbox(
                "Show convergence to Black-Scholes",
                value=False,
                help="Re-prices the option for every n from 1 to 200 and compares it with the "
                     "Black-Scholes European price.",
            )

        if st.button("Calculate CRR price"):
            try:
                early_exercise = (
                    [int(value.strip()) for value in bermudan_steps.split(",") if value.strip()]
                    if exercise_type == "bermudan"
                    else []
                )
                ignored = [i for i in early_exercise if not 0 <= i < int(n)]
                if ignored:
                    st.warning(f"Steps {ignored} are outside 0 … {int(n) - 1} and are ignored.")

                pricer = BinomialTreeOptionPricer(
                    S, K, T, r, sigma, q, int(n),
                    option_type,
                    exercise_type,
                    early_exercise,
                )
                price, delta, gamma, theta = pricer.binomial_CRR_pricing_model()
                vega, rho = pricer.CRR_vega_and_rho()
                bs_reference = bs_pricer(*market)

                st.session_state["crr_result"] = {
                    "params": market,
                    "n": int(n),
                    "exercise": exercise_type,
                    "early": early_exercise,
                    "price": float(price),
                    "greeks": {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
                               "theta": float(theta), "rho": float(rho)},
                    "bs_price": float(bs_reference.BSM_option_pricing()),
                    "bs_greeks": {k: float(v) for k, v in bs_greeks(bs_reference).items()},
                    "tree": pricer.CRR_tree() if int(n) <= 5 else None,
                    "convergence": None,
                }
                if show_convergence:
                    steps = np.arange(1, 201)
                    prices = [
                        BinomialTreeOptionPricer(S, K, T, r, sigma, q, int(m), option_type,
                                                 exercise_type,
                                                 [round(i * m / int(n)) for i in early_exercise]
                                                 ).binomial_CRR_pricing_model()[0]
                        for m in steps
                    ]
                    st.session_state["crr_result"]["convergence"] = (steps, prices)
            except Exception as error:
                st.error(
                    "Your CRR class returned an error. "
                    f"Error: {error}"
                )

        if "crr_result" in st.session_state:
            result = st.session_state["crr_result"]
            s0, k0, t0, r0, v0, q0, kind = result["params"]
            stale_warning((result["params"], result["n"], result["exercise"]),
                          (market, int(n), exercise_type))

            c1, c2, c3 = st.columns(3)
            with c1:
                result_card("Option price", f"{result['price']:.4f}",
                            f"CRR tree · {result['exercise'].title()} · {result['n']} steps")
            with c2:
                if result["exercise"] == "european":
                    result_card("Black-Scholes reference", f"{result['bs_price']:.4f}",
                                f"Tree error {result['price'] - result['bs_price']:+.4f}")
                else:
                    result_card("Early-exercise premium",
                                f"{result['price'] - result['bs_price']:.4f}",
                                f"vs European BS price {result['bs_price']:.4f}")
            with c3:
                result_card("Intrinsic value", f"{intrinsic(s0, k0, kind):.4f}", kind.title())

            st.markdown("#### Greeks")
            greek_cards(result["greeks"], reference=result["bs_greeks"])
            if result["exercise"] != "european":
                st.caption("The BS values under each Greek are for the European option, "
                           "shown only as a reference point.")

            if result["tree"] is not None:
                steps = result["n"]
                dt = t0 / steps
                u = np.exp(v0 * np.sqrt(dt))
                d = 1 / u
                p = (np.exp((r0 - q0) * dt) - d) / (u - d)
                st.markdown("#### The tree, step by step")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    result_card("Up factor u", f"{u:.4f}", "u = e^(σ√Δt)")
                with c2:
                    result_card("Down factor d", f"{d:.4f}", "d = 1/u")
                with c3:
                    result_card("Risk-neutral p", f"{p:.4f}", "p = (e^((r−q)Δt) − d)/(u − d)")
                with c4:
                    result_card("Discount per step", f"{np.exp(-r0 * dt):.4f}", f"e^(−rΔt), Δt = {dt:.4f}")
                if not 0 < p < 1:
                    st.warning("p is outside (0, 1): the tree allows arbitrage for these inputs. "
                               "Use more steps or a higher volatility.")
                stock, value, exercised = result["tree"]
                plot_binomial_tree(stock, value, exercised, steps, result["early"],
                                   result["exercise"], dt)
                st.markdown(
                    '<div class="info-box">Read it from right to left. At maturity V is the payoff. '
                    'At every earlier node V = e<sup>−rΔt</sup>[p·V<sub>up</sub> + (1−p)·V<sub>down</sub>]; '
                    'for American and Bermudan options (on allowed steps, shaded) this is compared with the '
                    'immediate exercise value and the larger one is kept.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Tip: set the number of steps to 5 or fewer to draw the full tree.")

            if result["convergence"] is not None:
                steps, prices = result["convergence"]
                fig, ax = styled_figure()
                ax.plot(steps, prices, color=CYAN, linewidth=1.4, marker="o", markersize=2.5,
                        label=f"CRR ({result['exercise']})")
                ax.axhline(result["bs_price"], color=ORANGE, linewidth=1.6,
                           label="Black-Scholes (European)")
                ax.set_title("CRR price as the number of steps grows")
                ax.set_xlabel("Number of steps n")
                ax.set_ylabel("Option value")
                ax.legend()
                show_figure(fig)
                st.caption("The zig-zag between odd and even n comes from where the strike falls relative "
                           "to the terminal nodes. For American puts the tree converges to a value above the "
                           "Black-Scholes line: the gap is the early-exercise premium. Bermudan exercise "
                           "steps are rescaled proportionally for each n.")

    else:
        st.markdown(
            '<div class="info-box">European options priced by simulating the risk-neutral GBM with the '
            'Euler-Maruyama scheme. Greeks use common random numbers: every bumped price is simulated '
            'with the same seed, so the simulation noise largely cancels in the differences.</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            n_steps = st.number_input("SDE time steps", min_value=1, value=252, step=21)
            estimate_greeks = st.checkbox("Estimate Greeks", value=True,
                                          help="Runs about 8 extra simulations.")
        with c2:
            n_paths = st.number_input("Monte Carlo paths", min_value=100, value=10_000, step=1_000)
            delta_method = st.radio("Delta estimator", ["pathwise", "finite difference"],
                                    horizontal=True,
                                    help="Pathwise: unbiased, lower variance, valid for GBM. "
                                         "Finite difference: works for any SDE.")
        with c3:
            seed = st.number_input("Random seed", min_value=0, value=42, step=1,
                                   help="Same seed gives the same paths, so results are reproducible.")

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
                np.random.seed(int(seed))
                result = pricer.MCeuropean(K, int(n_paths))
                if result is False:
                    raise ValueError("The Monte Carlo method returned False.")
                price, standard_error = result

                greeks = None
                if estimate_greeks:
                    with st.spinner("Estimating Greeks..."):
                        greeks = pricer.MCgreeks(
                            K, sigma=sigma, q=q, num_paths=int(n_paths), seed=int(seed),
                            delta_method="pathwise" if delta_method == "pathwise" else "fd",
                        )
                        greeks = {k: float(v) for k, v in greeks.items()}

                # same seed again -> exactly the paths used for the price, for the charts
                np.random.seed(int(seed))
                terminal = sde.EulerTerminalValues(int(n_paths))
                bs_reference = bs_pricer(*market)

                st.session_state["mc_result"] = {
                    "params": market,
                    "n_paths": int(n_paths),
                    "price": float(price),
                    "se": float(standard_error),
                    "greeks": greeks,
                    "terminal": None if terminal is False else np.asarray(terminal),
                    "bs_price": float(bs_reference.BSM_option_pricing()),
                    "bs_greeks": {k: float(v) for k, v in bs_greeks(bs_reference).items()},
                }
            except Exception as error:
                st.error(
                    "Your Monte Carlo class returned an error. "
                    f"Error: {error}"
                )

        if "mc_result" in st.session_state:
            result = st.session_state["mc_result"]
            s0, k0, t0, r0, v0, q0, kind = result["params"]
            stale_warning((result["params"], result["n_paths"]), (market, int(n_paths)))
            price, standard_error = result["price"], result["se"]
            lower = price - 1.96 * standard_error
            upper = price + 1.96 * standard_error
            inside = lower <= result["bs_price"] <= upper

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                result_card("Estimated price", f"{price:.4f}", "Monte Carlo")
            with c2:
                result_card("Standard error", f"{standard_error:.4f}", f"{result['n_paths']:,} paths")
            with c3:
                result_card("95% confidence interval", f"{lower:.4f} – {upper:.4f}", "Normal approximation")
            with c4:
                result_card("Black-Scholes reference", f"{result['bs_price']:.4f}",
                            "Inside the 95% CI" if inside else "Outside the 95% CI")

            if result["greeks"] is not None:
                st.markdown("#### Greeks")
                greek_cards(result["greeks"], reference=result["bs_greeks"])

            terminal = result["terminal"]
            if terminal is not None:
                if kind == "call":
                    payoffs = np.maximum(terminal - k0, 0)
                else:
                    payoffs = np.maximum(k0 - terminal, 0)
                discounted = np.exp(-r0 * t0) * payoffs
                counts = np.arange(1, len(discounted) + 1)
                running_mean = np.cumsum(discounted) / counts
                running_sq = np.cumsum(discounted ** 2) / counts
                running_se = np.sqrt(np.maximum(running_sq - running_mean ** 2, 0) / counts)

                fig, (ax1, ax2) = styled_figure(1, 2, figsize=(12, 4.5))
                ax1.plot(counts, running_mean, color=CYAN, linewidth=1.6, label="Running MC estimate")
                ax1.fill_between(counts, running_mean - 1.96 * running_se,
                                 running_mean + 1.96 * running_se, color=BLUE, alpha=0.2,
                                 label="95% confidence band")
                ax1.axhline(result["bs_price"], color=ORANGE, linewidth=1.4, label="Black-Scholes")
                ax1.set_xscale("log")
                ax1.set_title("Convergence of the estimate")
                ax1.set_xlabel("Number of paths (log scale)")
                ax1.set_ylabel("Option value")
                ax1.legend()

                ax2.hist(terminal, bins=60, color=BLUE, alpha=0.75, edgecolor="#071A33")
                ax2.axvline(k0, color=ORANGE, linewidth=1.6, label="Strike K")
                ax2.axvline(s0, color=CYAN, linewidth=1.2, linestyle="--", label="Spot S₀")
                ax2.set_title("Simulated terminal prices S(T)")
                ax2.set_xlabel("S(T)")
                ax2.set_ylabel("Number of paths")
                ax2.legend()
                show_figure(fig)
                st.caption("The confidence band shrinks like 1/√(paths): four times more paths halves the "
                           "error. In the histogram, only the paths beyond the strike pay off.")

# ---------------------------------------------------------------------------
# Greeks explorer
# ---------------------------------------------------------------------------

with greeks_tab:
    st.subheader("How the Greeks change with the spot price")
    st.write(
        "Black-Scholes Greeks for the inputs in the sidebar, plotted across spot prices. "
        "Compare several maturities or volatilities to see how the shape of each sensitivity changes."
    )
    compare = st.radio("Compare curves across", ["Time to maturity", "Volatility"], horizontal=True)

    if compare == "Time to maturity":
        scenarios = [(T, sigma, f"T = {T:g}"), (T / 2, sigma, f"T = {T / 2:g}"),
                     (T / 8, sigma, f"T = {T / 8:g}")]
    else:
        scenarios = [(T, sigma / 2, f"σ = {sigma / 2:g}"), (T, sigma, f"σ = {sigma:g}"),
                     (T, sigma * 2, f"σ = {sigma * 2:g}")]

    spots = np.linspace(max(0.01, K * 0.5), K * 1.5, 120)
    panels = [("price", "Option value", "V")] + [
        (key, GREEK_INFO[key][0], GREEK_INFO[key][1]) for key in GREEK_ORDER
    ]
    colours = [MUTED, CYAN, ORANGE]

    fig, axes = styled_figure(2, 3, figsize=(13, 7.2))
    for (maturity, vol, label), colour in zip(scenarios, colours):
        curves = {key: [] for key, _, _ in panels}
        for x in spots:
            pricer = bs_pricer(x, K, maturity, r, vol, q, option_type)
            curves["price"].append(pricer.BSM_option_pricing())
            for key, value in bs_greeks(pricer).items():
                curves[key].append(value)
        for ax, (key, title, symbol) in zip(axes.ravel(), panels):
            ax.plot(spots, curves[key], color=colour, linewidth=2, label=label)

    for ax, (key, title, symbol) in zip(axes.ravel(), panels):
        ax.axvline(K, color="#28517A", linewidth=1, linestyle="--")
        ax.axvline(S, color="white", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{title} ({symbol})")
        ax.set_xlabel("Spot price")
    axes[0, 0].legend()
    show_figure(fig)
    st.caption("Dashed line: strike K. Thin white line: current spot S₀. Theta is per year; "
               "vega and rho are per 1.00 change (divide by 100 for a 1% change).")

    with st.expander("What to look for"):
        st.markdown(
            """
            - **Delta** is the slope of the price curve and the hedge ratio: how many shares offset
              one option. It moves from 0 to e^(−qT) for a call (−e^(−qT) to 0 for a put) and becomes a
              step function as maturity approaches.
            - **Gamma** is how fast delta changes. It peaks around the strike and the peak gets taller
              and narrower as maturity shrinks, which is why short-dated at-the-money options are the
              hardest to hedge.
            - **Vega** is largest at the money and grows with time to maturity: there is more time for
              volatility to matter.
            - **Theta** is usually negative (time decay) and most negative near the strike for
              short maturities. Deep in-the-money European puts can have positive theta.
            - **Rho** is positive for calls and negative for puts and grows with maturity and moneyness.
            - Gamma and theta mirror each other: in a delta-hedged position, gaining from large moves
              (positive gamma) is paid for by time decay (negative theta).
            """
        )

# ---------------------------------------------------------------------------
# SDE solver
# ---------------------------------------------------------------------------

with sde_tab:
    st.subheader("Stochastic differential-equation solver")
    st.write("Solve SDEs dy = f(x, y) dx + g(x, y) dW using the Euler-Maruyama, Tamed Euler and Milstein methods.")

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
                raise ValueError("The solver did not return a path.")

            times = np.linspace(0, horizon, int(intervals) + 1)
            st.session_state["sde_path"] = (times, np.asarray(path), method)
        except Exception as error:
            st.error(f"Your SDE solver returned an error: {error}")

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
        st.line_chart(chart_data, color="#55CCFF")

# ---------------------------------------------------------------------------
# Project notes
# ---------------------------------------------------------------------------

with notes_tab:
    st.subheader("About")
    st.markdown(
        """
        This toolkit prices European, American and Bermudan options with four approaches and puts
        their results side by side, so you can see where they agree, where they differ and why.

        | Method | Exercise styles | Greeks |
        |---|---|---|
        | Black-Scholes-Merton | European | Closed form |
        | CRR binomial tree | European, American, Bermudan | Delta, gamma, theta from the tree; vega, rho by re-pricing |
        | Monte Carlo (Euler-Maruyama GBM) | European | Pathwise / finite differences with common random numbers |
        | SDE solver | – | Euler-Maruyama, Milstein, Tamed Euler paths for any drift and diffusion |

        **Conventions.** Rates, dividend yield and volatility are annual and continuously compounded.
        Theta is per year (divide by 365 for one calendar day). Vega and rho are per 1.00 change
        (divide by 100 for a 1% change).

        **Author.** Developed by Tvrtko Šapina —
        [personal website](https://tvrtkosapina.github.io/webpage/) ·
        [GitHub](https://github.com/tvrtkosapina).

        Thank you for using this option pricing tool!
        """
    )
