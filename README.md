# Option Pricing Toolkit

An interactive Streamlit app and a small Python library for pricing equity options with four
complementary approaches, analytic Black-Scholes-Merton, the Cox-Ross-Rubinstein binomial tree,
Monte Carlo simulation and numerical SDE solvers, together with the Greeks for each.

The goal is educational as much as practical: every method is shown next to the Black-Scholes
benchmark, so you can see where the methods agree, where they differ and why.

## Features

**Black-Scholes-Merton**
- European calls and puts with a continuous dividend yield
- Closed-form delta, gamma, vega, theta and rho
- Price-versus-spot chart with the payoff at maturity

**CRR binomial tree**
- European, American and Bermudan exercise (Bermudan: choose the tree steps where exercise is allowed)
- Delta, gamma and theta read directly from the first two levels of the tree; vega and rho by
  bump-and-reprice with central differences
- Early-exercise premium against the European Black-Scholes price
- Full tree diagram for n ≤ 5: stock price and option value at every node, with the nodes where
  early exercise is optimal highlighted, plus u, d, p and the per-step discount factor
- Convergence chart of the CRR price for n = 1…200 against Black-Scholes

**Monte Carlo**
- European options under risk-neutral geometric Brownian motion, simulated with Euler-Maruyama
- Standard error and 95% confidence interval, checked against the Black-Scholes price
- All five Greeks using common random numbers, with a choice of pathwise or finite-difference delta
- Reproducible results via a random seed
- Running-estimate convergence chart with confidence band, and a histogram of terminal prices

**Greeks explorer**
- All Black-Scholes Greeks plotted across spot prices, compared across maturities or volatilities,
  with notes on what each shape means

**SDE solver**
- Solve any scalar SDE `dy = f(x, y) dx + g(x, y) dW` typed as text (parsed with SymPy)
- Euler-Maruyama, Milstein and Tamed Euler schemes

## Project structure

```
├── app.py              # Streamlit interface
├── BlackScholes.py     # BlackScholesMertonOptionPricer: price + analytic Greeks
├── BinomialTree.py     # BinomialTreeOptionPricer: CRR tree, Greeks, full tree for plotting
├── MonteCarlo.py       # MonteCarloOptionPricer: MC price, standard error, Greeks
├── SDE_solvers.py      # StohasticOrdinaryDiffEquation, set_GBM: SDE schemes
├── requirements.txt
└── README.md
```

> The SDE module must be named `SDE_solvers.py` (with an underscore). A file called
> `SDE solvers.py` cannot be imported.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/tvrtkosapina/<repository-name>.git
cd <repository-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Set the market inputs (spot, strike, maturity, rate,
volatility, dividend yield, call or put) in the sidebar; they are shared by all tabs.

A few things worth trying:
- **See early exercise happen.** CRR tree, 5 steps, American put, strike 110. The orange nodes show
  where exercising beats holding.
- **Watch the tree converge.** Tick “Show convergence to Black-Scholes”. European prices oscillate
  around the Black-Scholes line; American puts settle above it.
- **Watch Monte Carlo error shrink.** Run with 1,000, then 16,000 paths: the standard error drops by
  roughly a factor of four.
- **See why short-dated options are hard to hedge.** Greeks explorer, compare across maturities:
  the gamma peak becomes tall and narrow as maturity shrinks.

## Using the library directly

```python
from BlackScholes import BlackScholesMertonOptionPricer
from BinomialTree import BinomialTreeOptionPricer
from MonteCarlo import MonteCarloOptionPricer
from SDE_solvers import set_GBM

S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02

# Black-Scholes
bs = BlackScholesMertonOptionPricer(S, K, T, r, sigma, q, "put")
print(bs.BSM_option_pricing(), bs.BSMdelta(), bs.BSMvega())

# CRR tree, American put
tree = BinomialTreeOptionPricer(S, K, T, r, sigma, q, 500, "put", "american")
price, delta, gamma, theta = tree.binomial_CRR_pricing_model()
vega, rho = tree.CRR_vega_and_rho()

# Bermudan: exercise allowed only at tree steps 125, 250 and 375
bermudan = BinomialTreeOptionPricer(S, K, T, r, sigma, q, 500, "put", "bermudan", [125, 250, 375])
print(bermudan.binomial_CRR_pricing_model()[0])

# Full tree for a small n (lists indexed [step][number of up-moves])
stock, value, exercised = BinomialTreeOptionPricer(S, K, T, r, sigma, q, 3, "put", "american").CRR_tree()

# Monte Carlo under the risk-neutral GBM dS = (r - q) S dt + sigma S dW
sde = set_GBM(str(r - q), str(sigma), x0=0, y0=S, xn=T, num_of_intervals=252)
mc = MonteCarloOptionPricer(sde, r, "put")
price, standard_error = mc.MCeuropean(K, num_paths=50_000)
greeks = mc.MCgreeks(K, sigma=sigma, q=q, num_paths=50_000, seed=42)
```

## Methods

### Black-Scholes-Merton

With continuous dividend yield $q$:

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$$

$$d_1 = \frac{\ln(S_0/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

### CRR binomial tree

With $\Delta t = T/n$, $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ and risk-neutral probability
$p = \dfrac{e^{(r-q)\Delta t} - d}{u - d}$, the value at each node is

$$V_{i,j} = e^{-r\Delta t}\left[p\,V_{i+1,j+1} + (1-p)\,V_{i+1,j}\right],$$

and for American options (and Bermudan options on allowed steps) $V_{i,j}$ is replaced by the
immediate exercise value when that is larger. Delta and gamma are finite differences of the option
values at steps 1 and 2; theta compares the middle node at step 2 (same stock price as today,
$2\Delta t$ later) with the root. The tree converges to Black-Scholes at rate $O(1/n)$, with the
characteristic odd/even oscillation.

### Monte Carlo

Terminal prices are simulated with the Euler-Maruyama scheme for the risk-neutral GBM and the price
is the discounted average payoff. The standard error is $e^{-rT} s / \sqrt{N}$ where $s$ is the
sample standard deviation of the payoffs, so the error falls like $1/\sqrt{N}$.

Greeks:

| Greek | Estimator |
|---|---|
| Delta | Pathwise $e^{-rT}\,\mathbb{E}\!\left[\mathbf 1_{\{S_T>K\}}\,S_T/S_0\right]$ (call), or central finite difference in $S_0$ |
| Gamma | Central second difference in $S_0$ |
| Theta | Central difference in maturity, $\Theta = -\partial V/\partial T$ |
| Vega | Central difference in $\sigma$ |
| Rho | Central difference in $r$, bumping both the drift $(r-q)$ and the discount factor |

All finite differences use **common random numbers**: each bumped price is simulated from the same
seed, so base and bumped paths share their Brownian increments and most of the simulation noise
cancels. With independent draws the error of a finite difference grows like $1/h$, which makes it
useless for small bumps. The pathwise delta is unbiased and has lower variance, but relies on $S_T$
being proportional to $S_0$ along each path, which holds for GBM. Delta, gamma and theta work for any
SDE; vega and rho assume the GBM model because a general SDE has no σ or r parameter.

### SDE schemes

For $dy = f(x,y)\,dx + g(x,y)\,dW$ with step $\Delta x$ and $\Delta W \sim N(0, \Delta x)$:

| Scheme | Update | Strong order |
|---|---|---|
| Euler-Maruyama | $y_{k+1} = y_k + f\,\Delta x + g\,\Delta W$ | 0.5 |
| Milstein | adds $\tfrac12\, g\, \partial_y g\,(\Delta W^2 - \Delta x)$ | 1.0 |
| Tamed Euler | drift replaced by $f / (1 + \sqrt{\Delta x}\,\lvert f\rvert)$ | 0.5 |

Tamed Euler keeps the moments bounded when the drift grows faster than linearly, where plain Euler
can diverge.

## Conventions

- Rates, dividend yield and volatility are annual and continuously compounded; maturity is in years.
- Theta is per year (divide by 365 for one calendar day).
- Vega and rho are per 1.00 change in σ or r (divide by 100 for a 1% change).

## Validation

Reference inputs: $S_0 = K = 100$, $T = 1$, $r = 5\%$, $\sigma = 20\%$, $q = 2\%$.
CRR with $n = 500$; Monte Carlo with 50,000 paths, 252 steps.

| Call | Price | Delta | Gamma | Vega | Theta | Rho |
|---|---|---|---|---|---|---|
| Black-Scholes | 9.2270 | 0.5869 | 0.0190 | 37.901 | −5.089 | 49.458 |
| CRR tree | 9.2231 | 0.5868 | 0.0190 | 37.881 | −5.096 | 49.457 |
| Monte Carlo | 9.246 ± 0.062 | 0.5881 | 0.0183 | 37.789 | −5.081 | 49.549 |

For the put, CRR gives 6.6588 (American) and 6.5649 (Bermudan, exercise at steps 125, 250, 375)
against 6.3301 for the European Black-Scholes price. As expected, the Bermudan value lies between
the European and the American value.

## Limitations

- Monte Carlo prices European payoffs only (no American or path-dependent options yet).
- The binomial tree is a pure-Python double loop, O(n²); n in the low thousands is the practical limit.
- `EulerTerminalValues` evaluates the drift and diffusion on NumPy arrays through SymPy's
  `lambdify(..., "math")`. This works for polynomial expressions such as GBM, but expressions that
  use functions like `exp` or `sin` fail on arrays.
- Errors in the SDE module are printed and signalled by returning `False`, so the app can only show a
  general error message.

## Author

**Tvrtko Šapina** · [Personal website](https://tvrtkosapina.github.io/webpage/) ·
[GitHub](https://github.com/tvrtkosapina)
