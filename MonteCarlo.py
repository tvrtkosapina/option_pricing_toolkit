import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from SDE_solvers import set_GBM

class MonteCarloOptionPricer:
    def __init__(self, sde, risk_free_rate, option_type = "call", exercise_type = "european"):
        self.sde = sde
        self.r = risk_free_rate
        self.option_type = option_type.lower()
        self.exercise_type = exercise_type.lower()   # only "european" is implemented

    def MCeuropean(self, K, num_paths=10000):
        T = self.sde.xn - self.sde.x0
        ST = self.sde.EulerTerminalValues(num_paths)

        if ST is False:
            return False

        if self.option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        discount = np.exp(-self.r * T)
        price = discount * np.mean(payoffs)

        standard_error = (discount* np.std(payoffs, ddof=1)/ np.sqrt(num_paths))

        return price, standard_error

    # ------------------------------------------------------------------
    # Greeks
    #
    # All finite-difference greeks use COMMON RANDOM NUMBERS: every bumped
    # price is simulated with the same seed, so the same Brownian increments
    # drive the base and the bumped paths. The noise then mostly cancels in
    # the difference, which is what makes MC finite differences usable at all
    # (independent draws would make the error ~ 1/h larger).
    # The global NumPy random state is saved and restored, so calling these
    # methods does not disturb other random draws in your program.
    # ------------------------------------------------------------------

    def _price(self, K, num_paths, seed, sde=None, r=None): # MC price with a fixed seed; sde and r can be replaced by bumped versions. 
        sde = self.sde if sde is None else sde
        r = self.r if r is None else r
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            result = MonteCarloOptionPricer(sde, r, self.option_type).MCeuropean(K, num_paths)
        finally:
            np.random.set_state(state)
        if result is False:
            raise ValueError("Monte Carlo simulation failed (EulerTerminalValues returned False).")
        return result[0]

    def _bumped_sde(self, y0=None, xn=None):
        ''' Copy of the SDE with a different initial value and/or end time (works for any SDE). '''
        new_sde = copy.copy(self.sde)
        if y0 is not None:
            new_sde.y0 = y0
        if xn is not None:
            new_sde.xn = xn
        return new_sde

    def _gbm(self, mu, sigma):
        ''' GBM with the same grid and initial value as self.sde, but new drift/volatility. '''
        return set_GBM(repr(float(mu)), repr(float(sigma)), x0=self.sde.x0, y0=self.sde.y0,
                       xn=self.sde.xn, num_of_intervals=self.sde.num_of_intervals)

    @staticmethod
    def _new_seed(seed):
        return int(np.random.randint(0, 2**31 - 1)) if seed is None else int(seed)

    def MCdelta(self, K, num_paths=10000, rel_bump=0.01, seed=None, method="fd"):
        ''' Delta = dV/dS0.
            method="fd":       central finite difference with common random numbers,
                               bump h = rel_bump * S0 (relative, so it scales with the spot). Works for any SDE.
            method="pathwise": pathwise estimator e^{-rT} E[ 1{S_T > K} * S_T / S0 ] (call).
                               Unbiased and lower variance, but only valid when S_T is proportional
                               to S0 along each path (true for GBM, also under the Euler scheme). '''
        seed = self._new_seed(seed)
        S0 = float(self.sde.y0)

        if method == "pathwise":
            T = self.sde.xn - self.sde.x0
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                ST = self.sde.EulerTerminalValues(num_paths)
            finally:
                np.random.set_state(state)
            if ST is False:
                raise ValueError("Monte Carlo simulation failed.")
            if self.option_type == "call":
                dpayoff = (ST > K).astype(float) * ST / S0
            else:
                dpayoff = -(ST < K).astype(float) * ST / S0
            return np.exp(-self.r * T) * np.mean(dpayoff)

        h = rel_bump * S0
        up = self._price(K, num_paths, seed, sde=self._bumped_sde(y0=S0 + h))
        down = self._price(K, num_paths, seed, sde=self._bumped_sde(y0=S0 - h))
        return (up - down) / (2 * h)

    def MCgamma(self, K, num_paths=10000, rel_bump=0.01, seed=None):
        ''' Gamma = d2V/dS0^2, central second difference with common random numbers.
            Gamma is the noisiest MC greek (the payoff has a kink), so use many paths. '''
        seed = self._new_seed(seed)
        S0 = float(self.sde.y0)
        h = rel_bump * S0
        up = self._price(K, num_paths, seed, sde=self._bumped_sde(y0=S0 + h))
        mid = self._price(K, num_paths, seed)
        down = self._price(K, num_paths, seed, sde=self._bumped_sde(y0=S0 - h))
        return (up - 2 * mid + down) / (h * h)

    def MCtheta(self, K, num_paths=10000, dt_bump=1/365, seed=None):
        ''' Theta = dV/dt = -dV/dT (per year), by changing the maturity xn.
            Central difference, or forward difference if T is shorter than the bump. '''
        seed = self._new_seed(seed)
        T = self.sde.xn - self.sde.x0
        longer = self._price(K, num_paths, seed, sde=self._bumped_sde(xn=self.sde.xn + dt_bump))
        if T - dt_bump > 0:
            shorter = self._price(K, num_paths, seed, sde=self._bumped_sde(xn=self.sde.xn - dt_bump))
            return -(longer - shorter) / (2 * dt_bump)
        base = self._price(K, num_paths, seed)
        return -(longer - base) / dt_bump

    def MCvega(self, K, sigma, q=0.0, num_paths=10000, vol_bump=0.01, seed=None):
        ''' Vega = dV/dsigma (per 1.00 of volatility).
            A general SDE has no "sigma" parameter, so this assumes self.sde is the risk-neutral GBM
            dS = (r - q) S dt + sigma S dW and rebuilds it with sigma +/- vol_bump. '''
        seed = self._new_seed(seed)
        vol_bump = min(vol_bump, 0.5 * sigma)
        up = self._price(K, num_paths, seed, sde=self._gbm(self.r - q, sigma + vol_bump))
        down = self._price(K, num_paths, seed, sde=self._gbm(self.r - q, sigma - vol_bump))
        return (up - down) / (2 * vol_bump)

    def MCrho(self, K, sigma, q=0.0, num_paths=10000, rate_bump=0.0001, seed=None):
        ''' Rho = dV/dr (per 1.00 of rate). Assumes the risk-neutral GBM, like MCvega.
            Under the risk-neutral measure r enters twice: in the drift (r - q) AND in the discount factor,
            so both are bumped together. '''
        seed = self._new_seed(seed)
        up = self._price(K, num_paths, seed, sde=self._gbm(self.r + rate_bump - q, sigma), r=self.r + rate_bump)
        down = self._price(K, num_paths, seed, sde=self._gbm(self.r - rate_bump - q, sigma), r=self.r - rate_bump)
        return (up - down) / (2 * rate_bump)

    def MCgreeks(self, K, sigma=None, q=0.0, num_paths=10000, seed=None, rel_bump=0.01, delta_method="pathwise"):
        ''' All greeks with one shared seed (so they are consistent with each other).
            Delta/gamma/theta work for any SDE; vega and rho need sigma (GBM assumption) and are
            omitted when sigma is None. Returns a dict. '''
        seed = self._new_seed(seed)
        greeks = {
            "delta": self.MCdelta(K, num_paths, rel_bump, seed, method=delta_method),
            "gamma": self.MCgamma(K, num_paths, rel_bump, seed),
            "theta": self.MCtheta(K, num_paths, seed=seed),
        }
        if sigma is not None:
            greeks["vega"] = self.MCvega(K, sigma, q, num_paths, seed=seed)
            greeks["rho"] = self.MCrho(K, sigma, q, num_paths, seed=seed)
        return greeks
