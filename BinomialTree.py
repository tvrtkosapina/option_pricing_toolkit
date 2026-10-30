import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

class BinomialTreeOptionPricer:
    def __init__(self,S, K, T, r, sigma, q, n, option_type = "call", exercise_type = "european", early_exercise = None):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.n = int(n)
        self.option_type = option_type.lower()
        self.exercise_type = exercise_type.lower()
        if early_exercise is None:
            self.early_exercise = []
        else:
            self.early_exercise = early_exercise

    def binomial_CRR_pricing_model(self):
        ''' S- underlying asset price
            K- strike price
            T- time to expiration
            r- interest rate
            sigma- volatility of the asset
            q- dividend yield
            n- number of steps 
            type = "call" or "put"
            exercise_type = "european", "american" or "bermudan" 
            early_exercise: if exercise_type = "bermudan" it is the list of tree steps (0..n) at which early exercise is allowed

            Returns (price, delta, gamma, theta). Theta is per year.
            Delta needs n >= 1, gamma and theta need n >= 2 (otherwise they are NaN).'''

        dt = self.T/self.n
        u = np.exp(self.sigma * np.sqrt(dt)) # d = 1/u
        d = 1/u
        p = np.zeros(self.n + 1)
        p0 = (u * np.exp(-self.q * dt) - np.exp(-self.r * dt)) / (u**2 - 1)
        p1 = np.exp(-self.r * dt) - p0

        for i in range(self.n+1):
            if self.option_type == "call":
                p[i] = max(self.S * u**(2*i - self.n) - self.K, 0)
                continue
            p[i] = max(self.K - self.S * u**(2*i - self.n), 0)

        # values at steps 1 and 2 (needed for greeks); stay NaN if the tree is too short
        V1_down = V1_up = np.nan
        V2dd = V2ud = V2uu = np.nan
        if self.n == 1:
            V1_down, V1_up = p[0], p[1]
        if self.n == 2:
            V2dd, V2ud, V2uu = p[0], p[1], p[2]

        for i in range(self.n-1, -1, -1):
            for j in range(0, i+1):
                p[j] = p0 * p[j+1] + p1 * p[j]

                if self.option_type == "put":
                    exercise = self.K - self.S * u**(2*j - i)
                if self.option_type == "call":
                    exercise = self.S * u**(2*j - i) - self.K

                if self.exercise_type == "american":
                    p[j] = max(exercise, p[j])

                if self.exercise_type == "bermudan":
                    if i in self.early_exercise:
                        p[j] = max(exercise, p[j])

            if i == 1:
                V1_down = p[0]
                V1_up = p[1]
                
            if i == 2:
                V2dd = p[0]
                V2ud = p[1]
                V2uu = p[2]

        S = self.S
        delta = (V1_up - V1_down)/(S*u - S*d)
        
        delta_up = (V2uu- V2ud)/(S*u*u - S)
        delta_down = (V2ud - V2dd)/(S - S*d*d)

        gamma = (delta_up - delta_down)/(0.5*(S*u*u - S*d*d))
        theta = (V2ud - p[0])/(2*dt)

        return (p[0], delta, gamma, theta)

    def CRR_vega_and_rho(self, vol_bump = 0.01, rate_bump = 0.0001):
        ''' Vega and rho by bump-and-reprice with central differences.
            Both trees use the same number of steps n, so the discretisation error mostly cancels.
            Returned per unit change (1.00 = 100 vol points / 100% rate), same convention as BlackScholes.py.
            Divide by 100 for "per 1% change". '''
        def price(sigma, r):
            option = BinomialTreeOptionPricer(self.S, self.K, self.T, r, sigma, self.q, self.n,
                                              self.option_type, self.exercise_type, self.early_exercise)
            return option.binomial_CRR_pricing_model()[0]

        vol_bump = min(vol_bump, 0.5 * self.sigma)   # keep sigma - bump > 0
        vega = (price(self.sigma + vol_bump, self.r) - price(self.sigma - vol_bump, self.r)) / (2 * vol_bump)
        rho = (price(self.sigma, self.r + rate_bump) - price(self.sigma, self.r - rate_bump)) / (2 * rate_bump)
        return vega, rho

    def CRR_tree(self):
        ''' Full tree for visualisation / teaching (use only for small n).
            Returns (stock, value, exercised), each a list of lists: element [i][j] is the node at step i
            with j up-moves. exercised[i][j] is True when early exercise is optimal at that node.
            Same algorithm as binomial_CRR_pricing_model(), but keeps every node instead of overwriting. '''
        dt = self.T/self.n
        u = np.exp(self.sigma * np.sqrt(dt))
        p0 = (u * np.exp(-self.q * dt) - np.exp(-self.r * dt)) / (u**2 - 1)
        p1 = np.exp(-self.r * dt) - p0
        sign = 1 if self.option_type == "call" else -1

        stock = [[self.S * u**(2*j - i) for j in range(i+1)] for i in range(self.n+1)]
        value = [[0.0]*(i+1) for i in range(self.n+1)]
        exercised = [[False]*(i+1) for i in range(self.n+1)]

        value[self.n] = [max(sign*(s - self.K), 0) for s in stock[self.n]]
        for i in range(self.n-1, -1, -1):
            can_exercise = (self.exercise_type == "american" or
                            (self.exercise_type == "bermudan" and i in self.early_exercise))
            for j in range(i+1):
                continuation = p0 * value[i+1][j+1] + p1 * value[i+1][j]
                exercise = sign*(stock[i][j] - self.K)
                if can_exercise and exercise > continuation:
                    value[i][j] = exercise
                    exercised[i][j] = True
                else:
                    value[i][j] = continuation
        return stock, value, exercised
