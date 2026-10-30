import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

class BlackScholesMertonOptionPricer():
    def __init__(self,S0, K, T, r, sigma, q, option_type = "call"):
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.option_type = option_type.lower()

    def BSM_option_pricing(self):
        d1 = (np.log(self.S0/self.K)+ (self.r - self.q + 0.5*self.sigma*self.sigma)*self.T)/(self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call": 
            return self.S0 * np.exp(-self.q*self.T) * norm.cdf(d1) - self.K * np.exp(-self.r*self.T) * norm.cdf(d2)
    
        return self.K * np.exp(-self.r*self.T) * norm.cdf(-d2) - self.S0 * np.exp(-self.q*self.T) * norm.cdf(-d1)

    def BSMdelta(self):
        d1 = (np.log(self.S0/self.K)+ (self.r - self.q + 0.5*self.sigma*self.sigma)*self.T)/(self.sigma * np.sqrt(self.T))
        if self.option_type == "call": 
            return np.exp(-self.q*self.T) * norm.cdf(d1)
    
        return -np.exp(-self.q*self.T) * norm.cdf(-d1)
    
    def BSMgamma(self):
        d1 = (np.log(self.S0/self.K)+ (self.r - self.q + 0.5*self.sigma*self.sigma)*self.T)/(self.sigma * np.sqrt(self.T))
        return np.exp(-self.q*self.T) * norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def BSMvega(self):
        d1 = (np.log(self.S0/self.K)+ (self.r - self.q + 0.5*self.sigma*self.sigma)*self.T)/(self.sigma * np.sqrt(self.T))
        return self.S0 * np.exp(-self.q*self.T) * norm.pdf(d1) * np.sqrt(self.T)

    def BSMtheta(self):
        d1 = (np.log(self.S0/self.K)+ (self.r - self.q + 0.5*self.sigma*self.sigma)*self.T)/(self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            return -(self.S0 * np.exp(-self.q*self.T) * norm.pdf(d1) * self.sigma)/(2*np.sqrt(self.T)) + self.q * self.S0 * np.exp(-self.q*self.T) * norm.cdf(d1) - self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(d2)

        return -(self.S0 * np.exp(-self.q*self.T) * norm.pdf(d1) * self.sigma)/(2*np.sqrt(self.T)) - self.q * self.S0 * np.exp(-self.q*self.T) * norm.cdf(-d1) + self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(-d2)

    def BSMrho(self):
        d1 = (np.log(self.S0/self.K)+ (self.r - self.q + 0.5*self.sigma*self.sigma)*self.T)/(self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            return self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(d2)

        return -self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(-d2)

        






    
