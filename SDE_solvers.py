import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from sympy import symbols, lambdify, parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
from sympy.utilities.lambdify import lambdify


class StohasticOrdinaryDiffEquation:
    def __init__(self, drift, diffusion, x0=0, y0=0, xn=0, num_of_intervals=1):
        self.x0 = x0
        self.y0 = y0
        self.xn = xn
        self.num_of_intervals = num_of_intervals

        x, y = symbols('x y')
        transformations = (standard_transformations + (implicit_multiplication_application,))

        # parse only once
        drift_expr = parse_expr(drift, transformations=transformations)
        diffusion_expr = parse_expr(diffusion, transformations=transformations)

        # numeric versions
        self.f = lambdify((x, y), drift_expr, "math")
        self.g = lambdify((x, y), diffusion_expr, "math")

        # store for Milstein
        dg_expr = diffusion_expr.diff(y)
        self.dg = lambdify((x, y), dg_expr, "math")


    
    def EulerMethod(self):
        N = self.num_of_intervals
        if(N<=0):
            print("Error: Number of intervals must be greater than 0.")
            return False
        y = np.array(self.y0, dtype=float)
        x = self.x0
        dx = (self.xn - x) / N
        if(dx==0):
            print("Error: dx = 0. Either the xn = x0 or the difference is too small to calculate")
            return False
        sqrt_dx = np.sqrt(abs(dx))

        f = self.f
        g = self.g
        try: # simple test call to check if the functions are properly in the function
            _ = self.f(x, y)
            _ = self.g(x, y)

        except ValueError:
            print("Error: ValeuError. Drift or diffusion expression for initial is equal to invalid value/s.")
            return False
        except ZeroDivisionError:
            print("Error: Drift/diffusion has division by zero at initial point.")
            return False
        except OverflowError:
            print("Error: Drift/diffusion evaluates to infinity at initial point.")
            return False
        except Exception:
            print("Error: Exception. Drift or diffusion expression is invalid.")
            return False
        
        Rj = np.zeros(N+1)
        Rj[0] = y

        Z = np.random.normal(0, 1, N)

        for i in range(N):
            dW = sqrt_dx * Z[i]
            
            try:
                y += dx * f(x, y) + g(x, y) * dW # drift = f(x, y), diffusion = g(x, y)
            except ZeroDivisionError:
                print("Error: ZeroDivisionError at x =", x, " y =", y,". Try again with different initial conditions or different equation.")
                return False
            except OverflowError: # it might occur if the ODEs have increasingly very high numbers and so the Euler method starts diverging
                print("Error: Overflow at x =", x, " y =", y,". Try again with more subintervals or switch to TamedEulerMethod().")
                return False
            except ValueError:
                print("Error: Overflow at x =", x, " y =", y,". Try again with different initial conditions or different equation.")
                return False
            except Exception as e:
                print(f"Unexpected error at step {i}: {e}")
                return False
            
            x += dx

            Rj[i+1] = y     

        return Rj
    
    def EulerTerminalValues(self, num_paths):

        N = self.num_of_intervals

        if N <= 0:
            print("Error: Number of intervals must be greater than 0.")
            return False

        if num_paths <= 0:
            print("Error: Number of paths must be greater than 0.")
            return False

        dx = (self.xn - self.x0) / N

        if dx <= 0:
            print("Error: xn must be greater than x0.")
            return False

        sqrt_dx = np.sqrt(dx)

        y = np.full(num_paths, self.y0)

        x = self.x0

        for i in range(N):

            Z = np.random.normal(0, 1, num_paths)
            dW = sqrt_dx * Z

            try:
                y = (
                    y
                    + dx * self.f(x, y)
                    + self.g(x, y) * dW
                )

            except ZeroDivisionError:
                print(f"Error: Division by zero at step {i}.")
                return False

            except OverflowError:
                print(f"Error: Overflow at step {i}.")
                return False

            except ValueError:
                print(f"Error: Invalid value at step {i}.")
                return False

            except Exception as e:
                print(f"Unexpected error at step {i}: {e}")
                return False

            x += dx

        return y

    def MilsteinMethod(self):
        N = self.num_of_intervals
        if(N<=0):
            print("Error: Number of intervals must be greater than 0.")
            return False
        y = self.y0
        x = self.x0
        dx = (self.xn - x) / N
        if(dx==0):
            print("Error: dx = 0. Either the xn = x0 or the difference is too small to calculate")
            return False
        sqrt_dx = np.sqrt(abs(dx))

        f = self.f
        g = self.g
        dg = self.dg  # derivative wrt y
        try: # simple test call to check if the functions are properly in the function
            _ = self.f(x, y)
            _ = self.g(x, y)
            _ = self.dg(x,y)
        except ValueError:
            print("Error: ValeuError. Drift or diffusion expression for initial is equal to invalid value/s.")
            return False
        except ZeroDivisionError:
            print("Error: Drift/diffusion has division by zero at initial point.")
            return False
        except OverflowError:
            print("Error: Drift/diffusion evaluates to infinity at initial point.")
            return False
        except Exception:
            print("Error: Exception. Drift or diffusion expression is invalid.")
            return False

        Rj = np.zeros(N+1)
        Rj[0] = y

        Z = np.random.normal(0, 1, N)

        for i in range(N):
            dW = sqrt_dx * Z[i]
            
            try:
                y += dx * f(x, y) + g(x, y) * dW + 0.5 * g(x, y) * dg(x, y) * (dW*dW - dx)
                
            except ZeroDivisionError:
                print("Error: ZeroDivisionError at x =", x, " y =", y,". Try again with different initial conditions or different equation.")
                return False
            except OverflowError: # it might occur if the ODEs have increasingly very high numbers and so the Euler method starts diverging
                print("Error: Overflow at x =", x, " y =", y,". Try again with more subintervals or switch to TamedEulerMethod().")
                return False
            except ValueError:
                print("Error: Overflow at x =", x, " y =", y,". Try again with different initial conditions or different equation.")
                return False
            except Exception as e:
                print(f"Unexpected error at step {i}: {e}")
                return False

            x += dx
            Rj[i+1] = y 

        return Rj

    def TamedEulerMethod(self): # when the drift grows super-linearly, it can be shown that the moments of Euler approximation could diverge to infinity ## see Literature [ZiJuLeLiYuWa]
        N = self.num_of_intervals
        if(N<=0):
            print("Error: Number of intervals must be greater than 0.")
            return False
        y = self.y0
        x = self.x0
        dx = (self.xn - x) / N
        if(dx==0):
            print("Error: dx = 0. Either the xn = x0 or the difference is too small to calculate")
            return False
        sqrt_dx = np.sqrt(abs(dx))

        f = self.f
        g = self.g
        try: # simple test call to check if the functions are properly in the function
            _ = self.f(x, y)
            _ = self.g(x, y)
        except ValueError:
            print("Error: ValeuError. Drift or diffusion expression for initial is equal to invalid value/s.")
            return False
        except ZeroDivisionError:
            print("Error: Drift/diffusion has division by zero at initial point.")
            return False
        except OverflowError:
            print("Error: Drift/diffusion evaluates to infinity at initial point.")
            return False
        except Exception:
            print("Error: Exception. Drift or diffusion expression is invalid.")
            return False

        Rj = np.zeros(N+1)
        Rj[0] = y

        Z = np.random.normal(0, 1, N)

        for i in range(N):
            dW = sqrt_dx * Z[i]
            
            try:
                y += (f(x, y) / (1 + sqrt_dx * abs(f(x, y)))) * dx + g(x, y) * dW # drift = f(x, y), diffusion = g(x, y) ### alpha = 1/2
   
            except ZeroDivisionError:
                print("Error: ZeroDivisionError at x =", x, " y =", y,". Try again with different initial conditions or different equation.")
                return False
            except OverflowError: # it might occur if the ODEs have increasingly very high numbers and so the Euler method starts diverging
                print("Error: Overflow at x =", x, " y =", y,". Try again with more subintervals or switch to TamedEulerMethod().")
                return False
            except ValueError:
                print("Error: Overflow at x =", x, " y =", y,". Try again with different initial conditions or different equation.")
                return False
            except Exception as e:
                print(f"Unexpected error at step {i}: {e}")
                return False
            
            x += dx

            Rj[i+1] = y     

        return Rj

        
    def plot(self, path):  # path should be solutions of SDEs
        if path is None or path is False: # Check correctly if Rj exists and contains any values
            print("No values for plotting. Run EulerMethod() to get the path first.")
            return
    
        Rjx = np.linspace(self.x0,self.xn,self.num_of_intervals + 1)

        plt.plot(Rjx, path)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Graph")
        plt.grid(True)
        plt.show()


def is_linear_in_y(expr):
    x, y = symbols('x y')
    return expr.expand().coeff(y) != 0 and expr.expand().coeff(y, 2) == 0

def validate_GBM(mu_str, sigma_str):
    x, y = symbols('x y')

    try:
        mu = parse_expr(mu_str)
        sigma = parse_expr(sigma_str)
    except:
        return False, "mu ili sigma nisu valjani matematički izrazi."

        # očekujemo da je mu neovisno o y (ali smije biti funkcija x!)
    if y in mu.free_symbols:
        return False, "mu u GBM-u ne smije ovisiti o y (može o x)."

        # isto za sigma
    if y in sigma.free_symbols:
        return False, "sigma u GBM-u ne smije ovisiti o y."

        # drift i diffusion moraju biti proporcionalni y
        # dakle mu*y i sigma*y
        
    drift = mu * y
    diffusion = sigma * y

        # korisnik je možda dao validne izraze, ali ti koristiš standardiziranu formu
    return True, ""

def set_GBM(mu, sigma, x0=0, y0=0, xn=0, num_of_intervals=1):
    """
    Sets StohasticOrdinaryDiffEquation() object for Geometric Brownian Motion:
    dS = mu*S dt + sigma*S dW
    """
    ok, msg = validate_GBM(mu, sigma)
    if not ok:
        raise ValueError(f"GBM parametri nisu kompatibilni: {msg}")
    drift_expr = '(' + mu + ')'+'* y'
    diffusion_expr = f"({sigma}) * y"
    return StohasticOrdinaryDiffEquation(drift_expr, diffusion_expr, x0, y0, xn, num_of_intervals)




    
