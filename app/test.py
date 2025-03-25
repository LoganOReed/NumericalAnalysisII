import numpy as np
import scipy as sp
from oct2py import octave
import matplotlib.pyplot as plt

def solveIVP(f, u0, tspan, h, solver):
    """
    sets up iteration which uses solver to solve each step.
    """
    t = np.arange(tspan[0], tspan[1] + h, h)
    u = np.zeros((len(t),len(u0)))
    t[0] = tspan[0]
    u[0,:] = u0

    if (solver == ab or solver == am):
        u[1,:] = uTrue(None, h)
        for n in range(1, len(t) - 1):
            u[n+1,:] = solver(f, u[n,:], t[n], h, u[n-1,:])
    else:
        for n in range(len(t) - 1):
            u[n+1,:] = solver(f, u[n,:], t[n], h)
    return u, t

def fe(f, u, t, h):
    """Forward Euler."""
    return u + h * f(u, t)

# back euler
def be(f, u, t, h):
    """Backwards Euler."""
    g = lambda x: x - u - h*f(x,t+h)
    res = sp.optimize.root(g, u, tol=1e-9)
    if not res.success:
        print("be failed")
        return u
    else:
        return res.x


def rk4(f, u, t, h):
    k1 = f(u, t)
    k2 = f(u + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(u + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(u + h * k3, t + h)
    return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# crank nicolson
def cn(f, u, t, h):
    """Crank Nicolson."""
    g = lambda x: x - u - 0.5*h*(f(u,t) + f(x,t+h))
    res = sp.optimize.root(g, u, tol=1e-9)
    if not res.success:
        print("cn failed")
        return u
    else:
        return res.x

# Adams-Bashforth (with n-1 true solutions)
def ab(f, u, t, h, uprev):
    """Adams-Bashforth 2 (with k-1 true solutions)."""
    return u + 0.5* h * ( 3*f(u,t) - f(uprev, t - h))

# Adams-Moulton (with n-1 true solutions)
def am(f, u, t, h, uprev):
    """Adams-Moulton 2 (with k-1 true solutions)."""
    g = lambda x: x - u - (1.0/12.0)*h*(8*f(u,t) + 5*f(x,t+h) - f(uprev,t-h))
    res = sp.optimize.root(g, u, tol=1e-9)
    if not res.success:
        print("am failed")
        return u
    else:
        return res.x



# BDF2

# BDF4

def f(u, t):
    A = np.array([[-5000,4999,0,0],
                  [4999,-5000,0,0],
                  [0,0,0,-10],
                  [0,0,10,0]
                  ])
    return np.dot(A,u)

def uTrue(u,t):
    ut = np.column_stack([
        np.exp(-t) + np.exp(-9999 * t),
        np.exp(-t) - np.exp(-9999 * t),
        np.cos(10 * t),
        np.sin(10 * t)
    ])
    return ut

def errorNorm( u, t):
    """get the 2norm of the difference of approx and true solutions."""
    return np.linalg.norm(uTrue(None, t) - u, axis=1)
    

def findStep(f, u0, tspan, solver, tol=0.05):
    """Bisection to find max step size for error < 5%"""
    h = 0.1
    error = 1000
    while (np.isnan(error)) or error >= 0.05:
        h = h / 10
        u, t = solveIVP(f, u0, tspan, h, solver)
        error = np.linalg.norm(uTrue(None,t[-1]) - u[-1,:], axis=1)
    return h

if __name__ == "__main__":

    # Define IVP parameters
    tspan = [0, 1]
    u0 = np.array([2,0,1,0])

    h = {}
    h["fe"] = findStep(f, u0, tspan, fe)
    h["be"] = findStep(f, u0, tspan, be)
    h["rk4"] = findStep(f, u0, tspan, rk4)
    h["cn"] = findStep(f, u0, tspan, cn)
    h["ab"] = findStep(f, u0, tspan, ab)
    h["am"] = findStep(f, u0, tspan, am)
    print(h["am"])


    # when hardcoded h is desired
    # h = {
    #         "fe": 1e-4,
    #         "be": 1e-3,
    #         "rk4": 2.5e-4,
    #         }
    
    t = {}
    u = {}
    error = {}
    name = ["fe", "be", "rk4", "cn", "ab", "am"]
    func = [fe, be, rk4, cn, ab, am]

    for i in range(len(name)):
        u[name[i]], t[name[i]] = solveIVP(f, u0, tspan, h[name[i]], func[i])
        error[name[i]] = errorNorm(u[name[i]], t[name[i]])
        print(f"{name[i]} h={h[name[i]]} \n\t Final Error: {error[name[i]][-1]}")


    # fig, ax = plt.subplots(figsize=(8, 6))
    # plt.plot(t["fe"], error["fe"], "bo-", label="Euler")
    # plt.plot(t["be"], error["be"], "ro-", label="Back Euler")
    # # plt.plot(t3, u3, "go-", label="True")
    # plt.xlabel("$t$", fontsize=14)
    # plt.ylabel("$y$", fontsize=14)
    # plt.legend(fontsize=12)
    # plt.savefig(f"test.png")
