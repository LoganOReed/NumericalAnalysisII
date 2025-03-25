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

    for n in range(len(t) - 1):
        u[n+1,:] = solver(f, u[n,:], t[n], h)
    return t, u

def fe(f, u, t, h):
    """Forward Euler."""
    return u + h * f(u, t)

# back euler
def be(f, u, t, h):
    """Backwards Euler (Newton)."""
    g = lambda x: x - u - h*f(x,t+h)
    res = sp.optimize.root(g, u, tol=1e-9)
    if not res.success:
        print("be failed")
        return u
    else:
        return res.x


def rk4(f, u, t, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# crank nicolson

# Adams-Bashforth (with n-1 true solutions)

# Adams-Moulton (with n-1 true solutions)

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

def errorNorm(uTrue, u, t):
    """get the 2norm of the difference of approx and true solutions."""
    return np.linalg.norm(uTrue(None, t) - u, axis=1)
    

if __name__ == "__main__":

    # Define IVP parameters
    tspan = [0, 1]
    u0 = np.array([2,0,1,0])
    h = {
            "fe": 1e-4,
            "be": 1e-3,
            "rk4": 2.5e-4,
            }
    
    t = {}
    u = {}
    error = {}
    # name = ["fe", "be", "rk4"]
    # func = [fe, be, rk4]
    name = ["fe", "be"]
    func = [fe, be]
    for i in range(len(name)):
        t[name[i]], u[name[i]] = solveIVP(f, u0, tspan, h[name[i]], func[i])
        error[name[i]] = errorNorm(uTrue, u[name[i]], t[name[i]])
        print(f"{name[i]} h={h[name[i]]} \n\t Final Error: {error[name[i]][-1]}")
    # print(f"FE h={h['fe']} \n\t Final Error: {error['fe'][-1]}")
    # print(f"BE h={h['be']} \n\t Final Error: {error['be'][-1]}")


    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(t["fe"], error["fe"], "bo-", label="Euler")
    plt.plot(t["be"], error["be"], "ro-", label="Back Euler")
    # plt.plot(t3, u3, "go-", label="True")
    plt.xlabel("$t$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(f"test.png")

