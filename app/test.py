import numpy as np
from oct2py import octave
import matplotlib.pyplot as plt

def solveIVP(f, u0, tspan, h, solver):
    """
    sets up iteration which uses solver to solve each step.
    """
    t = np.arange(tspan[0], tspan[1] + h, h)
    y = np.zeros((len(t),len(u0)))
    t[0] = tspan[0]
    u[0,:] = u0

    for n in range(len(t) - 1):
        y[n+1,:] = solver(f, u[n,:], t[n], h)
    return t, y

def rk4(f, u, t, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def f(u, t):
    A = np.array([[-5000,4999,0,0],
                  [4999,-5000,0,0],
                  [0,0,0,-10],
                  [0,0,10,0]
                  ])
    return np.dot(A,u)

def uTrue(u,t):
    ut = np.array([
        np.exp(-t) + np.exp(-9999 * t),
        np.exp(-t) - np.exp(-9999 * t),
        np.cos(10*t),
        np.sin(10*t)
        ])
    return ut


if __name__ == "__main__":
    u0 = np.array([2,0,1,0])
    print(f"u0: {u0}")
    print(f"u0': {f(u0,0)}")

