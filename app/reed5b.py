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
    elif (solver == bdf4):
        u[1,:] = uTrue(None,t[0] + h)
        u[2,:] = uTrue(None,t[0] + 2*h)
        u[3,:] = uTrue(None, t[0] + 3*h)
        for n in range(3, len(t) - 1):
            u[n+1,:] = solver(f, u[n,:], t[n], h, [u[n-1,:],u[n-2,:],u[n-3,:]])
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
def bdf2(f, u, t, h):
    """BDF2."""
    y1 = u
    y2 = lambda x: x - u - (1.0/4.0)*h*(f(u,t) + f(x,t+0.5*h))
    res = sp.optimize.root(y2, u, tol=1e-9)
    if not res.success:
        print("bdf2 y2 failed")
        return u
    y2 = res.x
    y3 = lambda x: x - u - (1.0/3.0)*h*(f(u,t) + f(y2,t+0.5*h) + f(x,t+h))
    res = sp.optimize.root(y3, u, tol=1e-9)
    if not res.success:
        print("bdf2 y3 failed")
        return u
    y3 = res.x
    return y3

# BDF4
def bdf4(f,u,t,h,uprevs):
    """BDF4."""
    g = lambda x: 25*x - 48*u + 36*uprevs[0] - 16*uprevs[1] +3*uprevs[2] - 12*h*f(x,t+h)
    res = sp.optimize.root(g, u, tol=1e-9)
    if not res.success:
        print("bdf4 failed")
        return u
    else:
        return res.x

   

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
    a = 1e-7
    b = 1e-1
    tol = 1e-9
    maxIter = 100
    u, t = solveIVP(f, u0, tspan, b, solver)
    error = np.linalg.norm(uTrue(None,t[-1]) - u[-1,:], axis=1) - 0.05
    if np.isnan(error):
        error = 1e5
    i = 1
    while i <= maxIter:
        c = (a + b) / 2
        u, t = solveIVP(f, u0, tspan, c, solver)
        error = np.linalg.norm(uTrue(None,t[-1]) - u[-1,:], axis=1) - 0.05
        if np.isnan(error):
            error = 1e5
        if error == 0 or (b-a)/2 < tol:
            return c
        i = i + 1
        if error > 0:
            b = c
        else:
            a = c
    print("could not find root within step limit")

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
    h["bdf2"] = findStep(f, u0, tspan, bdf2)
    h["bdf4"] = findStep(f, u0, tspan, bdf4)

    # To create text file for writeup
    # fout = "./steps.txt"
    # fo = open(fout, "w")
    # for k, v in h.items():
    #     fo.write(str(k) + '\t' + str(v) + '\n')
    # fo.close()

    # when hardcoded h is desired
    # h = {
    #         "fe":	0.00019995342902019621,
    #         "be":	0.0010252355350650847,
    #         "rk4":	0.00027849588101431724,
    #         "cn":	0.010945369696036728,
    #         "ab":	0.0001000025001473725,
    #         "am":	0.0005991655690036715,
    #         "bdf2":	0.03431726929393187,
    #         "bdf4":	0.04101112199843749
    #         }
    
    t = {}
    u = {}
    error = {}
    name = ["fe", "be", "rk4", "cn", "ab", "am", "bdf2", "bdf4"]
    func = [fe, be, rk4, cn, ab, am, bdf2, bdf4]

    for i in range(len(name)):
        u[name[i]], t[name[i]] = solveIVP(f, u0, tspan, h[name[i]], func[i])
        error[name[i]] = errorNorm(u[name[i]], t[name[i]])
        print(f"{name[i]} h={h[name[i]]} \n\t Final Error: {error[name[i]][-1]}")

