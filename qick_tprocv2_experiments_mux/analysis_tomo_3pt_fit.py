import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar

def model(n_g, d, nu, mv2e, phi):
    P = d + nu*np.cos(np.pi*np.cos(2*np.pi*mv2e*n_g - phi))
    return P

def phi_minimization(vsweep, data, d, nu, mv2e):
    vsweep = np.asarray(vsweep)
    data = np.asarray(data)

    def cost(phi):
        data_pred = model(vsweep, d, nu, mv2e, phi)
        return np.sum((data - data_pred)**2)

    res = minimize_scalar(cost,bounds = (0, 2*np.pi), method = 'bounded')

    return res.x, res.fun, res.success

def phi_curvefit(vsweep, y, d, nu, mv2e):
    popt, pcov = curve_fit(model, vsweep, data, p0 = [0.0], bounds = (0, 2*np.pi))
    phi = popt[0]
    return phi, pcov

def phi_minimization_normalized(frac_T, data, d, nu):
    x = (np.asarray(data) - d) / abs(nu)

    def cost(phi):
        return np.sum(
            (x + np.cos(np.pi * np.cos(2 * np.pi * frac_T - phi)))**2
        )

    res = minimize_scalar(cost, bounds = (0, 2*np.pi), method = 'bounded')
    return res.x, res.fun, res.success

# unwrap phi: np.unwrap(phi)


