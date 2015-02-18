
# coding: utf-8

import numpy as np
from scipy import integrate

def N_p(n):
    exponent = 0.532301 * np.log10(n) + 19.636552
    return 10**exponent
def N_eq(n):
    exponent = 0.626204 * np.log10(n) + 19.573490
    return 10**exponent
def N_eff(n):
    Omega_pole = 1.840282
    fourpi = 4 * np.pi
    a = 2 * Omega_pole / fourpi
    b = (fourpi - 2*Omega_pole)/fourpi
    x = N_p(n)
    y = N_eq(n)
    return a*x + b*y

def sigma(freq, species):
    def epsilon(nu, nu_ion):
        return np.sqrt(nu / nu_ion - 1.0)
    
    if (species == 'H'):
        Z = 1.
        nu_ion = 3.3e15
    elif (species == 'He'):
        Z = 0.89
        nu_ion = 5.95e15
    elif (species == 'HeII'):
        Z = 2.0
        nu_ion = 1.32e16
    else:
        raise KeyError

    ep = epsilon(freq, nu_ion)
    x =  np.exp(4.0 - (4.0*np.arctan(ep) / ep)) / (1.0-np.exp(-2.0*np.pi / ep))
    return x * 6.3e-18 / np.power(Z,2.0) * np.power(nu_ion/freq,4.0) 

def k_integrand(frequency, species, nu_0, alpha, n, attenuation=True, tau=None, tau_sig=None):
    h_nu = 6.6262e-27
    f_nu = unnormed_flux(frequency, nu_0, alpha)
    sigma_nu = sigma(frequency, species)
    if attenuation:
        if tau is None:
            if tau_sig is None:
                tau = - sigma_nu * N_eff(n)
            else:
                tau = - tau_sig * N_eff(n)
        return f_nu * sigma_nu * np.exp(tau) / (h_nu * frequency)
    else:
        return f_nu * sigma_nu / (h_nu * frequency)

def h_integrand(frequency, species, nu_0, alpha, n, attenuation=True, tau=None, tau_sig=None):
    h_nu = 6.6262e-27
    if (species == 'H'):
        nu_ion = 3.3e15
    elif (species == 'He'):
        nu_ion = 5.95e15
    elif (species == 'HeII'):
        nu_ion = 1.32e16
    else:
        raise KeyError

    f_nu = unnormed_flux(frequency, nu_0, alpha)
    sigma_nu = sigma(frequency, species)
    if attenuation:
        if tau is None:
            if tau_sig is None:
                tau = - sigma_nu * N_eff(n)
            else:
                tau = - tau_sig * N_eff(n)
        return f_nu * sigma_nu * np.exp(tau) * (1.0 - nu_ion / frequency)
    else:
        return f_nu * sigma_nu * (1.0 - nu_ion / frequency)

@np.vectorize
def k_rates(n, species, attenuation=True, tau=None, tau_sig=None):
    h_eV = 4.13567e-15
    E_0 = 1e3
    E_min = 1e3
    E_max = 1e4
    alpha = -1.5
    ans, err = integrate.quad(k_integrand, E_min/h_eV, E_max/h_eV, args=(species, E_0/h_eV, alpha, n, attenuation, tau, tau_sig))
    return ans

@np.vectorize
def h_rates(n, species, attenuation=True, tau=None, tau_sig=None):
    h_eV = 4.13567e-15
    E_0 = 1e3
    E_min = 1e3
    E_max = 1e4
    alpha = -1.5
    ans, err = integrate.quad(h_integrand, E_min/h_eV, E_max/h_eV, args=(species, E_0/h_eV, alpha, n, attenuation, tau, tau_sig))
    return ans

