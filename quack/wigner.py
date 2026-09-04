import numpy as np
from typing import Tuple, Optional

from scipy.interpolate import interp1d
from quack.constants import *

from functools import partial
from scipy.optimize import minimize

def log(msg: str):
    """Print a log message."""
    print(msg)

def inf_norm(v: np.ndarray):
    """Returns the infinite norm of v."""
    return np.amax(np.abs(v), axis=-1)

def to_real(v: np.ndarray) -> np.ndarray:
    """Transform a vector of dimension `D` into a vector of twice its dimension containing its real and imaginary parts."""
    return np.concatenate((np.real(v), np.imag(v)))

def to_complex(v: np.ndarray) -> np.ndarray:
    """Transform a vector containing the concatenation of real and imaginary parts into a complex vector with half the dimension."""
    N = len(v)//2
    return v[:N] + 1j*v[N:]

class WignerDistribution(object):
    """
    Representation of a Wigner distribution.

    Args:
      W: The wigner distribution as a 2D array.
      t: The time axis [fs].
      photon_energy: The energy axis [eV].
      dW: Uncertainty.
    """
    def __init__(self, W: np.ndarray, t: np.ndarray, photon_energy: np.ndarray, dW=None):
        self.W = W
        self.dW = dW
        self.t = t
        self.omega = photon_energy

    @property
    def time_spectrum(self):
        return self.W.sum(0)

    @property
    def energy_spectrum(self):
        return self.W.sum(1)

def get_wigner(td_scale: np.ndarray, field: np.ndarray, unc=None) -> WignerDistribution:
    """
    Calculate the Wigner distribution from the field.

    Args:
      td_scale: The time axis in au.
      field: An array containing the electric field in a SVAE complex representation.
    """
    N = len(field)
    def get_W(field):
        # matrix field with copies of itself over rows
        field = np.tile(field, (N, 1))
        # two copies of it
        F1 = field
        F2 = np.copy(F1)
        # for each row, roll F1 and F2 by delta t/2, scanning delta t
        # F1[delta t, t] = F1(t - delta t/2)
        # F2[delta t, t] = F2(t + delta t/2)
        for i in range(N):
            ind1 = -int(np.floor((N / 2 - i) / 2))
            ind2 = int(np.ceil((N / 2 - i) / 2))
            F1[i] = np.roll(F1[i], ind1)
            F2[i] = np.roll(F2[i], ind2)
        # Gamma = F1^* F2 and shift omega to center it
        Gamma = np.fft.fftshift(np.conj(F1) * F2, 0)
        # W(omega, t) = fft(Gamma(delta t, t))
        W = np.fft.fft(Gamma, axis=0)
        # shift back towards original fft format
        W = np.fft.fftshift(W, 0)
        W = np.real(W / N)
        return W
    W = get_W(field)
    dW = None if unc is None else get_W(field+unc) - W

    # time scale
    t_hat = np.copy(td_scale)

    # get energy scale
    h_eV_fs = h*eV_per_au*fs_per_au
    d = (td_scale[1] - td_scale[0])
    #phen = h_eV_fs*0.5*np.linspace(-1.0/d, 1.0/d, len(td_scale))
    phen = np.linspace(-np.pi/d, np.pi/d, len(td_scale))
    return WignerDistribution(W, t_hat, phen, dW=dW)

