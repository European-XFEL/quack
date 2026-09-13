from typing import Tuple, Optional
import numpy as np
import os

from . import Quack

def get_gaussian(
              omega: np.ndarray,
              E1: np.ndarray,
              E2: np.ndarray,
              O: np.ndarray,
              weight: np.ndarray,
              initial: np.ndarray,
              spectrum: Optional[np.ndarray]=None,
              spectrum_mode: int=-1,
              tol: float=1e-5,
              max_iter: int=2000,
              kappa: float=5,
              step: float=1.0,
              alpha: float=0,
              ):
    r"""
    Find a set of coefficients x for the electric field, such that the intensity square |Ex|^2 matches the observation
    Assume x is described by the sum of two Gaussians.

    Returns: the complex values of the coefficients of the FEL electric field, in time and in energy.
    """

    NW, Ntheta = O.shape
    NW = int(NW)
    Ntheta = int(Ntheta)
    n_obs = NW*Ntheta
    n_support = E1.shape[1]
    w = 1.0

    # reshape and normalize O
    O = np.reshape(O, (-1,))/np.sqrt(np.sum(O**2))

    # reshape weight
    weight = np.reshape(weight, (-1,))

    # initialize
    # initial observations
    y = np.copy(O)
    X = np.ones((n_support,), dtype=O.dtype)
    if spectrum is None:
        spectrum = np.nan*np.zeros((n_support//2,), dtype=O.dtype)

    X = initial.copy()
    assert len(spectrum) == n_support//2
    evolution = np.zeros(max_iter, dtype=np.float64)
    evolution_spec = np.zeros(max_iter, dtype=np.float64)
    evolution_X = np.zeros((max_iter, n_support), dtype=np.float64)
    opt_index = Quack.solve_gaussian_parallel_b(omega, y, X, E1, E2, O, weight, spectrum, spectrum_mode, evolution, evolution_spec,
                                       evolution_X,
                                       tol=tol, max_iter=max_iter, kappa=float(kappa), step=float(step))

    converged = True
    pred = (E1[:,:,opt_index-1] @ X)**2 + (E2[:,:,opt_index-1] @ X)**2
    pred /= np.sqrt(np.sum(pred**2))

    return opt_index-1, converged, X, pred, evolution, evolution_spec, evolution_X


def get_gaussian_two_pols(
              E1A: np.ndarray,
              E2A: np.ndarray,
              E1B: np.ndarray,
              E2B: np.ndarray,
              O: np.ndarray,
              weight: np.ndarray,
              initial: np.ndarray,
              spectrum: Optional[np.ndarray]=None,
              spectrum_mode: int=-1,
              tol: float=1e-5,
              max_iter: int=2000,
              kappa: float=5,
              step: float=1.0,
              alpha: float=0
              ):
    r"""
    Find a set of coefficients x for the electric field, such that the intensity square |Ex|^2 matches the observation
    Assume x is described by the sum of two Gaussians.

    Returns: the complex values of the coefficients of the FEL electric field, in time and in energy.
    """

    NW, Ntheta = O.shape
    NW = int(NW)
    Ntheta = int(Ntheta)
    n_obs = NW*Ntheta
    n_support = E1A.shape[1]
    w = 1.0

    # reshape and normalize O
    O = np.reshape(O, (-1,))/np.sqrt(np.sum(O**2))

    # reshape weight
    weight = np.reshape(weight, (-1,))

    # initialize
    # initial observations
    y = np.copy(O)
    XA = np.ones((n_support,), dtype=O.dtype)
    XB = np.ones((n_support,), dtype=O.dtype)
    if spectrum is None:
        spectrum = np.nan*np.zeros((n_support//2,), dtype=O.dtype)

    XA = initial.copy()
    XB = initial.copy()
    assert len(spectrum) == n_support//2
    evolution = np.zeros(max_iter, dtype=np.float64)
    evolution_spec = np.zeros(max_iter, dtype=np.float64)
    opt_index = Quack.solve_gaussian_parallel_two_pols_b(y,
                                                XA, XB,
                                                E1A, E2A,
                                                E1B, E2B,
                                                O,
                                                weight,
                                                spectrum, spectrum_mode,
                                                evolution, evolution_spec,
                                                tol=tol, max_iter=max_iter,
                                                kappa=float(kappa),
                                                step=float(step),
                                                alpha=float(alpha)
                                                )

    converged = True
    pred = (E1A[:,:,opt_index-1] @ XA + E1B[:,:,opt_index-1] @ XB)**2 + (E2A[:,:,opt_index-1] @ XA + E2B[:,:,opt_index-1] @ XB)**2
    pred /= np.sqrt(np.sum(pred**2))

    return opt_index-1, converged, XA, XB, pred, evolution, evolution_spec
