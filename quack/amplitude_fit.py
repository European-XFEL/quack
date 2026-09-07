from typing import Tuple, Dict, Union, List, Optional
from threading import Lock
import numpy as np
import h5py

import dataclasses


from quack.constants import (eV_per_au,
                             fs_per_au,
                             c)
from quack.wigner import WignerDistribution, get_wigner
from quack.utils import fw

from quack.amplitude_torch import optimize_with_torch
from quack.amplitude_julia import get_field as get_field_julia
from quack.amplitude_julia import get_field_two_pols as get_field_two_pols_julia
from quack.amplitude_julia import get_field_nlls

from functools import partial
import scipy
from scipy.interpolate import PchipInterpolator

import concurrent

from copy import deepcopy

import logging

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_rnorm(a: np.ndarray, b: np.ndarray, w: Optional[np.ndarray]=None) -> float:
    an = a/np.sqrt(np.nansum(a**2))
    bn = b/np.sqrt(np.nansum(b**2))
    if w is None:
        w = np.ones_like(a)
    return np.nansum((w**2)*(an - bn)**2)/np.nansum(w**2)

def get_cnorm(a: np.ndarray, b: np.ndarray, w: Optional[np.ndarray]=None) -> float:
    if w is None:
        w = np.ones_like(a)
    an = (w*a)/np.sqrt(np.nansum((w*a)**2))
    bn = (w*b)/np.sqrt(np.nansum((w*b)**2))
    return np.nansum(an*bn)

def get_masked_cnorm(a: np.ndarray, b: np.ndarray, w: Optional[np.ndarray]=None) -> float:
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if w is None:
        w = np.ones_like(a)
    an = (w*a)[mask]
    bn = (w*b)[mask]
    an = an/np.sqrt(np.nansum(an**2))
    bn = bn/np.sqrt(np.nansum(bn**2))
    return np.nansum((w)*(an*bn))

def get_l1(a: np.ndarray, b: np.ndarray, w: Optional[np.ndarray]=None) -> float:
    an = a/np.amax(a)
    bn = b/np.amax(b)
    if w is None:
        w = np.ones_like(a)
    return np.sum(w*np.abs(an - bn))/np.sum(w)

@dataclasses.dataclass
class QUACKAmpSolution:
    """
    Data class representing the output of amplitude calculations.
    """
    Ew: np.ndarray | None = None
    Et: np.ndarray | None = None
    additional_Ew: np.ndarray | None = None
    additional_Et: np.ndarray | None = None
    omega: np.ndarray | None = None
    t: np.ndarray | None = None
    pred: np.ndarray | None = None
    unc: np.ndarray | None = None
    unc_per_angle: np.ndarray | None = None
    unc_per_energy: np.ndarray | None = None
    wigner: np.ndarray | None = None
    additional_wigner: np.ndarray | None = None
    Up: float | None = None
    raw: np.ndarray | None = None
    additional_raw: np.ndarray | None = None
    converged: bool = False


def get_E(b_all, real_omega_support):
    NW, Ntheta, Nomega, NA = b_all.shape
    E1 = list()
    E2 = list()
    for A in range(NA):
        b = np.reshape(b_all[..., A], (NW*Ntheta, Nomega))
        re_b = np.real(b)
        im_b = np.imag(b)
        A = re_b
        B = -im_b
        C = im_b
        D = re_b
        # re_Ey = A re_y + B im_y
        # im_Ey = C re_y + D im_y
        E1 += [np.concatenate([A, B], axis=1)[:, real_omega_support]]
        E2 += [np.concatenate([C, D], axis=1)[:, real_omega_support]]
    E1 = np.stack(E1)
    E2 = np.stack(E2)
    return E1, E2

#from scipy.interpolate import CubicSpline
def interpolate_energy_axis(target_energy: np.ndarray, original_energy: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Receives a 2D data of shape (n_energy, n_angles) and interpolates the energy axis trom the original
    values to the target values.
    """
    #return np.apply_along_axis(lambda arr: np.interp(target_energy, original_energy, arr,
    #                                                 left=0, right=0),
    #                           axis=0, arr=data)
    o = np.apply_along_axis(lambda arr: PchipInterpolator(original_energy, arr, extrapolate=False)(target_energy),
                               axis=0, arr=data)
    o = np.nan_to_num(o)
    return o

@dataclasses.dataclass
class Basis:
    b: np.ndarray | None = None
    # real and imaginary projections
    E1: np.ndarray | None = None
    E2: np.ndarray | None = None
    # photo-electron kinetic energy
    W: np.ndarray | None = None
    # photo-electron angle
    theta: np.ndarray | None = None
    # FEL energy
    omega: np.ndarray | None = None
    # streaking laser time of interaction
    time: np.ndarray | None = None
    # ponderomotive potential
    Up: np.ndarray | None = None
    # period of the laser
    Tl: float = 0.0
    # Ip
    Ip: float = 0.0
    # oversampling
    oversampling: int = 1

    # omega support
    omega_support: np.ndarray | None = None
    # omega support in the (Re{x}, Im{x}) complexification domain
    real_omega_support: np.ndarray | None = None
    # FEL energy axis
    fel_energy_axis: np.ndarray | None = None

    # data photo-electron kinetic energy axis
    data_energy_axis: np.ndarray | None = None

    # number of bins in photo-electron kinetic energy
    NW: int = 0
    # number of angle bins
    Ntheta: int = 0
    # number of FEL energy bins
    Nomega: int = 0
    # number of ponderomotive potential bins
    NA: int = 0

    @classmethod
    def from_hdf5(cls, h5grp, energy_axis: Optional[np.ndarray]=None, angle_axis: Optional[np.ndarray]=None, extra_energy_support: float=0.0):
        """
        Load simulation data from HDF5 group object and return a `Basis` object.
        """
        basis = cls()
        basis.b = h5grp["b"][()]
        basis.W = h5grp["W_axis"][()]
        basis.theta = h5grp["theta_axis"][()]
        basis.omega = h5grp["omega_axis"][()]
        basis.time = h5grp["time_axis"][()]
        basis.Up = h5grp["Up_axis"][()]
        basis.Tl = h5grp["Tl"][()]
        basis.Ip = h5grp["Ip"][()]
        basis.oversampling = h5grp["oversampling"][()]
        basis.NW, basis.Ntheta, basis.Nomega, basis.NA = basis.b.shape
        basis.data_energy_axis = deepcopy(basis.W)

        return basis.slice(energy_axis, angle_axis, extra_energy_support)

    def slice(self,
                   energy_axis: Optional[np.ndarray]=None,
                   angle_axis: Optional[np.ndarray]=None,
                   extra_energy_support: float=0.0):
        """Slice W and and axes from `self` and return a new self."""
    
        if energy_axis is not None:
            logging.info(f"Energy axis received with bounds ({energy_axis[0]:.2f}, {energy_axis[-1]:.2f})")
            logging.info(f"Existing energy axis with bounds ({self.W[0]:.2f}, {self.W[-1]:.2f})")
            min_energy = np.amin(energy_axis)
            max_energy = np.amax(energy_axis)
            energy_slice = (self.W >= min_energy-1e-6) & (self.W <= max_energy+1e-6)
            self.b = self.b[energy_slice, ...]
            self.W = self.W[energy_slice]
            self.NW = len(self.W)
            logging.info(f"Selected energy axis with bounds ({self.W[0]:.2f}, {self.W[-1]:.2f}) and {self.NW} bins")
            self.data_energy_axis = deepcopy(energy_axis)
            logging.info(f"Given energy axis shall be used for interpolation.")
        if angle_axis is not None:
            logging.info(f"Angle axis received with bounds ({angle_axis[0]:.2f}, {angle_axis[-1]:.2f})")
            logging.info(f"Existing angle axis with bounds ({self.theta[0]:.2f}, {self.theta[-1]:.2f})")
            angle_index = []
            for angle in angle_axis:
                angle_index += [find_nearest(self.theta, angle)]
            angle_index = np.asarray(angle_index)
            self.b = self.b[:, angle_index, ...]
            self.theta = self.theta[angle_index]
            self.Ntheta = len(self.theta)
            logging.info(f"Selected angle axis with bounds ({self.theta[0]:.2f}, {self.theta[-1]:.2f}) and {self.Ntheta} bins")
        assert self.NW == self.b.shape[0]
        assert self.Ntheta == self.b.shape[1]
    
        self.fel_energy_axis = np.ascontiguousarray((self.Ip + self.W).astype(np.float64))
        min_e = np.amin(self.fel_energy_axis)
        max_e = np.amax(self.fel_energy_axis)
        self.omega_support = (self.omega >= min_e - extra_energy_support) & (self.omega <= max_e + extra_energy_support)
        self.real_omega_support = np.concatenate((self.omega_support, self.omega_support))
    
        self.time = np.ascontiguousarray(self.time.astype(np.float64))
        self.omega = np.ascontiguousarray(self.omega.astype(np.float64))
    
        self.E1, self.E2 = get_E(self.b, self.real_omega_support)
        self.E1 = np.moveaxis(self.E1, 0, -1) # move outer most axis to last place such that
                                                    # E1[:,:,i] is a fortran contigous array
        self.E2 = np.moveaxis(self.E2, 0, -1) # move outer most axis to last place such that
                                                    # E1[:,:,i] is a fortran contigous array
        self.E1 = np.asfortranarray(self.E1.astype(np.float64))
        self.E2 = np.asfortranarray(self.E2.astype(np.float64))
    
        return self

    def is_compatible_with(self, other):
        """Returns True if the two bases are compatible."""
        good = np.allclose(self.W, other.W)
        good &= np.allclose(self.theta, other.theta)
        good &= np.allclose(self.omega, other.omega)
        good &= np.allclose(self.time, other.time)
        good &= np.allclose(self.Up, other.Up)
        good &= np.allclose(self.Tl, other.Tl)
        good &= np.allclose(self.Ip, other.Ip)
        good &= np.allclose(self.oversampling, other.oversampling)
        return good


class AmplitudeSolver(object):
    """
    Given precalculated simulated basis b in `simulation` (obtained from quack-sim), this solves
    the problem |b c|^2 = O to find electric field Fourier coefficients that explain an observation O.

    Use it as folllows:
    ```
    # energy and angle axes from observations
    # energy_axis is the photo-electron kinetic energy axes expected for the observation
    # it is used to slice the simulation data to select only relevant entries
    # in this example: bins of 1 eV in the photo-electron kinetic energy of [85 eV, 135 eV]
    energy_axis = np.arange(85.0, 135.0)
    angle_axis = np.arange(0.0, 360.0, 22.5)*np.pi/180.0
    with h5py.File("my_simulation_basis.h5", "r") as fid:
        basis_horizontal = Basis.from_hdf5(fid, energy_axis, angle_axis)
    with h5py.File("my_simulation_basis2.h5", "r") as fid:
        basis_circular = Basis.from_hdf5(fid, energy_axis, angle_axis)

    # additional_basis can be omitted (set to `None`) if the FEL has a unique polarization
    # if not, both polarizations are fit simultaneously
    solver = AmplitudeSolver(basis=basis_horizontal, additional_basis=basis_circular)

    # apply a fit to an observation with shape (NW, Ntheta)
    # where NW is the number of photo-electron momentum bins
    # and Ntheta is the number of photo-electron angle bins
    fit_result = solver.solve(obs)
    ```

    Args:
      basis: Basis object.
      additional_basis: Basis object for fitting two FEl polarizations at once. Set to `None` to fit a single polarization.
    """
    def __init__(self, basis: Basis, additional_basis: Optional[Basis]=None):
        self.basis = basis
        self.additional_basis = additional_basis
        # if there is an additional basis, they must match
        if self.additional_basis is not None:
            assert self.basis.is_compatible_with(self.additional_basis)

    @property
    def oversampled_energy_axis(self):
        """Energy axis."""
        return basis.omega

    @property
    def time_axis(self):
        n_omega = len(self.basis.omega)
        dt = self.basis.oversampling*self.basis.Tl/n_omega
        return np.arange(0.0, self.basis.Tl, dt)

    @property
    def energy_axis(self):
        n_omega = len(self.basis.omega)
        nbins_time = n_omega//self.basis.oversampling
        dt = self.basis.Tl/nbins_time
        return  np.fft.fftshift(np.fft.fftfreq(nbins_time, dt/fs_per_au)*eV_per_au*2*np.pi + self.basis.omega[0])

    def post_process(self, raw: np.ndarray):
        """
        Get spectra from a solution containing only the raw data.

        Args:
          raw: Raw solution in the complexified domain.

        Returns: Et, Ew to be used in the solution object.
        """
        # add support in omega
        n_omega = len(self.basis.omega)
        shape = 2*n_omega
        if len(raw.shape) == 2:
            shape = (raw.shape[0], 2*n_omega)
        new_X = np.zeros(shape, dtype=raw.dtype)
        new_X[..., self.basis.real_omega_support] = raw
        X = new_X[..., :n_omega] + 1j*new_X[..., n_omega:]
        X = np.concatenate((X[..., 0, np.newaxis], X[..., 1:][..., ::-1]), axis=-1)

        #if estimate_evolution:
        #    new_eX = np.zeros((sol.evolution_Et.shape[0], 2*n_omega), dtype=sol.raw.dtype)
        #    new_eX[:, self.real_omega_support] = sol.evolution_Et
        #    eX = new_eX[:, :n_omega] + 1j*new_eX[:, n_omega:]
        #    eX = np.concatenate((eX[:, 0, np.newaxis], eX[:, 1:][:, ::-1]), axis=-1)

        # get time spectrum from omega spectrum
        if self.basis.oversampling == 1:
            Et = np.conj(np.fft.ifft(X, axis=-1))
        else:
            n = n_omega
            k = np.arange(n)[:, None]
            m = np.arange(n//self.basis.oversampling)[None, :]
            x = (1.0/float(n))*(X @ np.exp(2*np.pi*1j*m.astype(np.float64)*k.astype(np.float64)/float(n)))
            Et = np.conj(x)

        # now get the corresponding energy spectrum
        Ew = np.fft.fftshift(np.fft.fft(Et, axis=-1), axes=-1)

        return Et, Ew

    def guess_bandwidth(self, obs: np.ndarray) -> int:
        """
        Produce educated guess for the bandwidth.

        Args:
          obs: Observed data.

        Returns: Bandwidth as sigma in eV.
        """
        opt_angle = np.unravel_index(np.argmax(obs), obs.shape)[1]
        e_projection = obs[:, opt_angle]
        fwhm = fw(self.basis.W, e_projection)
        return fwhm/2.355

    def get_moments(self, obs: np.ndarray) -> Tuple[float, float]:
        """
        Produce educated guess for the mean photo-electron energy.

        Args:
          obs: Observed data.

        Returns: Mean and sigma photo-electron energy in eV.
        """
        e_projection = np.sum(obs, 1)/np.sum(obs)
        mu = np.sum(e_projection*self.basis.W)
        #sigma = np.sqrt(np.sum(e_projection*(self.W - mu)**2))
        sigma = fw(self.basis.W, e_projection)/2.355
        return mu, sigma

    def guess_Up(self, e: float, sigma: float, bw: float) -> float:
        """
        Initial guess of Up from mean energy, sqrt. var. and bandwidth.

        Returns: Guess of Up.
        """
        # estimate maximum momentum kick
        deltaE = sigma*2.355/2 - bw*2.355/2
        A = (np.sqrt(2*e/eV_per_au + 2*deltaE/eV_per_au) - np.sqrt(2*e/eV_per_au))*c
        if A < 0:
            A = 0.0
        # estimate the ponderomotive potential
        Up = A**2/(4*c**2)*eV_per_au
        return Up

    def get_initial_guess(self, s: np.ndarray) -> np.ndarray:
        """
        Produce initial guess for E(omega).
        """
        bw = self.guess_bandwidth(s)
        if bw <= 0:
            bw = 1.0 # TODO: no idea what to do in this case, but prevent nans
        e, sigma = self.get_moments(s)
        e += self.basis.Ip
        #Up = self.guess_Up(e, sigma, bw)
        initial = np.exp(-0.5*(self.basis.omega - e)**2/(bw**2))*np.exp(1j*1.0)
        initial = np.concatenate((np.real(initial), np.imag(initial)))
        initial = initial[self.basis.real_omega_support]
        return initial

    def solve(self, obs: np.ndarray,
              weight: Optional[np.ndarray]=None,
              tol: Optional[float]=None,
              method: str="julia",
              guess_initial: bool=True,
              max_iter: Optional[int]=None,
              nthreads: int=40,
              Up: Optional[np.array]=None,
              calculate_wigner: bool=True,
              spectrum: Optional[np.ndarray]=None,
              spectrum_axis: Optional[np.ndarray]=None,
              spectrum_mode: int=-1,
              kappa: float=200.0,
              gpu: bool=False,
              constrain: str="everywhere",
              step: float=1.0,
              alpha: float=0.0,
              ) -> QUACKAmpSolution:
        """
        Find the spectrogram for a given eTOF observation in `obs`.

        Args:
          obs: Observed data.
          weight: Weight proportional to the data accuracy. Same shape as observation.
          tol: Tolerance for convergence. For "julia": |rnorm[100-i] - rnorm[i]| < tol (default 1e-20).
               For "nlls": relative step size below which the iteration stops (default 1e-4).
          method: "julia" for the primal-dual (proximal) iteration, "nlls" for Levenberg-Marquardt on the
                  equivalent nonlinear least-squares problem (single basis only), "torch" for gradient descent.
          guess_initial: Make a guess of the initial energy spectrum to get faster convergence.
          max_iter: Maximum number of iterations (default 2000 for "julia", 30 for "nlls").
          nthreads: Number of threads if parallelizing.
          Up: If given, restrict Up values to test to these.
          kappa: Ratio of step sizes between angular streaking observation and spectral constraint. Must be bigger than 0.
          constrain: If "everywhere", constrain the energy spectrum to zero when nan.
                     If "partial", only constrain where given, but constrain to zero out of the eTOF support.

        Returns: A QUACKAmpSolution object with several details.
        """
        # find out maximum Up, but clip it to maximum in basis
        if Up is None:
            Up = self.basis.Up[:]
        # set up default tolerance
        if tol is None:
            if method == 'nlls':
                tol = 1e-4
            else:
                tol = 1e-20
        if max_iter is None:
            if method == 'nlls':
                max_iter = 30
            else:
                max_iter = 2000

        iUp = np.unique(np.searchsorted(self.basis.Up, Up))
        iUp[iUp >= len(self.basis.Up)] = len(self.basis.Up) - 1
        # if weight is not provided, fallback to ones
        if weight is None:
            w = np.ones_like(obs)
        else:
            w = np.copy(weight)
        # interpolate energy axis to match self.W
        s = interpolate_energy_axis(self.basis.W, self.basis.data_energy_axis, obs)
        w = interpolate_energy_axis(self.basis.W, self.basis.data_energy_axis, w)

        # set up spectrum in amplitude and interpolate it if it is given
        int_spectrum = None
        reference_spectrum = None
        if spectrum is not None and spectrum_axis is not None:
            reference_spectrum = PchipInterpolator(spectrum_axis, np.sqrt(np.clip(spectrum, a_min=0, a_max=None)), extrapolate=False)(self.basis.omega)
            #reference_spectrum = np.sqrt(np.interp(self.omega, spectrum_axis, np.clip(spectrum, a_min=0, a_max=None), left=np.nan, right=np.nan))
            int_spectrum = np.copy(reference_spectrum)
            if constrain == "partial":
                int_spectrum[(self.basis.omega < self.basis.data_energy_axis[0]+self.basis.Ip) | (self.basis.omega > self.basis.data_energy_axis[-1]+self.basis.Ip)] = 0.0
            int_spectrum = int_spectrum[self.basis.omega_support]
            int_spectrum /= np.sqrt(np.nansum(int_spectrum**2))
            if constrain == "everywhere":
                np.nan_to_num(int_spectrum, copy=False)

        # also clip it at zero: it is nonsense to get negative intensities or weights
        s = np.ascontiguousarray(np.clip(s/np.sqrt(np.sum(s**2)), a_min=0, a_max=None))
        w = np.ascontiguousarray(np.clip(w/np.amax(w), a_min=0, a_max=None))

        # get an initial guess
        initial = None
        if guess_initial:
            if spectrum is None or spectrum_axis is None:
                initial = self.get_initial_guess(s)
            else:
                initial = PchipInterpolator(spectrum_axis, np.sqrt(np.clip(spectrum, a_min=0, a_max=None)), extrapolate=False)(self.basis.omega)
                np.nan_to_num(initial, copy=False)
                #initial = np.sqrt(np.interp(self.omega, spectrum_axis, np.clip(spectrum, a_min=0, a_max=None), left=0, right=0))
                initial /= np.sqrt(np.sum(initial**2))
                initial = initial*np.exp(1j*0.00)
                initial = np.concatenate((np.real(initial), np.imag(initial)))
                initial = initial[self.basis.real_omega_support]

        # prepare output
        solution = QUACKAmpSolution()
        solution.converged = True
        solution.omega = self.energy_axis
        solution.t = self.time_axis
        # solve it and fill the output
        if method == 'torch':
            idx_A, converged, X, pred, evolution, evolution_spec, eX = optimize_with_torch(E1=self.basis.E1[:,:,iUp],
                                                                       E2=self.basis.E2[:,:,iUp],
                                                                       O=s,
                                                                       weight=w,
                                                                       spectrum=int_spectrum,
                                                                       spectrum_mode=spectrum_mode,
                                                                       initial=initial,
                                                                       tol=tol,
                                                                       max_iter=max_iter,
                                                                       kappa=kappa,
                                                                       gpu=gpu,
                                                                       )
            solution.raw = X
            solution.pred = pred.reshape(self.basis.NW, self.basis.Ntheta)
            solution.evolution = evolution
            solution.evolution_spec = evolution_spec
            solution.evolution_Et = eX
            solution.real_omega_support = self.basis.real_omega_support
            solution.rnorm = get_rnorm(pred.flatten(), s.flatten(), w.flatten())
            solution.target = solution.rnorm
            solution.unc = np.amax(np.fabs(solution.pred/np.amax(pred) - s/np.amax(s)))
            Et, Ew = self.post_process(X)
            solution.evolution_Et, _ = self.post_process(eX)
            solution.Et = Et
            solution.Ew = Ew
        elif method in ('julia', 'nlls'):
            if method == 'nlls':
                if self.additional_basis is not None:
                    raise NotImplementedError("method='nlls' does not support an additional basis yet.")
                idx_A, converged, X, pred = get_field_nlls(
                    E1=self.basis.E1[:,:,iUp],
                    E2=self.basis.E2[:,:,iUp],
                    O=s,
                    weight=w,
                    initial=initial,
                    spectrum=int_spectrum,
                    spectrum_mode=spectrum_mode,
                    tol=tol,
                    max_iter=max_iter,
                    kappa=kappa,
                )
                # no per-iteration trace from the least-squares solver
                evolution = np.zeros(0)
                evolution_spec = np.zeros(0)
                solution.raw = X
                Et, Ew = self.post_process(X)
                solution.Et = Et
                solution.Ew = Ew
            elif self.additional_basis is not None:
                idx_A, converged, XA, XB, pred, evolution, evolution_spec = get_field_two_pols_julia(
                    E1A=self.basis.E1[:,:,iUp],
                    E2A=self.basis.E2[:,:,iUp],
                    E1B=self.additional_basis.E1[:,:,iUp],
                    E2B=self.additional_basis.E2[:,:,iUp],
                    O=s,
                    weight=w,
                    initial=initial,
                    spectrum=int_spectrum,
                    spectrum_mode=spectrum_mode,
                    tol=tol,
                    max_iter=max_iter,
                    kappa=kappa,
                    step=step,
                    alpha=alpha,
                )
                solution.raw = XA
                solution.additional_raw = XB
                Et, Ew = self.post_process(XA)
                solution.Et = Et
                solution.Ew = Ew
                a_Et, a_Ew = self.post_process(XB)
                solution.additional_Et = a_Et
                solution.additional_Ew = a_Ew
            else:
                idx_A, converged, X, pred, evolution, evolution_spec, eX = get_field_julia(
                    E1=self.basis.E1[:,:,iUp],
                    E2=self.basis.E2[:,:,iUp],
                    O=s,
                    weight=w,
                    initial=initial,
                    spectrum=int_spectrum,
                    spectrum_mode=spectrum_mode,
                    tol=tol,
                    max_iter=max_iter,
                    kappa=kappa,
                    step=step,
                    alpha=alpha,
                )
                solution.raw = X
                Et, Ew = self.post_process(X)
                solution.evolution_Et, _ = self.post_process(eX)
                solution.Et = Et
                solution.Ew = Ew
            solution.pred = pred.reshape(self.basis.NW, self.basis.Ntheta)
            solution.evolution = evolution
            solution.evolution_spec = evolution_spec
            solution.real_omega_support = self.basis.real_omega_support
            solution.rnorm = get_rnorm(pred.flatten(), s.flatten(), w.flatten())
            solution.target = solution.rnorm
            solution.unc = np.amax(np.fabs(solution.pred/np.amax(pred) - s/np.amax(s)))
        else:
            raise NotImplementedError("Methods avalable are: 'julia', 'nlls', 'torch'.")
        solution.Up = self.basis.Up[iUp[idx_A]]
        solution.kick = np.sqrt(4*solution.Up/eV_per_au)*c*eV_per_au
        solution.unc_per_angle = np.amax(np.fabs(solution.pred/np.amax(solution.pred) - s/np.amax(s)), axis=-2)
        solution.unc_per_energy = np.amax(np.fabs(solution.pred/np.amax(solution.pred) - s/np.amax(s)), axis=-1)
        solution.obs = s
        solution.weight = w
        solution.cnorm = get_cnorm(solution.pred.flatten(), s.flatten(), w.flatten())
        solution.rspec = 0.0
        solution.cspec = 1.0

        # if Wigner calculation is requested
        solution.wigner = None
        if calculate_wigner:
            solution.wigner = get_wigner(solution.t/fs_per_au, solution.Et)
            solution.wigner.t *= fs_per_au
            solution.wigner.omega = solution.wigner.omega*eV_per_au + self.basis.omega[0]
            if self.additional_basis is not None:
                solution.additional_wigner = get_wigner(solution.t/fs_per_au, solution.additional_Et)
                solution.additional_wigner.t *= fs_per_au
                solution.additional_wigner.omega = solution.additional_wigner.omega*eV_per_au + self.basis.omega[0]

        if reference_spectrum is not None:
            #np.nan_to_num(reference_spectrum, copy=False)
            target_Ew2 = np.fft.fftshift(reference_spectrum)**2
            if self.basis.oversampling > 1:
                target_Ew2 = target_Ew2[::self.basis.oversampling]
            if self.additional_basis is not None:
                pred_Ew2 = np.abs(solution.Ew + solution.additional_Ew)**2
            else:
                pred_Ew2 = np.abs(solution.Ew)**2
            solution.rspec = get_rnorm(pred_Ew2, target_Ew2)
            solution.cspec = get_cnorm(pred_Ew2, target_Ew2)
        logging.info(f"Best Up: {solution.Up:.2f} eV, eTOF sim = {solution.cnorm:.5f}, spectral sim = {solution.cspec:.5f}")
        return solution

