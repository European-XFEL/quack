
from typing import Optional, List, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.signal import fftconvolve
from scipy.interpolate import PchipInterpolator

def central_quantile(x, y, q=0.67):
    """
    From a pdf given in y, over x, find the width of the central x range corresponding to the
    probability mass given by q around the maximum.

    Args:
      x: Consecutive array of x values.
      y: Probability values.
      q: Confidence level.

    Returns: The x width.
    """
    # no negative probabilities
    y_in = np.clip(y, a_min=0, a_max=None)
    # roll data to center maximum
    x_mid = len(x)//2
    x_max = np.argmax(y_in, axis=-1, keepdims=True)
    r = x_mid - x_max
    rows, column_indices = np.ogrid[:y_in.shape[0], :y_in.shape[1]]
    r %= y_in.shape[1]
    column_indices = column_indices - r
    yr = y_in[rows, column_indices]
    # get quantiles relative to the maximum
    def get_q(yy, qq):
        tmp = np.cumsum(yy, axis=-1)
        tmp -= tmp[0]
        tmp /= tmp[-1]
        return np.where(tmp > qq)[0][0]
    # this uses the maximum as a reference instead
    def w(yy):
        iM = np.argmax(yy, axis=-1)
        return x[iM+get_q(yy[iM:], qq=q)] - x[get_q(yy[:iM], qq=1-q)]
    return np.apply_along_axis(w, axis=-1, arr=yr)

def central_quantile_median(x, y, q=0.67):
    """
    From a pdf given in y, over x, find the width of the central x range corresponding to the
    probability mass given by q around the median.

    Args:
      x: Consecutive array of x values.
      y: Probability values.
      q: Confidence level.

    Returns: The x width.
    """
    # no negative probabilities
    y_in = np.clip(y, a_min=0, a_max=None)
    # roll data to center maximum
    x_mid = len(x)//2
    x_max = np.argmax(y_in, axis=-1, keepdims=True)
    r = x_mid - x_max
    rows, column_indices = np.ogrid[:y_in.shape[0], :y_in.shape[1]]
    r %= y_in.shape[1]
    column_indices = column_indices - r
    yr = y_in[rows, column_indices]
    # this would find quantiles relative to the mean, but the median is not often a good indicator
    iy = np.cumsum(yr, axis=-1)/np.sum(yr, axis=-1, keepdims=True)
    def w(yy):
        return x[np.where(yy > 0.5+q/2)[0][0]] - x[np.where(yy > 0.5 - q/2)[0][0]]
    return np.apply_along_axis(w, axis=-1, arr=iy)

def fw(x, y, f=0.5, oversample: int=1):
    """
    Full width at fraction f of the maximum.

    Args:
      x: x axis.
      y: y axis.
      f: Fracton of the maximum.
      oversample: HOw many times to oversample plot, to mitigate binning effect.

    Returns: Full width at fraction f of the maximum.
    """
    if np.any(np.isnan(y)):
        return 0

    if oversample > 1:
        xx = np.linspace(x[0], x[-1], int(len(x)*oversample))
        yy = PchipInterpolator(x, y)(xx)
    else:
        xx = x
        yy = y

    dx = xx[1] - xx[0]
    norm = np.sum(yy*dx)
    if norm <= 0:
        return 0
    y_norm = yy/norm
    x_mid = len(xx)//2
    x_max = np.argmax(y_norm)
    m = x_mid - x_max
    yr = np.roll(y_norm, m)
    a = f*np.amax(yr)
    dx = np.where(yr >= a)[0]
    k1 = dx[-1]
    k2 = dx[0]
    dx = xx[k1] - xx[k2]
    if k1 == k2:
        dx += xx[1] - xx[0]
    return dx

def plot_data(img: np.ndarray, energy_axis: np.ndarray, theta_axis: np.ndarray, vmin: Optional[float]=None, vmax: Optional[float]=None) -> plt.Figure:
    """
    Plot data in the observed space.

    Args:
      img: (NW, Ntheta) array containing intensity for the energy axis
           in the first dimension and angle axis in the second dimension.
      energy_axis: (NW,)-shaped array containing the photo-electron energy axis bins in eV.
      theta_axis: (Ntheta,)-shaped array containing the azimuthal
                   angle bins in degrees.

    Returns: The Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    a = ax.imshow(img.T,
                  extent=(energy_axis[0], energy_axis[-1], -0.5, len(theta_axis)-0.5),
                  interpolation='none',
                  origin='lower',
                  aspect='auto',
                  vmin=vmin, vmax=vmax)
    fig.colorbar(a, orientation="horizontal")
    ax.set(xlabel="Photo-electron energy [eV]",
           ylabel="Angle [deg]",
           title="",
           #xticks=range(len(energy_axis)),
           yticks=range(len(theta_axis)),
           #xticklabels=energy_axis,
           yticklabels=theta_axis,
          )
    return fig

def plot_field(t: List[np.ndarray], E: List[np.ndarray], label: List[str]=[""],
               ls: Optional[List[str]]=None, alpha: Optional[List[float]]=None) -> plt.Figure:
    """
    Plot electric field in the SVAE.

    Args:
      t: (N,)-shaped array containing the time axis in fs.
      E: (N,)-shaped complex array containing the FEL electric field.

    Returns: The Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 12), nrows=2)
    if isinstance(E, np.ndarray):
        E = [E]
    if isinstance(t, np.ndarray):
        t = [t]
    if ls is None:
        ls = ['-' for _ in E]
    if alpha is None:
        alpha = [1.0 for _ in E]
    for t_, E_, label_, ls_, alpha_ in zip(t, E, label, ls, alpha):
        dt = t_[1] - t_[0]
        a = np.abs(E_)**2
        ax[0].plot(t_, a/np.sum(a*dt), lw=3, label=label_, ls=ls_, alpha=alpha_)
        ax[1].plot(t_, np.angle(E_), lw=3, label=label_, ls=ls_, alpha=alpha_)
    ax[0].legend(frameon=False, loc='upper right', bbox_to_anchor=(0.97, 0.97))
    #ax[1].legend()
    ax[0].set(ylabel="Electric field magnitude squared [a.u.]",
              xticks=[])
    ax[1].set(ylabel="Electric field phase [rd]",
              xlabel="Time [fs]")
    fig.subplots_adjust(hspace=0)
    return fig

def plot_spec(fel_time_axis: np.ndarray, fel_energy_axis: np.ndarray, spec: np.ndarray,
              sim: Optional[List[Tuple[float, float]]]=None, vmin=None, vmax=None, with_cbar=True) -> plt.Figure:
    """
    Plot reconstructed spectrogram.

    Args:
      fel_time_axis: (Nkappa,)-shaped array containing the FEL energy axis bins in eV.
      fel_energy_axis: (Nomega,)-shaped array containing the FEL time bins in fs.
      spec: (Nkappa, Nomega) array containing intensity for the FEL time axis
           in the first dimension and FEL energy axis in the second dimension.
      sim: List containing tuples with time and energy of simulated pulses if provided.

    Returns: The Figure.
    """
    if vmin is not None:
        kwargs = {'vmin': vmin, 'vmax': vmax}
    fig, ax = plt.subplots(figsize=(10, 8))
    a = ax.imshow(spec,
                  extent=(fel_time_axis[0], fel_time_axis[-1], fel_energy_axis[0], fel_energy_axis[-1]),
                  interpolation='none',
                  origin='lower',
                  aspect='auto', **kwargs)
    if with_cbar:
        cbar = fig.colorbar(a, orientation="vertical")
        cbar.set_label('W(t, E) [a.u.]', rotation=90)
        if vmin is not None:
            a.set_clim(vmin, vmax)
    ax.set(xlabel="Pulse time [fs]",
           ylabel="FEL energy [eV]",
           title="",
           #xticks=range(len(fel_time_axis)),
           #yticks=range(len(fel_energy_axis)),
           #xticklabels=fel_time_axis,
           #yticklabels=fel_energy_axis,
          )
    if sim is not None:
        for t, e in sim:
            if t > 0 and e > 0:
                ax.add_patch(Ellipse((t, e),
                                      width=0.5, height=2.0, angle=0,
                                      lw=2, facecolor="none", edgecolor='r'))
    return fig

def plot_time(fel_time_axis: Union[List[np.ndarray], np.ndarray],
              spectrum: Union[List[np.ndarray], np.ndarray],
              label: Optional[Union[List[str], str]]=None,
              sim: Optional[List[float]]=None) -> plt.Figure:
    """
    Plot time marginal.

    Args:
      fel_time_axis: (Nkappa,)-shaped array containing the FEL time axis bins in fs.
      spectrum: (Nkappa,)-shaped array, or list of arrays containing intensity for the FEL time axis.
      labels: Labels to show in legend.
      sim: List containing simulated times if provided.

    Returns: The Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    if isinstance(spectrum, np.ndarray):
        spectrum = [spectrum]
    if isinstance(fel_time_axis, np.ndarray):
        fel_time_axis = [fel_time_axs]
    if isinstance(label, str):
        label = [label]
    if label is None:
        label = [None for _ in spectrum]
    for l, x, s in zip(label, fel_time_axis, spectrum):
        dx = x[1] - x[0]
        norm = np.sum(s*dx)
        plt.plot(x, s/norm, lw=2, label=l)
    ax.set(xlabel="Pulse time [fs]",
           ylabel="Probability",
           title="")
    #fig.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.92, 0.9))
    if sim is not None:
        for t in sim:
            if t > fel_time_axis[0]:
                ax.axvline(t, lw=2, ls='--', c="violet")
    return fig


def plot_energy(fel_energy_axis: Union[List[np.ndarray], np.ndarray],
                spectrum: Union[List[np.ndarray], np.ndarray],
                label: Optional[Union[List[str], str]]=None,
                sim: Optional[List[float]]=None) -> plt.Figure:
    """
    Plot energy marginal.

    Args:
      fel_energy_axis: (Nomega,)-shaped array containing the FEL energy axis bins in eV.
      spectrum: (Nomega,)-shaped array, or list of arrays containing intensity for the FEL energy axis.
      labels: Labels to show in legend.
      sim: List containing simulated times if provided.

    Returns: The Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    if isinstance(spectrum, np.ndarray):
        spectrum = [spectrum]
    if isinstance(fel_energy_axis, np.ndarray):
        fel_energy_axis = [fel_energy_axis]
    if isinstance(label, str):
        label = [label]
    if label is None:
        label = [None for _ in spectrum]
    for l, x, s in zip(label, fel_energy_axis, spectrum):
        dx = x[1] - x[0]
        norm = np.sum(s*dx)
        plt.plot(x, s/norm, lw=2, label=l)
    ax.set(xlabel="FEL energy [eV]",
           ylabel="Probability",
           title="")
    #fig.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.92, 0.9))
    if sim is not None:
        for t in sim:
            if t > fel_energy_axis[0]:
                ax.axvline(t, lw=2, ls='--', c="violet")
    return fig

def pes_simulation(energy: np.ndarray, spectrum: np.ndarray,
                   fwhm: float=1.6, adu_amplitude: float=160, sigma: float=0.1,
                   BE: float=870.2+99.0, PL: float=14860.0) -> Dict[str, np.ndarray]:
    """
    Simulate the eTOF resolution, Gaussian noise and Poisson noise.

    Args:
      energy: Photon energy axis in eV.
      spectrum: The spectrum as a 2D array of (energy, angle).
      fwhm: The resolution of the eTOF in eV, FWHM.
      adu_amplitude: Maximum amplitude in the eTOF in ADU (affects Poisson noise).
      sigma: Noise scale in ADU.
      BE: Binding energy of the PES gas in eV.
      PL: Time-per-energy squared.

    Returns: A dictionary containing the keys:
             "tof", with the time-of-flght axis in ns,
             "trace", with the PES trace in ADU,
             and "spectrum", with the reconstructed energy spectrum, in ADU.
    """
    rng = np.random.default_rng(seed=0)

    ns_per_sample = 0.5

    n_energy = energy.shape[0]
    n_spectra = spectrum.shape[1]
    # E = BE + PL/t^2
    # t = sqrt(PL/(E - BE))
    max_tof = np.sqrt(PL/(np.amin(energy) - BE)) + 1.0 # ns
    n_tof = int(max_tof/ns_per_sample)

    # get instrument function
    if fwhm > 0:
        h = np.exp(-0.5*(energy - np.mean(energy))**2/((fwhm/2.355)**2))
        h /= np.sum(h)
        # reduce resolution
        s_r = fftconvolve(spectrum, np.broadcast_to(h[:,None], (n_energy, n_spectra)), mode='same', axes=0)
    else:
        s_r = spectrum

    # create tof axis in ns
    tof = np.arange(n_tof)*ns_per_sample
    # energy in eV for each tof entry
    selected_energy = BE + PL/(tof**2)
    # interpolate values to get points at the tof entries
    s_i = np.stack([np.interp(selected_energy, energy, s_r[:,k], left=0.0, right=0.0)
                    for k in range(s_r.shape[1])], axis=1)

    # add noise
    s_n = s_i
    if adu_amplitude > 0:
        s_n = rng.poisson(np.clip(adu_amplitude*s_n/np.amax(s_n), a_min=0, a_max=None)).astype(np.float32)
    if sigma > 0:
        s_n += rng.normal(loc=0.0, scale=sigma, size=s_i.shape)

    # interpolate back
    s_e = np.stack([np.interp(energy, selected_energy[::-1], s_n[::-1,k], left=0.0, right=0.0)
                    for k in range(s_n.shape[1])], axis=1)

    # output dictionary
    return dict(tof=tof,
                trace=s_n,
                spectrum=s_e)

