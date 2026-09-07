Usage
=====


1. Simulate the basis functions with `QuackSim`_, which can be done in Julia (see below).

2. Use the produced H5 file to fit the solution for each individual frame::

    import h5py
    import numpy as np
    from quack.amplitude_fit import AmplitudeSolver, Basis


    # define photo-electron kinetic energy axis
    W = np.linspace(85, 170, 170-85+1)

    # define the angle axis
    theta_deg = np.arange(0, 360.0, step=360/22.5)[:16]
    theta = theta_deg*np.pi/180.0

    # read basis from QuackSim
    with h5py.File("basis.h5", "r") as fid:
        # W and theta are optional
        # they can be used to reduce the observational range
        # in the eTOFs
        basis = Basis.from_hdf5(fid, W, theta)
    
    # input the basis simulated with QuackSim here
    solver = AmplitudeSolver(basis)

    # these are optional parameters:
    spectrum_mode = -1       # if set to 1, use the spectrum input in the fit (default: -1, meaning that the spectrum information is not used)
    spectrum = None          # if spectrum_mode == 1, this is used to fit the FEL spectrum
    spectrum_axis = None     # axis in eV corresponding to the spectrum values
    calculate_wigner = False # if True, return the Wigner distribution in `solution`

    with h5py.File("input_data.h5", "r") as fid:
        img = fid["observation"][train_id, pulse_id, :, :]
        solution = solver.solve(img,
                                # optional parameters
                                #spectrum_mode=spectrum_mode,
                                #spectrum=spectrum,
                                #spectrum_axis=spectrum,
                                #calculate_wigner=calculate_wigner,
                                )

The ``solution`` variable is of type ``QUACKAmpSolution`` and it contains the following:

+-------------+---------------------------------------------------------------------------------------+
| Variable    | Description                                                                           |
+=============+=======================================================================================+
| ``Ew``      | Complex-valued electric field coefficients in the FEL energy domain.                  |
+-------------+---------------------------------------------------------------------------------------+
| ``omega``   | FEL energy axis in eV.                                                                |
+-------------+---------------------------------------------------------------------------------------+
| ``Et``      | Complex-valued electric field in the time domain.                                     |
+-------------+---------------------------------------------------------------------------------------+
| ``t``       | FEL time axis in fs.                                                                  |
+-------------+---------------------------------------------------------------------------------------+
| ``pred``    | Prediction.                                                                           |
+-------------+---------------------------------------------------------------------------------------+
| ``obs``     | Observation after interpolation to the basis energy axis.                             |
+-------------+---------------------------------------------------------------------------------------+
| ``cnorm``   | Overlap between observation and prediction.                                           |
+-------------+---------------------------------------------------------------------------------------+
| ``cspec``   | Overlap between FEL spectrum and predicted FEL spectrum                               |
+-------------+---------------------------------------------------------------------------------------+
| ``Up``      | Ponderomotive potential in eV.                                                        |
+-------------+---------------------------------------------------------------------------------------+

If the parameter ``calculate_wigner`` was set to ``True``, it also returns the Wigner distribution in the
field ``wigner``.
Given the solution, if the 
The ``WignerDistribution`` object ``wigner`` contains the following variables:

+---------------------+-----------------------------------------------------------+
| Variable            | Description                                               |
+=====================+===========================================================+
| ``W``               | 2D representation of the Wigner distribution.             |
+---------------------+-----------------------------------------------------------+
| ``t``               | Time axis in fs.                                          |
+---------------------+-----------------------------------------------------------+
| ``omega``           | Energy axis in eV.                                        |
+---------------------+-----------------------------------------------------------+
| ``time_spectrum``   | Projection of the Wigner distribution in the time axis.   |
+---------------------+-----------------------------------------------------------+
| ``energy_spectrum`` | Projection of the Wigner distribution in the energy axis. |
+---------------------+-----------------------------------------------------------+

If the Wigner distribution was not originally calculated, it can always be calculated from ``Et`` using the following snippet::

    from quack.wigner import get_wigner
    from quack.constants import fs_per_au, eV_per_au
    # the time input is expected in atomic units
    W = get_wigner(solution.t/fs_per_au, solution.Et)
    # the result may be converted back to fs and eV
    # omega0 is the center energy of the basis and can be taken from np.mean(basis.energy_axis)
    W.t = W.t*fs_per_au
    W.omega = W.omega*eV_per_au + omega0

.. _QuackSim: https://github.com/European-XFEL/QuackSim

