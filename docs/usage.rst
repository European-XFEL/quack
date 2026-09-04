Usage
=====


1. Simulate the basis functions with `QuackSim`_, which can be done in Julia (see below).

2. Use the produced H5 file to fit the solution for each individual frame::

    import h5py
    import numpy as np
    from quack.amplitude_fit import AmplitudeSolver

    # input the basis simulated wth QuackSim here
    solver = Solver("basis.h5")

    with h5py.File("input_data.h5", "r") as fid:
        img = fid["observation"][train_id, pulse_id, :, :]
        norm_img = img/np.amax(img)
        solution = solver.solve(norm_img)

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
| ``unc``     | Maximum difference between prediction and observed data.                              |
+-------------+---------------------------------------------------------------------------------------+
| ``wigner``  | Object of type `WignerDistribution` containing the Wigner quasi-probability function. |
+-------------+---------------------------------------------------------------------------------------+
| ``Up``      | Ponderomotive potential in eV.                                                        |
+-------------+---------------------------------------------------------------------------------------+
| ``raw``     | Raw value of the coefficients without support in the FEL energy.                      |
+-------------+---------------------------------------------------------------------------------------+

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

.. _QuackSim: https://git.xfel.eu/machineLearning/quack-sim

