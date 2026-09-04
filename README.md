# QUACK

This repository contains software used to reconstruct information about FEL time and energy after angular streaking.
Being able to perform this reconstruction depends on the simulation of the photo-electron propagation from a FEL basis set,
which is done in the [QuackSim](https://github.com/European-XFEL/quacksim) Julia package.

Check instructions in [QuackSim](https://github.com/European-XFEL/quacksim) for how to simulate it in details, or see the quick start introduction
below for a summary.

## Installation of QUACK

To perform the reconstructon using this package, install it with the following from the root directory.
```
pip install .
```

It is recommended to use the EuXFEL python version compiled with BLAS. One can create an environment to that end as follows:
```
module load exfel exfel-python
python -m venv --system-site-packages quack
source quack/bin/activate
pip install -U pip
pip install .
```

## Usage

1. Simulate the basis functions with `QuackSim` (see below).

2. Use the produced H5 file to fit the solution for each individual frame:

```
import h5py
import numpy as np
from quack.amplitude_fit import AmplitudeSolver

# define photo-electron kinetic energy axis
W = np.linspace(85, 170, 170-85+1)

# define the angle axis
theta_deg = np.arange(0, 360.0, step=360/22.5)[:16]
theta = theta_deg*np.pi/180.0

# read basis from QuackSim
with h5py.File("basis_Ne1s_hor.h5", "r") as fid:
    basis_horizontal = Basis.from_hdf5(fid, W, theta)

# input the basis simulated with QuackSim here
solver = AmplitudeSolver(basis_horizontal)

# how to combine FEL spectral information
# use a negative number to ignore the spectral information
# use 1 for an L2 norm with weight kappa on the spectrum
spectrum_mode = 1
constrain = "everywhere"

# reciprocal weight of the spectral constraint
# use a high number to ignore the spectrum information with spectrum_mode=1
# use a low number to rely mostly on the spectral information
kappa = 200

with h5py.File("input_data.h5", "r") as fid:
    # Read the angular streaking information.
    # Its shape should correspond to (photo-electron energy bins, angular bins)
    # and it should match the binning provided in the basis H5 file above.
    img = fid["observation"][train_id, pulse_id, :, :]
    norm_img = img/np.amax(img)

    # If available, obtain the FEL *photon* energy spectrum information.
    # Its energy axis in `spectrum_axis` is used to interpolate
    # from the input to the binning used in the basis.
    spectrum = fid["fel_energy"][train_id, pulse_id, :, :]
    spectrum_axis = fid["fel_energy_axis"][train_id, pulse_id, :, :]

    # Obtain the reconstructed result.
    solution = solver.solve(norm_img,
                            spectrum=spectrum,
                            spectrum_axis=spectrum,
                            spectrum_mode=spectrum_mode,
                            kappa=kappa,
                            constrain=constrain,
                            )
```

The `spectrum_mode` argument may be one of the following.

Mode | Description
---|---
-1 | Do not constrain the FEL spectrum anywhere.
1  | Use the L2 norm to constrain the FEL spectrum to the `spectrum` argument.
2  | Constrain the FEL spectrum to zero everywhere, ignoring `spectrum`.

The option `constrain` establishes how missing data (`np.nan` or regions not covered in the spectrum)
are handled if the option `spectrum_mode=1` is used. The following options are available:

`constrain` | Description
---|---
`everywhere` | Constrain missing regions to zero intensity.
`partial` | Leave the missing regions free to vary.

It is recommended to use `spectrum_mode=1` and `constrain="everywhere"` if the FEL spectrum is available.
If the FEL spectrum is not available, it is recommended to use `spectrum_mode=-1`. In both cases,
it is highly advisable to deconvolve the input data to remove effects of the instrumental
response function.

The `solution` variable is of type `QUACKAmpSolution` and it contains the following:

Variable | Description
---|---
`Ew`|Complex-valued electric field coefficients in the FEL energy domain.
`Et`|Complex-valued electric field in the time domain.
`additional_Ew`|Complex-valued electric field coefficients in the FEL energy domain for the additional basis if provided.
`additional_Et`|Complex-valued electric field in the time domain for the additional basis if provided.
`omega`|FEL energy axis in eV.
`t`|FEL time axis in fs.
`pred`|Prediction.
`unc`|Maximum difference between prediction and observed data.
`wigner`|Object of type `WignerDistribution` containing the Wigner quasi-probability function.
`additional_wigner`|Object of type `WignerDistribution` containing the Wigner quasi-probability function for the additional basis if provided.
`Up`|Ponderomotive potential in eV.
`raw`|Raw value of the coefficients without support in the FEL energy.
`additional_raw`|Raw value of the coefficients without support in the FEL energy for the additional basis if provided.

The `WignerDistribution` object `wigner` and `additional_wigner` contains the following variables:

Variable | Description
---|---
`W`|2D representation of the Wigner distribution.
`t`|Time axis in fs.
`omega`|Energy axis in eV.
`time_spectrum`|PRojection of the Wigner distribution in the time axis.
`energy_spectrum`|Projection of the Wigner distribution in the energy axis.


## Usage of an additional basis

If an additional basis is used, the fit can be done considering two FEL pulses with different polarizations.
That is the basis used can be simulated with QuackSim using a horizontal polarization, and another basis is simulated with a circular polarization.
With that unsatz, two electric fields are provided as an output.
In this case the output variables named `additional_*` contain the solution corresponding to this additional basis.
An example of this use-case can be seen below:

```
import h5py
import numpy as np
from quack.amplitude_fit import AmplitudeSolver

# define photo-electron kinetic energy axis
W = np.linspace(85, 170, 170-85+1)

# define the angle axis
theta_deg = np.arange(0, 360.0, step=360/22.5)[:16]
theta = theta_deg*np.pi/180.0

# horizontal FEL polarization
with h5py.File("basis_Ne1s_hor.h5", "r") as fid:
    basis_horizontal = Basis.from_hdf5(fid, W, theta)

# circular FEL polarization
with h5py.File("basis_Ne1s_circ.h5", "r") as fid:
    basis_circular = Basis.from_hdf5(fid, W, theta)

# input the basis simulated with QuackSim here
solver = AmplitudeSolver(basis_horizontal, additional_basis=basis_circular)

with h5py.File("input_data.h5", "r") as fid:
    img = fid["observation"][train_id, pulse_id, :, :]

    spectrum = fid["fel_energy"][train_id, pulse_id, :, :]
    spectrum_axis = fid["fel_energy_axis"][train_id, pulse_id, :, :]

    # Obtain the reconstructed result.
    solution = solver.solve(img,
                            spectrum=spectrum,
                            spectrum_axis=spectrum,
                            spectrum_mode=1
                            )
```

## Production of the basis set with QUACKSim

A basis set simulation may be done using Julia or in Python.
If using Python, install QuackSim as follows:

```
pip install https://github.com/European-XFEL/quacksim
```

The simulation is done in Julia and hence when the software is used for the first time, Julia is installed, if it is not present,
with all dependencies needed.
If using Maxwell, the Julia installation in Maxwell can be used.

To actually use the software, look into `notebooks/Producing_simulation_basis.ipynb` (in Python) or
`notebooks/Producing_simulation_basis_Julia.ipynb` (in Julia) inside the QuackSim package.

The output file should be provided to Quack.

### Note on the Julia installation

One may install `julia` with `juliaup`, or if in Maxwell in the EuXFEL/DESY, one may set up a globally installed version with:
```
module load exfel julia
```

When opening the basis generation notebook, the Julia installation in MAxwell can be used by running the following lines
in the notebook before any other step:

```
import envmodules
envmodules.load('exfel', 'julia')
```

Julia version greater or equal to 1.10 is recommended.

