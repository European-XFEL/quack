Installation
============

To perform the reconstructon using this package, install it with the following from the root directory.::

    pip install .

It is recommended to use the EuXFEL python version compiled with BLAS. One can create an environment to that end as follows::


    module load exfel exfel-python
    python -m venv --system-site-packages quack
    source quack/bin/activate
    pip install -U pip
    pip install .
