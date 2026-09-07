
import os

# The Julia solvers scan the Up values on Julia threads. juliacall starts Julia with one thread
# unless told otherwise, and with several threads it needs to own the signal handlers or the
# process segfaults. Both can still be overridden from the environment.
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

try:
    from juliacall import Main as jl
    mydir = os.path.dirname(__file__)
    jl.include(f"{mydir}/Quack.jl")
    jlpkgname = 'Quack'
    globals()[jlpkgname] = jl.seval(jlpkgname)
    for _name in jl.seval('string.(names(' + jlpkgname + '))'):
        globals()[_name.replace('!', '_b')] = jl.seval(jlpkgname + '.' + _name)
except ModuleNotFoundError:
    print("Julia support not available. Install `juliacall` and Julia to use it.")

