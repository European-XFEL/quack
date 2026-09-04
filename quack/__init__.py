
import os

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

