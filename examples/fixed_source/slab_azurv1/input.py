import numpy as np

import sys

N_hist = int(sys.argv[1])
dijit  = eval(sys.argv[2])
output = sys.argv[3]

# Disable Numba-JIT for pure Python mode
from numba import config
config.DISABLE_JIT = dijit

# Get path to mcdc (not necessary if mcdc is installed)
import sys
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set model
# =============================================================================

# Set materials
m = mcdc.material(capture=np.array([1.0/3.0]), scatter=np.array([[1.0/3.0]]),
                  fission=np.array([1.0/3.0]), nu_p=np.array([2.3]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=-1E10, bc="reflective")
s2 = mcdc.surface('plane-x', x=1E10,  bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(point=[0.0,0.0,0.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
mcdc.tally(scores=['flux'], x=[-20.5, 20.5, 201], t=[0.0, 20.0, 20])

# Setting
mcdc.setting(N_hist=N_hist, time_boundary=20.0, progress_bar=False, output=output)

# Run
mcdc.run()