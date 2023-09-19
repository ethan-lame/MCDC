import matplotlib.pyplot as plt
import h5py
import numpy as np

from reference import reference


# Load results
with h5py.File("output.h5", "r") as f:
    z = f["tally/grid/z"][:]
    dz = z[1:] - z[:-1]
    z_mid = 0.5 * (z[:-1] + z[1:])

    mu = f["tally/grid/mu"][:]
    dmu = mu[1:] - mu[:-1]
    mu_mid = 0.5 * (mu[:-1] + mu[1:])

    psi = f["tally/flux/mean"][:]
    psi_sd = f["tally/flux/sdev"][:]

    J = f["tally/current/mean"][:, 2]
    J_sd = f["tally/current/sdev"][:, 2]


I = len(z) - 1
N = len(mu) - 1

# Scalar flux
phi = np.zeros(I)
phi_sd = np.zeros(I)
for i in range(I):
    phi[i] += np.sum(psi[i, :])
    phi_sd[i] += np.linalg.norm(psi_sd[i, :])

# Normalize
phi /= dz
phi_sd /= dz
J /= dz
J_sd /= dz
for n in range(N):
    psi[:, n] = psi[:, n] / dz / dmu[n]
    psi_sd[:, n] = psi_sd[:, n] / dz / dmu[n]

# Reference solution
phi_ref, phi_z_ref, J_ref, J_z_ref, psi_ref, psi_z_ref = reference(z, mu)

# Flux - spatial average
plt.plot(z_mid, phi, "-b", label="MC")
plt.fill_between(z_mid, phi - phi_sd, phi + phi_sd, alpha=0.2, color="b")
plt.plot(z_mid, phi_ref, "--r", label="Ref.")
plt.xlabel(r"$z$, cm")
plt.ylabel("Flux")
plt.ylim([0.06, 0.16])
plt.grid()
plt.legend()
plt.title(r"$\bar{\phi}_i$")
plt.show()



# Current - spatial average
plt.plot(z_mid, J, "-b", label="MC")
plt.fill_between(z_mid, J - J_sd, J + J_sd, alpha=0.2, color="b")
plt.plot(z_mid, J_ref, "--r", label="Ref.")
plt.xlabel(r"$z$, cm")
plt.ylabel("Current")
plt.ylim([-0.03, 0.045])
plt.grid()
plt.legend()
plt.title(r"$\bar{J}_i$")
plt.show()


