"""Differentiable 1-compartment oral PK simulator (Euler method).

ODE system
----------
    dA_gut /dt  = -ka * A_gut
    dA_cent/dt  =  ka * A_gut  -  (CL / V) * A_cent

Concentration = A_cent / V

Uses fixed-step explicit Euler so the entire forward pass stays on the
PyTorch computation graph and gradients flow through the PK parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor


def simulate(
    times_hr: Tensor,
    dose_mg: Tensor,
    CL: Tensor,
    V: Tensor,
    ka: Tensor,
    *,
    n_euler_steps: int = 500,
) -> Tensor:
    """Simulate a 1-compartment oral PK profile.

    All scalar inputs should be 0-dim or 1-dim tensors on the same device.

    Parameters
    ----------
    times_hr : Tensor, shape (T,)
        Observation time points (hours).
    dose_mg : Tensor, scalar
        Total oral dose in mg.
    CL : Tensor, scalar
        Clearance (L/h).
    V : Tensor, scalar
        Volume of distribution (L).
    ka : Tensor, scalar
        First-order absorption rate constant (1/h).
    n_euler_steps : int
        Number of Euler integration steps over [0, t_max].

    Returns
    -------
    Tensor, shape (T,)
        Predicted concentration (mg/L) at each observation time.
    """
    t_max = times_hr[-1]
    dt = t_max / n_euler_steps

    ke = CL / V  # elimination rate constant

    # Initial state: entire dose in gut, nothing in central compartment
    A_gut  = dose_mg.clone()
    A_cent = torch.zeros_like(dose_mg)

    # Pre-compute the fine Euler time grid
    t_euler = torch.linspace(0.0, t_max.item(), n_euler_steps + 1, device=times_hr.device)

    # Store state at each Euler knot so we can interpolate later
    A_cent_traj = torch.zeros(n_euler_steps + 1, device=times_hr.device)
    A_cent_traj[0] = A_cent

    for i in range(n_euler_steps):
        dA_gut  = -ka * A_gut
        dA_cent =  ka * A_gut - ke * A_cent
        A_gut   = A_gut  + dA_gut  * dt
        A_cent  = A_cent + dA_cent * dt
        A_cent_traj[i + 1] = A_cent

    # Concentration at Euler knots
    conc_euler = A_cent_traj / V

    # Linear interpolation to the actual observation times
    conc_obs = _interp(t_euler, conc_euler, times_hr)

    # Clamp any numerical noise below zero
    return conc_obs.clamp(min=0.0)


def _interp(x: Tensor, y: Tensor, xnew: Tensor) -> Tensor:
    """Piecewise-linear interpolation (differentiable)."""
    # searchsorted gives the index of the right boundary
    idx = torch.searchsorted(x, xnew).clamp(1, len(x) - 1)
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    slope = (y1 - y0) / (x1 - x0 + 1e-12)
    return y0 + slope * (xnew - x0)
