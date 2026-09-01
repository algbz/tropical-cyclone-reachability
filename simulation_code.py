#!/usr/bin/env python3
"""Reproduce all manuscript figures for the reduced tropical-cyclone steering study.

This is the single public-facing reproduction script for the manuscript
"Physical and Information-Limited Reachability in a Reduced Tropical Cyclone
Steering Model".

Scope
-----
The model is an uncalibrated, nondimensional four-state normal form motivated by
local deformation steering. It is not a forecast model and the bounded input is
not a proposed physical hurricane-modification actuator.

The state is S = [x, y, z1, z2]. The terminal branch is defined by the sign of
x(T). The deterministic model is

    dx/dt = lambda(y) x - kappa x^3 + 0.75 z1 + 0.10 z2
    dy/dt = 0.52 + 0.06 x + 0.20 z2
    dz/dt = -gamma z + B(y) u,

with B(y) = R(theta(y)) diag(1, 0.25), fixed pulse duration 0.45, and
||u|| <= 0.90. All quantities are nondimensional.

Reproduction design
-------------------
A single command

    python simulation_code.py

writes every figure used by the manuscript, including appendix figures, to
``figures/``.

The deterministic reachability figures are recomputed directly from the model.
The high-sample posterior, robustness, and observation-phase calculations are
archived as machine-readable CSV outputs in ``derived/``. Those high-cost
Monte Carlo outputs are plotted directly here so figure regeneration is fast,
deterministic, and does not require a collection of audit scripts. The exact
processed observational tables used by the manuscript figures are archived as
CSV files under ``observational/``.

This layout deliberately separates *figure reproduction* from the internal audit
workflow that was used while developing the paper. The upload package contains
only the material an external reader needs to inspect the model, numerical values,
source observations, and final plots.

Outputs
-------
figures/fig01.png                                    (manuscript Fig. 1)
figures/fig02.png                                    (Fig. 2)
figures/fig03.png                                    (Fig. 3)
figures/fig04.png                                    (Fig. 4)
figures/fig05.png                                    (Fig. 5)
figures/fig06.png                                    (Fig. 6)
figures/fig07.png                                    (Fig. 7)
figures/fig08.png                                    (Fig. 8)
figures/figA1.png                                    (Appendix Fig. A1)
figures/figB1.png                                    (Appendix Fig. B1)

"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "derived"
OBS = ROOT / "observational"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

PURPLE = "#6f4aa2"
TEAL = "#2a9d8f"
GREEN = "#3a8f5d"
GRAY = "#7a7f85"
DARK = "#222222"
LIGHT_PURPLE = "#f2eef8"
RECOVERY_CMAP = LinearSegmentedColormap.from_list(
    "recoverability", [LIGHT_PURPLE, "#bba7d1", PURPLE]
)

T = 12.0
DT = 0.02
H = 0.005
X0 = np.array([0.055, -3.2, 0.0, 0.0], dtype=float)
GAMMA = 0.90
KAPPA = 0.35
PULSE_DURATION = 0.45
U_MAX = 0.90
RELIABILITY = 0.95


def lam(y):
    return 0.15 + 0.85 * np.exp(-(np.asarray(y) / 1.25) ** 2)


def lam_prime(y):
    y = np.asarray(y)
    e = np.exp(-(y / 1.25) ** 2)
    return 0.85 * e * (-2.0 * y / 1.25**2)


def theta(y):
    return 0.50 + 0.90 * np.tanh(np.asarray(y) / 1.8)


def rhs(S, U):
    x, y, z1, z2 = S.T
    th = theta(y)
    c, s = np.cos(th), np.sin(th)
    u1, u2 = U.T
    dz1 = -GAMMA * z1 + c * u1 - 0.25 * s * u2
    dz2 = -GAMMA * z2 + s * u1 + 0.25 * c * u2
    dx = lam(y) * x - KAPPA * x**3 + 0.75 * z1 + 0.10 * z2
    dy = 0.52 + 0.06 * x + 0.20 * z2
    return np.column_stack([dx, dy, dz1, dz2])


def rk2(S, U, h):
    k1 = rhs(S, U)
    return S + h * rhs(S + 0.5 * h * k1, U)


def baseline_trajectory(step=DT):
    n = int(round(T / step))
    traj = np.empty((n + 1, 4), dtype=float)
    S = X0[None, :].copy()
    traj[0] = X0
    zero = np.zeros((1, 2))
    for i in range(n):
        S = rk2(S, zero, step)
        traj[i + 1] = S[0]
    return traj


def _rk2_subset(S, U, h):
    if len(S) == 0 or h <= 0:
        return S
    return rk2(S, U, h)


def pulse_sweep(pulse_times, angles, duration=PULSE_DURATION, magnitude=U_MAX):
    pulse_times = np.asarray(pulse_times, float)
    angles = np.asarray(angles, float)
    nstep = int(round(T / DT))
    starts = np.repeat(np.rint(pulse_times / DT).astype(int) * DT, len(angles))
    ends = starts + duration
    ang = np.tile(angles, len(pulse_times))
    S = np.tile(X0, (len(starts), 1)).astype(float)
    dirs = magnitude * np.column_stack([np.cos(ang), np.sin(ang)])
    zero = np.zeros_like(dirs)
    tol = 1e-13
    for i in range(nstep):
        t, tn = i * DT, (i + 1) * DT
        inactive = (tn <= starts + tol) | (t >= ends - tol)
        active = (t >= starts - tol) & (tn <= ends + tol)
        cross_end = (~inactive) & (~active) & (t < ends - tol) & (tn > ends + tol)
        if np.any(inactive):
            S[inactive] = _rk2_subset(S[inactive], zero[inactive], DT)
        if np.any(active):
            S[active] = _rk2_subset(S[active], dirs[active], DT)
        if np.any(cross_end):
            h1 = ends[cross_end] - t
            for h in np.unique(np.round(h1, 14)):
                ids = np.where(cross_end)[0][np.isclose(h1, h, atol=1e-13, rtol=0.0)]
                S[ids] = _rk2_subset(S[ids], dirs[ids], float(h))
                S[ids] = _rk2_subset(S[ids], zero[ids], float(DT - h))
    return S.reshape(len(pulse_times), len(angles), 4)


def jacobian(state):
    x, y, _, _ = state
    A = np.zeros((4, 4), dtype=float)
    A[0, 0] = lam(y) - 3.0 * KAPPA * x**2
    A[0, 1] = lam_prime(y) * x
    A[0, 2] = 0.75
    A[0, 3] = 0.10
    A[1, 0] = 0.06
    A[1, 3] = 0.20
    A[2, 2] = -GAMMA
    A[3, 3] = -GAMMA
    return A


def input_matrix(y):
    th = float(theta(y))
    c, s = math.cos(th), math.sin(th)
    G = np.zeros((4, 2))
    G[2:, :] = np.array([[c, -0.25 * s], [s, 0.25 * c]])
    return G


def terminal_sensitivity_and_kernel(traj):
    nstep = len(traj) - 1
    ex = np.array([[1.0, 0.0, 0.0, 0.0]])
    P = ex.copy()
    sens = np.empty(nstep + 1)
    lever = np.empty(nstep + 1)
    angle = np.empty(nstep + 1)

    def record(i, Pnow):
        sens[i] = np.linalg.norm(Pnow)
        k = (Pnow @ input_matrix(traj[i, 1]))[0]
        lever[i] = np.linalg.norm(k)
        angle[i] = math.atan2(-k[1], -k[0])

    record(nstep, P)
    for i in range(nstep - 1, -1, -1):
        P = P @ expm(jacobian(traj[i]) * DT)
        record(i, P)
    return sens, lever, np.unwrap(angle)


def refined_physical_boundary(n_angles=288):
    times = np.arange(3.8, 4.401, DT)
    angles = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    out = pulse_sweep(times, angles)
    margin = -out[:, :, 0].min(axis=1)
    cross = np.where((margin[:-1] > 0) & (margin[1:] <= 0))[0]
    if len(cross) != 1:
        raise RuntimeError(f"Expected one crossing, found {len(cross)}")
    j = int(cross[0])
    a, b = times[j], times[j + 1]
    ma, mb = margin[j], margin[j + 1]
    return float(a - ma * (b - a) / (mb - ma))


def exact_saddle_benchmark():
    horizon = 8.0
    tt = np.linspace(0.0, horizon, 401)
    ell, q0, b, umax = 0.50, 0.020, 1.0, 0.270
    D = abs(q0) * np.exp(ell * horizon) * np.ones_like(tt)
    tau = np.minimum(PULSE_DURATION, np.maximum(horizon - tt, 0.0))
    R = abs(b) * umax / ell * np.exp(ell * (horizon - tt)) * (1.0 - np.exp(-ell * tau))
    feasible = R >= D
    lock = tt[np.where(feasible)[0][-1]]
    return tt, R, D, lock


def _last_boundary_interpolation(table):
    rows = table.sort_values("time")
    vals = list(zip(rows.time.to_numpy(), rows.switch_probability.to_numpy()))
    crossings = [i for i in range(len(vals) - 1) if vals[i][1] >= RELIABILITY and vals[i + 1][1] < RELIABILITY]
    if not crossings:
        raise RuntimeError("No 95% crossing in information boundary table")
    i = crossings[-1]
    t1, p1 = vals[i]
    t2, p2 = vals[i + 1]
    return float(t1 + (RELIABILITY - p1) * (t2 - t1) / (p2 - p1))


def figure_1_and_A1():
    traj = baseline_trajectory(DT)
    nstep = len(traj) - 1
    grid = np.linspace(0.0, T, nstep + 1)
    sens, lever, linear_angle = terminal_sensitivity_and_kernel(traj)
    sweep_times = np.arange(0.0, 10.51, 0.10)
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    sweep = pulse_sweep(sweep_times, angles)

    min_x = sweep[:, :, 0].min(axis=1)
    max_x = sweep[:, :, 0].max(axis=1)
    argmin = sweep[:, :, 0].argmin(axis=1)
    best_ang = np.unwrap(angles[argmin])
    tphys = refined_physical_boundary(288)
    interp_sens = np.interp(sweep_times, grid, sens)
    interp_lev = np.interp(sweep_times, grid, lever)

    fig, axs = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    axs[0].fill_between(sweep_times, min_x, max_x, color=TEAL, alpha=0.18, label="terminal x reachable range")
    axs[0].plot(sweep_times, min_x, lw=2.2, color=PURPLE, label="best terminal x")
    axs[0].axhline(0.0, ls="--", lw=1.6, color=GRAY, label="branch boundary")
    axs[0].axvline(tphys, ls=":", lw=1.8, color=DARK, label="refined policy-class lock")
    axs[0].set_ylabel("terminal branch coordinate")
    axs[0].legend(frameon=True, ncol=2)
    axs[0].grid(alpha=0.18)

    axs[1].plot(sweep_times, interp_sens / interp_sens.max(), lw=2.2, color=GREEN, label="branch-state sensitivity")
    axs[1].plot(sweep_times, interp_lev / interp_lev.max(), lw=2.2, color=TEAL, label="branch input leverage")
    axs[1].axvline(tphys, ls=":", lw=1.8, color=DARK)
    axs[1].set_ylabel("normalized magnitude")
    axs[1].legend(frameon=True)
    axs[1].grid(alpha=0.18)

    axs[2].plot(sweep_times, best_ang, lw=2.2, color=PURPLE, label="finite-pulse optimum")
    lin = np.interp(sweep_times, grid, linear_angle)
    offset = 2 * np.pi * np.round((best_ang[0] - lin[0]) / (2 * np.pi))
    axs[2].plot(sweep_times, lin + offset, "--", lw=2.0, color=GRAY, label="linear kernel prediction")
    axs[2].axvline(tphys, ls=":", lw=1.8, color=DARK)
    axs[2].set(xlabel="pulse start time", ylabel="optimal action angle [rad]")
    axs[2].legend(frameon=True)
    axs[2].grid(alpha=0.18)
    fig.suptitle("Deformation-steering normal form: sensitivity, interventionability, and action drift", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "fig01.png", dpi=220)
    plt.close(fig)

    tt, R, D, lock = exact_saddle_benchmark()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(tt, R, lw=2.4, color=TEAL, label="maximum reachable shift R(t)")
    ax.plot(tt, D, "--", lw=2.2, color=PURPLE, label="distance to basin boundary D")
    ax.axvspan(lock, tt[-1], color=PURPLE, alpha=0.12, label="locked")
    ax.axvline(lock, ls=":", lw=1.6, color=GRAY)
    ax.set(xlabel="intervention start time", ylabel="terminal displacement", title="Exact local saddle benchmark")
    ax.legend(frameon=True)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIG / "figA1.png", dpi=220)
    plt.close(fig)


def figure_2_information_boundary():
    curve = pd.read_csv(DERIVED / "information_curve_highres.csv")
    audit = pd.read_csv(DERIVED / "information_boundary_highres.csv")
    tphys = 4.15457130980865
    tinfo = _last_boundary_interpolation(audit)
    latent_table = pd.read_csv(DERIVED / "latent_uncertainty_curve.csv")
    if not np.allclose(latent_table.time.to_numpy(), curve.time.to_numpy(), atol=1e-12, rtol=0.0):
        raise RuntimeError("Latent-uncertainty time grid does not match information curve")
    latent = latent_table.latent_posterior_uncertainty.to_numpy()

    fig, axs = plt.subplots(2, 1, figsize=(8.8, 6.9), sharex=True)
    axs[0].plot(curve.time, curve.switch_probability, marker="o", markersize=3.0, lw=1.25,
                color=PURPLE, label="resolved active-branch propagation")
    for k, row in audit.iterrows():
        axs[0].errorbar([row.time], [row.switch_probability],
                        yerr=[[row.switch_probability-row.wilson_low], [row.wilson_high-row.switch_probability]],
                        fmt="s", capsize=3, color=TEAL,
                        label="independent high-sample audit" if k == audit.index[0] else None)
    axs[0].axhline(RELIABILITY, ls="--", lw=1.2, color=GRAY, label="95% reliability")
    axs[0].axvline(tinfo, ls="-.", lw=1.4, color=PURPLE, label="final information boundary")
    axs[0].axvline(tphys, ls=":", lw=1.6, color=DARK, label="physical policy boundary")
    axs[0].axvspan(tinfo, tphys, color=PURPLE, alpha=0.10)
    for tobs in [3.50, 3.75, 4.00]:
        axs[0].axvline(tobs, lw=0.7, color=GRAY, alpha=0.25)
    axs[0].set_ylabel("switch probability")
    axs[0].set_ylim(0.55, 1.01)
    axs[0].grid(alpha=0.18)
    axs[0].legend(frameon=True, fontsize=8, ncol=2)

    axs[1].plot(curve.time, latent, lw=1.8, color=TEAL, label="latent steering uncertainty")
    axs[1].axvline(tinfo, ls="-.", lw=1.4, color=PURPLE)
    axs[1].axvline(tphys, ls=":", lw=1.6, color=DARK)
    axs[1].axvspan(tinfo, tphys, color=PURPLE, alpha=0.10)
    for tobs in [3.50, 3.75, 4.00]:
        axs[1].axvline(tobs, lw=0.7, color=GRAY, alpha=0.25)
    axs[1].set_xlabel("intervention start time")
    axs[1].set_ylabel("latent posterior uncertainty")
    axs[1].grid(alpha=0.18)
    axs[1].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig02.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_3_branch_split():
    summary = pd.read_csv(DERIVED / "branch_split_exact_cadence.csv")
    histogram = pd.read_csv(DERIVED / "branch_split_histogram.csv")
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.45), sharey=True)
    for ax, (_, row) in zip(axs, summary.iterrows()):
        t = float(row.time)
        h = histogram.loc[np.isclose(histogram.time, t)].copy()
        ax.bar(h.bin_left, h.density, width=h.bin_right-h.bin_left, align="edge",
               color=PURPLE, alpha=0.55, label="nonlinear posterior")
        xx = np.linspace(-1.15, 1.15, 500)
        ax.plot(xx, norm.pdf(xx, loc=row.local_gaussian_mean, scale=row.local_gaussian_sd),
                "--", lw=1.8, color=TEAL, label="local Gaussian")
        ax.axvline(0.0, ls=":", lw=1.3, color=GRAY)
        ax.set_title(f"t={t:.2f}: P={row.nonlinear_switch_probability:.2f}, Gaussian={row.local_gaussian_switch_probability:.2f}")
        ax.set_xlabel("terminal branch coordinate")
        ax.grid(alpha=0.14)
    axs[0].set_ylabel("density")
    axs[0].legend(frameon=True, fontsize=8)
    fig.suptitle("Finite uncertainty near the terminal branch boundary produces branch-split outcomes")
    fig.tight_layout()
    fig.savefig(FIG / "fig03.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_4_uncertainty_scaling():
    df = pd.read_csv(DERIVED / "small_uncertainty_gap_scaling_highres.csv")
    grouped = df.groupby("epsilon").gap.agg(["mean", "std"]).reset_index()
    xfit = grouped.loc[grouped.epsilon <= 0.20, "epsilon"].to_numpy()
    yfit = grouped.loc[grouped.epsilon <= 0.20, "mean"].to_numpy()
    slope = float(np.dot(xfit, yfit) / np.dot(xfit, xfit))
    pred = slope * xfit
    r2 = float(1.0 - np.sum((yfit - pred) ** 2) / np.sum((yfit - yfit.mean()) ** 2))
    theory = 0.3756622281

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.errorbar(grouped.epsilon, grouped["mean"], yerr=grouped["std"], fmt="o", capsize=4,
                color=PURPLE, label="nonlinear repeated-seed audit")
    xx = np.linspace(0, 0.31, 180)
    ax.plot(xx, theory * xx, ls="--", lw=1.8, color=TEAL, label=rf"local theory: {theory:.3f}$\epsilon$")
    ax.plot(xx, slope * xx, lw=1.8, color=GREEN, label=rf"fit for $\epsilon\leq0.20$: {slope:.3f}$\epsilon$")
    ax.axvspan(0.20, 0.31, color=GRAY, alpha=0.06, label="outside fitted small-noise range")
    ax.set_xlabel(r"posterior uncertainty amplitude $\epsilon$")
    ax.set_ylabel(r"information-to-physical gap $\Delta t$")
    ax.grid(alpha=0.18)
    ax.legend(frameon=True, fontsize=8)
    ax.text(0.02, 0.93, rf"$R^2={r2:.4f}$", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig04.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_5_robustness():
    struct = pd.read_csv(DERIVED / "structural_robustness.csv")
    dur = pd.read_csv(DERIVED / "pulse_duration_robustness.csv")
    mean = pd.read_csv(DERIVED / "posterior_mean_offset_robustness.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.0))

    piv = struct.pivot(index="deformation_multiplier", columns="coupling_multiplier", values="posterior_switch_probability")
    im = axs[0].imshow(piv.values, origin="lower", aspect="auto", vmin=min(0.74, piv.values.min()), vmax=0.96, cmap=RECOVERY_CMAP)
    axs[0].set_xticks(range(len(piv.columns)), [f"{x:.1f}" for x in piv.columns])
    axs[0].set_yticks(range(len(piv.index)), [f"{x:.1f}" for x in piv.index])
    axs[0].set_xlabel("latent coupling multiplier")
    axs[0].set_ylabel("deformation multiplier")
    axs[0].set_title(r"(a) Reliability 0.20 before $t_{phys}$")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axs[0].text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, label="best resolved switch probability")

    axs[1].plot(dur.duration, dur.gap, marker="o", lw=1.8, color=PURPLE)
    axs[1].set_xlabel("pulse duration")
    axs[1].set_ylabel(r"$t_{phys}-t_{info}$")
    axs[1].set_title("(b) Policy-duration robustness")
    axs[1].grid(alpha=0.18)

    axs[2].plot(mean.projected_mean_bias_sigma, mean.switch_probability, marker="o", lw=1.8, color=TEAL)
    axs[2].axhline(RELIABILITY, ls="--", lw=1, color=GRAY)
    axs[2].set_xlabel(r"posterior-mean bias [$\sigma$]")
    axs[2].set_ylabel("best resolved switch probability at t=4.00")
    axs[2].set_title("(c) Mean-offset robustness")
    axs[2].grid(alpha=0.18)

    fig.tight_layout()
    fig.savefig(FIG / "fig05.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_6_phase():
    df = pd.read_csv(DERIVED / "observation_phase_sweep.csv")
    phases = df.observation_phase.to_numpy()
    gaps = df.gap.to_numpy()
    nominal = float(gaps[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(phases, gaps, marker="o", lw=1.8, color=PURPLE, label="phase-shift audit")
    ax.axhline(nominal, ls="--", lw=1.3, color=TEAL, label="phase 0")
    ax.fill_between(phases, gaps.min(), gaps.max(), color=LIGHT_PURPLE, alpha=0.35, zorder=0)
    ax.set_xlabel("observation-grid phase")
    ax.set_ylabel(r"information-to-physical gap $\Delta t$")
    ax.set_xlim(-0.005, 0.23)
    ax.grid(alpha=0.18)
    ax.legend(frameon=True, fontsize=8)
    ax.text(0.02, 0.96, f"range: {gaps.min():.2f}--{gaps.max():.2f}", transform=ax.transAxes, va="top", color=DARK)
    fig.tight_layout()
    fig.savefig(FIG / "fig06.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_7_sampling_null():
    null = pd.read_csv(DERIVED / "dorian_sampling_geometry_null.csv")
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    ax.plot(null.strength_multiplier, null.false_saddle_fraction, marker="o", lw=1.8, color=PURPLE,
            label=r"$P(\widehat{\det J}<0)$")
    ax.plot(null.strength_multiplier, null.observed_or_more_negative_fraction, marker="s", lw=1.8, color=TEAL,
            label=r"$P(\widehat{\det J}\leq\det J_{obs})$")
    ax.axvline(1.0, ls=":", lw=1.3, color=GRAY, label="matched gradient strength")
    ax.set_xlabel("positive-det null gradient strength / observed strength")
    ax.set_ylabel("simulation fraction")
    ax.set_title("Dorian sampling-geometry stress test")
    ax.grid(alpha=0.18)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig07.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_8_dorian():
    aff = pd.read_csv(OBS / "dorian_giv_affine_steering.csv")
    rob = pd.read_csv(OBS / "dorian_steering_robustness_summary.csv")
    grid = pd.read_csv(OBS / "dorian_steering_robustness_grid.csv")
    snd = pd.read_csv(OBS / "dorian_giv_layermean_soundings.csv")

    d = aff.merge(rob[["mission", "negative_fraction"]], on="mission", how="left").sort_values("midtime")
    x = np.arange(len(d))
    labels = [m[4:8] for m in d.mission]

    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.5))
    axs[0, 0].plot(x, d.track_speed_ms, marker="o", lw=1.7, color=PURPLE)
    axs[0, 0].set_xticks(x, labels, rotation=45)
    axs[0, 0].set_ylabel(r"translation speed (m s$^{-1}$)")
    axs[0, 0].set_title("(a) Dorian translation during surveillance")
    axs[0, 0].grid(alpha=0.18)

    axs[0, 1].plot(x, d.negative_fraction, marker="o", lw=1.7, color=TEAL)
    axs[0, 1].axhline(0.5, ls="--", lw=1, color=GRAY, alpha=0.65)
    axs[0, 1].set_xticks(x, labels, rotation=45)
    axs[0, 1].set_ylabel(r"fraction with $\det J<0$")
    axs[0, 1].set_ylim(-0.03, 1.03)
    axs[0, 1].set_title("(b) Layer/radius robustness")
    axs[0, 1].grid(alpha=0.18)

    g = grid[(grid.mission == "20190902N1") & (grid.bottom == 850) & (grid.top == 300) & grid.outer.isin([700.0, 900.0])]
    for outer, gg in g.groupby("outer"):
        gg = gg.sort_values("inner")
        axs[1, 0].plot(gg.inner, gg.detJ, marker="o", lw=1.7,
                       color=PURPLE if int(outer) == 700 else GREEN,
                       label=f"outer {int(outer)} km")
    axs[1, 0].axhline(0, ls="--", lw=1, color=GRAY)
    axs[1, 0].set_xlabel("inner radius (km)")
    axs[1, 0].set_ylabel(r"$\det J$")
    axs[1, 0].set_title("(c) Sep 2 near-storm 850--300 hPa")
    axs[1, 0].grid(alpha=0.18)
    axs[1, 0].legend(frameon=True)

    ss = snd[(snd.mission == "20190902N1") & (snd.r_km >= 300) & (snd.r_km <= 900)].copy()
    X = np.c_[np.ones(len(ss)), ss.x_km.to_numpy()/1000.0, ss.y_km.to_numpy()/1000.0]
    cu = np.linalg.lstsq(X, ss.u850_300.to_numpy(), rcond=None)[0]
    cv = np.linalg.lstsq(X, ss.v850_300.to_numpy(), rcond=None)[0]
    lim = 950
    gx = np.linspace(-lim, lim, 15)
    gy = np.linspace(-lim, lim, 15)
    XX, YY = np.meshgrid(gx, gy)
    UU = cu[0] + cu[1]*(XX/1000) + cu[2]*(YY/1000)
    VV = cv[0] + cv[1]*(XX/1000) + cv[2]*(YY/1000)
    axs[1, 1].streamplot(gx, gy, UU, VV, density=0.8, linewidth=0.8, arrowsize=0.7, color=GRAY)
    axs[1, 1].quiver(ss.x_km, ss.y_km, ss.u850_300, ss.v850_300,
                     angles="xy", scale_units="xy", scale=0.06, width=0.004,
                     color=TEAL, label="layer-mean winds")
    axs[1, 1].scatter([0], [0], marker="*", s=90, color=PURPLE, label="storm center")
    axs[1, 1].set_xlim(-lim, lim)
    axs[1, 1].set_ylim(-lim, lim)
    axs[1, 1].set_aspect("equal")
    axs[1, 1].set_xlabel("storm-relative x (km)")
    axs[1, 1].set_ylabel("storm-relative y (km)")
    axs[1, 1].set_title("(d) Sep 2 sampled affine field")
    axs[1, 1].legend(frameon=True, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG / "fig08.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _short_date(mission):
    digits = "".join(ch for ch in str(mission) if ch.isdigit())[:8]
    dt = datetime.strptime(digits, "%Y%m%d")
    return f"{dt.strftime('%b')} {dt.day}"


def figure_B1_positive_control():
    ds = pd.read_csv(OBS / "dorian_steering_robustness_summary.csv")
    da = pd.read_csv(OBS / "dorian_giv_affine_steering.csv")
    js = pd.read_csv(OBS / "joaquin_steering_robustness_summary.csv")
    jb = pd.read_csv(OBS / "joaquin_steering_base_bootstrap.csv")
    d = ds.merge(da[["mission", "track_speed_ms"]], on="mission", how="left")
    j = js.merge(jb[["mission", "track_speed_ms"]], on="mission", how="left")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(d.track_speed_ms, d.negative_fraction, s=55, color=PURPLE, label="Dorian 2019")
    ax.scatter(j.track_speed_ms, j.negative_fraction, s=70, marker="^", color=TEAL, label="Joaquin 2015")
    ax.axhline(0.5, ls="--", lw=1, color=GRAY, alpha=0.65)

    for _, r in d[d.mission.isin(["20190902N1", "20190903N1"])].iterrows():
        ax.annotate(_short_date(r.mission), (r.track_speed_ms, r.negative_fraction),
                    xytext=(5, 5), textcoords="offset points", fontsize=7.5)
    for _, r in j.iterrows():
        ax.annotate(_short_date(r.mission), (r.track_speed_ms, r.negative_fraction),
                    xytext=(5, -12), textcoords="offset points", fontsize=7.5)

    ax.set_xlabel(r"observed translation speed (m s$^{-1}$)")
    ax.set_ylabel(r"fraction of reductions with $\det J<0$")
    ax.set_ylim(-0.03, 1.04)
    ax.grid(alpha=0.18)
    ax.legend(frameon=True)
    ax.set_title("Observed affine-geometry robustness: Dorian and Joaquin")
    fig.tight_layout()
    fig.savefig(FIG / "figB1.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def required_inputs():
    return [
        DERIVED / "information_curve_highres.csv",
        DERIVED / "information_boundary_highres.csv",
        DERIVED / "branch_split_exact_cadence.csv",
        DERIVED / "branch_split_histogram.csv",
        DERIVED / "small_uncertainty_gap_scaling_highres.csv",
        DERIVED / "structural_robustness.csv",
        DERIVED / "pulse_duration_robustness.csv",
        DERIVED / "posterior_mean_offset_robustness.csv",
        DERIVED / "dorian_sampling_geometry_null.csv",
        DERIVED / "observation_phase_sweep.csv",
        DERIVED / "latent_uncertainty_curve.csv",
        OBS / "dorian_giv_affine_steering.csv",
        OBS / "dorian_steering_robustness_summary.csv",
        OBS / "dorian_steering_robustness_grid.csv",
        OBS / "dorian_giv_layermean_soundings.csv",
        OBS / "joaquin_steering_robustness_summary.csv",
        OBS / "joaquin_steering_base_bootstrap.csv",
    ]


def main():
    parser = argparse.ArgumentParser(description="Regenerate every manuscript figure.")
    parser.add_argument("--check", action="store_true", help="only verify that all required inputs are present")
    args = parser.parse_args()

    missing = [p for p in required_inputs() if not p.exists()]
    if missing:
        raise SystemExit("Missing required input files:\n" + "\n".join(str(p) for p in missing))
    if args.check:
        print(f"All {len(required_inputs())} required input files are present.")
        return

    jobs = [
        ("Fig. 1 + A1", figure_1_and_A1),
        ("Fig. 2", figure_2_information_boundary),
        ("Fig. 3", figure_3_branch_split),
        ("Fig. 4", figure_4_uncertainty_scaling),
        ("Fig. 5", figure_5_robustness),
        ("Fig. 6", figure_6_phase),
        ("Fig. 7", figure_7_sampling_null),
        ("Fig. 8", figure_8_dorian),
        ("Fig. B1", figure_B1_positive_control),
    ]
    for label, func in jobs:
        print(f"Generating {label} ...", flush=True)
        func()

    expected = [
        "fig01.png", "fig02.png", "fig03.png", "fig04.png", "fig05.png",
        "fig06.png", "fig07.png", "fig08.png", "figA1.png", "figB1.png",
    ]
    absent = [name for name in expected if not (FIG / name).exists()]
    if absent:
        raise SystemExit("Figure generation incomplete: " + ", ".join(absent))
    print(f"Done. Generated all {len(expected)} manuscript figure files in {FIG}")


if __name__ == "__main__":
    main()
