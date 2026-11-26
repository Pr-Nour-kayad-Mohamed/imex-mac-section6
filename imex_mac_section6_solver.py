#!/usr/bin/env python3
# ============================================================================
# IMEX-MAC Solver 2D — Suite numerique de la Section 6
# Auteur : NOUR KAYAD, Mohamed — MENFOP, Republique de Djibouti
# ORCID  : 0009-0002-1443-7599
#
# Solveur IMEX-MAC 2D incompressible avec diffusion thermique anisotrope
# implicite, advection MUSCL limitee (minmod / vanleer / none), champ Phi
# a relaxation implicite et positivite thermique controlee (T_min).
#
# Portee : resultats numeriques exploratoires sur cas synthetique
# (Taylor-Green thermique periodic). Ne constitue ni une validation
# industrielle, ni une certification de surete, ni une preuve formelle.
# ============================================================================
import json
import math
import time
from dataclasses import dataclass, asdict
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class SimConfig:
    nx: int = 32
    ny: int = 32
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    dt: float = 1e-3
    t_end: float = 0.05
    rho0: float = 1.0
    cv: float = 1.0
    eta: float = 0.05
    a: float = 1.0          # a > 0 constant (hypothese H-B revisee)
    m_phi: float = 0.2
    k_xx: float = 0.1
    k_yy: float = 0.05
    T_min: float = 1e-6
    limiter: str = "minmod" # "none" | "minmod" | "vanleer"
    seed: int = 42


def limiter_fn(r: np.ndarray, kind: str = "minmod") -> np.ndarray:
    if kind == "none":
        return np.ones_like(r)
    if kind == "minmod":
        return np.maximum(0.0, np.minimum(1.0, r))
    if kind == "vanleer":
        return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-30)
    raise ValueError(f"Limiteur inconnu: {kind}")


def periodic_laplacian_2d(nx: int, ny: int, dx: float, dy: float, coeff: float = 1.0) -> sparse.csr_matrix:
    """Operateur Laplacien centre d'ordre 2 sur grille centree periodique nx x ny."""
    n = nx * ny
    cx = coeff / (dx * dx)
    cy = coeff / (dy * dy)
    diag_main = np.full(n, -2.0 * (cx + cy))

    def idx(i, j):
        return (i % nx) * ny + (j % ny)

    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            p = idx(i, j)
            rows.append(p); cols.append(p); vals.append(diag_main[p])
            for ni, nj, c in [(i + 1, j, cx), (i - 1, j, cx), (i, j + 1, cy), (i, j - 1, cy)]:
                rows.append(p); cols.append(idx(ni, nj)); vals.append(c)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def periodic_anisotropic_diff_2d(nx, ny, dx, dy, k_xx, k_yy) -> sparse.csr_matrix:
    """Operateur div(K grad) centre d'ordre 2 avec K = diag(k_xx, k_yy) constant."""
    n = nx * ny
    cx = k_xx / (dx * dx)
    cy = k_yy / (dy * dy)

    def idx(i, j):
        return (i % nx) * ny + (j % ny)

    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            p = idx(i, j)
            rows.append(p); cols.append(p); vals.append(-2.0 * (cx + cy))
            for ni, nj, c in [(i + 1, j, cx), (i - 1, j, cx), (i, j + 1, cy), (i, j - 1, cy)]:
                rows.append(p); cols.append(idx(ni, nj)); vals.append(c)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def advect_cell_centered_muscl(q, u_face, v_face, dx, dy, limiter="minmod"):
    """(v . grad_h) q aux centres de cellules, conservatif, limiteur de pente, MAC periodique."""
    nx, ny = q.shape
    dqx = np.roll(q, -1, axis=0) - q
    dqx_b = q - np.roll(q, 1, axis=0)
    eps = 1e-14
    rx = np.where(np.abs(dqx) > eps, dqx_b / np.where(np.abs(dqx) > eps, dqx, 1.0), 0.0)
    phix = limiter_fn(rx, limiter)
    q_L_x = np.roll(q, 1, axis=0) + 0.5 * phix * (q - np.roll(q, 1, axis=0))
    q_R_x = q - 0.5 * np.roll(phix, -1, axis=0) * (np.roll(q, -1, axis=0) - q)
    u_pos = np.maximum(u_face, 0.0)
    u_neg = np.minimum(u_face, 0.0)
    flux_x_left = u_pos * q_L_x + u_neg * q_R_x
    flux_x_right = np.roll(flux_x_left, -1, axis=0)

    dqy = np.roll(q, -1, axis=1) - q
    dqy_b = q - np.roll(q, 1, axis=1)
    ry = np.where(np.abs(dqy) > eps, dqy_b / np.where(np.abs(dqy) > eps, dqy, 1.0), 0.0)
    phiy = limiter_fn(ry, limiter)
    q_L_y = np.roll(q, 1, axis=1) + 0.5 * phiy * (q - np.roll(q, 1, axis=1))
    q_R_y = q - 0.5 * np.roll(phiy, -1, axis=1) * (np.roll(q, -1, axis=1) - q)
    v_pos = np.maximum(v_face, 0.0)
    v_neg = np.minimum(v_face, 0.0)
    flux_y_bot = v_pos * q_L_y + v_neg * q_R_y
    flux_y_top = np.roll(flux_y_bot, -1, axis=1)
    return (flux_x_right - flux_x_left) / dx + (flux_y_top - flux_y_bot) / dy


def strain_norm_sq_centered(u_face, v_face, dx, dy):
    """Estime |D(v)|^2 aux centres des cellules a partir des vitesses MAC."""
    du_dx = (np.roll(u_face, -1, axis=0) - u_face) / dx
    dv_dy = (np.roll(v_face, -1, axis=1) - v_face) / dy
    u_c = 0.5 * (u_face + np.roll(u_face, -1, axis=0))
    v_c = 0.5 * (v_face + np.roll(v_face, -1, axis=1))
    du_dy = (np.roll(u_c, -1, axis=1) - np.roll(u_c, 1, axis=1)) / (2.0 * dy)
    dv_dx = (np.roll(v_c, -1, axis=0) - np.roll(v_c, 1, axis=0)) / (2.0 * dx)
    d11 = du_dx; d22 = dv_dy; d12 = 0.5 * (du_dy + dv_dx)
    return d11**2 + d22**2 + 2.0 * (d12**2)


def divergence_mac(u_face, v_face, dx, dy):
    """Divergence discrete aux centres de cellules sur grille MAC periodique."""
    return (np.roll(u_face, -1, axis=0) - u_face) / dx + (np.roll(v_face, -1, axis=1) - v_face) / dy


def total_energy(u_face, v_face, T, phi, cfg, dx, dy):
    """Energie totale discrete E(t) = int [ (rho0/2)|v|^2 + rho0*cv*T + (a/2)*phi^2 ] dx."""
    u_c = 0.5 * (u_face + np.roll(u_face, -1, axis=0))
    v_c = 0.5 * (v_face + np.roll(v_face, -1, axis=1))
    v2_c = u_c**2 + v_c**2
    integrand = 0.5 * cfg.rho0 * v2_c + cfg.rho0 * cfg.cv * T + 0.5 * cfg.a * (phi**2)
    return float(np.sum(integrand) * dx * dy)


class IMEXMacSolver2D:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.dx = cfg.Lx / cfg.nx
        self.dy = cfg.Ly / cfg.ny
        self.n_cells = cfg.nx * cfg.ny
        lap_unpinned = periodic_laplacian_2d(cfg.nx, cfg.ny, self.dx, self.dy, coeff=1.0 / cfg.rho0)
        A_proj = lap_unpinned.tolil()
        A_proj[0, :] = 0.0
        A_proj[0, 0] = 1.0
        self.A_proj = A_proj.tocsr()
        diff_op = periodic_anisotropic_diff_2d(cfg.nx, cfg.ny, self.dx, self.dy, cfg.k_xx, cfg.k_yy)
        mass_T = (cfg.rho0 * cfg.cv / cfg.dt) * sparse.eye(self.n_cells, format="csr")
        self.A_T = (mass_T - diff_op).tocsr()

    def solve_projection(self, u_star, v_star):
        div_star = divergence_mac(u_star, v_star, self.dx, self.dy)
        rhs = (div_star / self.cfg.dt).ravel()
        rhs[0] = 0.0
        phi_p = spsolve(self.A_proj, rhs).reshape((self.cfg.nx, self.cfg.ny))
        phi_p = phi_p - np.mean(phi_p)  # jauge integrale nulle
        dphi_dx_face = (phi_p - np.roll(phi_p, 1, axis=0)) / self.dx
        dphi_dy_face = (phi_p - np.roll(phi_p, 1, axis=1)) / self.dy
        u_new = u_star - (self.cfg.dt / self.cfg.rho0) * dphi_dx_face
        v_new = v_star - (self.cfg.dt / self.cfg.rho0) * dphi_dy_face
        return u_new, v_new, phi_p

    def step(self, u, v, T, phi):
        dt = self.cfg.dt; dx, dy = self.dx, self.dy; nx, ny = self.cfg.nx, self.cfg.ny
        adv_u = advect_cell_centered_muscl(u, u, v, dx, dy, self.cfg.limiter)
        adv_v = advect_cell_centered_muscl(v, u, v, dx, dy, self.cfg.limiter)
        visc_op = periodic_laplacian_2d(nx, ny, dx, dy, coeff=self.cfg.eta)
        A_v = ((self.cfg.rho0 / dt) * sparse.eye(self.n_cells, format="csr") - visc_op).tocsr()
        rhs_u = ((self.cfg.rho0 / dt) * u - self.cfg.rho0 * adv_u).ravel()
        rhs_v = ((self.cfg.rho0 / dt) * v - self.cfg.rho0 * adv_v).ravel()
        u_star = spsolve(A_v, rhs_u).reshape((nx, ny))
        v_star = spsolve(A_v, rhs_v).reshape((nx, ny))
        u_next, v_next, p_corr = self.solve_projection(u_star, v_star)
        adv_phi = advect_cell_centered_muscl(phi, u, v, dx, dy, self.cfg.limiter)
        denom_phi = (1.0 / dt) + self.cfg.m_phi * self.cfg.a
        phi_next = ((phi / dt) - adv_phi) / denom_phi
        adv_T = advect_cell_centered_muscl(T, u, v, dx, dy, self.cfg.limiter)
        diss_visc = 2.0 * self.cfg.eta * strain_norm_sq_centered(u_next, v_next, dx, dy)
        mu_phi_next = self.cfg.a * phi_next
        relax_heat = self.cfg.m_phi * (mu_phi_next**2)
        rhs_T = ((self.cfg.rho0 * self.cfg.cv / dt) * T - self.cfg.rho0 * self.cfg.cv * adv_T
                 + diss_visc + relax_heat).ravel()
        T_next = spsolve(self.A_T, rhs_T).reshape((nx, ny))
        clamped_mask = T_next < self.cfg.T_min
        n_clamped = int(np.sum(clamped_mask))
        if n_clamped > 0:
            T_next = np.maximum(T_next, self.cfg.T_min)
        return u_next, v_next, T_next, phi_next, p_corr, n_clamped

    def run(self, u0, v0, T0, phi0):
        t0_wall = time.perf_counter()
        u, v, T, phi = u0.copy(), v0.copy(), T0.copy(), phi0.copy()
        n_steps = max(1, int(round(self.cfg.t_end / self.cfg.dt)))
        E0 = total_energy(u, v, T, phi, self.cfg, self.dx, self.dy)
        max_energy_rel_drift = 0.0
        max_div = float(np.max(np.abs(divergence_mac(u, v, self.dx, self.dy))))
        min_T_observed = float(np.min(T))
        total_clamped_cells = 0
        for _ in range(n_steps):
            u, v, T, phi, _, n_clamped = self.step(u, v, T, phi)
            total_clamped_cells += n_clamped
            min_T_observed = min(min_T_observed, float(np.min(T)))
            div_now = float(np.max(np.abs(divergence_mac(u, v, self.dx, self.dy))))
            max_div = max(max_div, div_now)
            E_now = total_energy(u, v, T, phi, self.cfg, self.dx, self.dy)
            if abs(E0) > 1e-30:
                max_energy_rel_drift = max(max_energy_rel_drift, abs(E_now - E0) / abs(E0))
        elapsed_s = time.perf_counter() - t0_wall
        return {
            "config": asdict(self.cfg),
            "steps_executed": n_steps,
            "final_time": n_steps * self.cfg.dt,
            "max_div_inf": max_div,
            "E_initial": E0,
            "E_final": total_energy(u, v, T, phi, self.cfg, self.dx, self.dy),
            "max_energy_rel_drift": max_energy_rel_drift,
            "min_T_observed": min_T_observed,
            "total_clamped_cells": total_clamped_cells,
            "elapsed_seconds": elapsed_s,
            "fields": {"u": u, "v": v, "T": T, "phi": phi},
        }


def initial_taylor_green_thermal(cfg):
    dx = cfg.Lx / cfg.nx; dy = cfg.Ly / cfg.ny
    x_u = np.arange(cfg.nx)[:, None] * dx
    y_u = (np.arange(cfg.ny)[None, :] + 0.5) * dy
    u0 = -np.cos(x_u) * np.sin(y_u)
    x_v = (np.arange(cfg.nx)[:, None] + 0.5) * dx
    y_v = np.arange(cfg.ny)[None, :] * dy
    v0 = np.sin(x_v) * np.cos(y_v)
    x_c = (np.arange(cfg.nx)[:, None] + 0.5) * dx
    y_c = (np.arange(cfg.ny)[None, :] + 0.5) * dy
    T0 = 1.0 + 0.25 * np.cos(x_c) * np.cos(y_c)
    phi0 = 0.5 * np.sin(x_c) * np.cos(y_c)
    return u0, v0, T0, phi0


def run_section6_suite() -> dict:
    report = {}
    grids = [16, 24, 32]
    ref_L2_T = []
    for n in grids:
        cfg = SimConfig(nx=n, ny=n, dt=2e-4, t_end=0.01, eta=0.05, k_xx=0.1, k_yy=0.05)
        u0, v0, T0, phi0 = initial_taylor_green_thermal(cfg)
        res = IMEXMacSolver2D(cfg).run(u0, v0, T0, phi0)
        ref_L2_T.append((n, float(np.sqrt(np.mean(res["fields"]["T"] ** 2)))))
    report["mesh_convergence_L2_T"] = ref_L2_T

    cfg_base = SimConfig(nx=32, ny=32, dt=1e-3, t_end=0.05)
    u0, v0, T0, phi0 = initial_taylor_green_thermal(cfg_base)
    res_base = IMEXMacSolver2D(cfg_base).run(u0, v0, T0, phi0)
    report["conservation_and_positivity"] = {k: v for k, v in res_base.items() if k != "fields"}

    peclet_sweep = []
    for k_scale in [1.0, 0.1, 0.01, 0.001]:
        cfg_pe = SimConfig(nx=32, ny=32, dt=1e-3, t_end=0.03,
                           k_xx=0.1 * k_scale, k_yy=0.05 * k_scale, limiter="minmod")
        u0, v0, T0, phi0 = initial_taylor_green_thermal(cfg_pe)
        r = IMEXMacSolver2D(cfg_pe).run(u0, v0, T0, phi0)
        pe_approx = (1.0 * (cfg_pe.Lx / cfg_pe.nx)) / max(cfg_pe.k_xx, 1e-12)
        peclet_sweep.append({"peclet_cell_approx": pe_approx, "min_T": r["min_T_observed"],
                             "clamped_cells": r["total_clamped_cells"],
                             "energy_rel_drift": r["max_energy_rel_drift"]})
    report["peclet_sweep"] = peclet_sweep

    cfl_sweep = []
    for dt_val in [5e-4, 1e-3, 2.5e-3, 5e-3]:
        cfg_cfl = SimConfig(nx=32, ny=32, dt=dt_val, t_end=0.02)
        u0, v0, T0, phi0 = initial_taylor_green_thermal(cfg_cfl)
        r = IMEXMacSolver2D(cfg_cfl).run(u0, v0, T0, phi0)
        cfl_approx = 1.0 * dt_val / (cfg_cfl.Lx / cfg_cfl.nx)
        cfl_sweep.append({"dt": dt_val, "cfl_approx": cfl_approx,
                          "max_div_inf": r["max_div_inf"], "min_T": r["min_T_observed"],
                          "energy_rel_drift": r["max_energy_rel_drift"]})
    report["cfl_sweep"] = cfl_sweep
    return report


if __name__ == "__main__":
    suite_report = run_section6_suite()
    print(json.dumps(suite_report, indent=2))
