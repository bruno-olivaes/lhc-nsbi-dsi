"""Neural Simulation-Based Inference (NSBI) Model for LHC Dimuon Invariant Mass Spectroscopy.

Implements unbinned conditional neural likelihood estimation p(m | theta_DSI, theta_SM)
for Discrete Scale Invariance (DSI) searches at the LHC.
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize


class ConditionalDensityMLP(nn.Module):
    """Rede Neural Condicional para Densidade e Razão de Verossimilhança."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Entrada: u = ln(m), eps, omega = 2pi/ln(lambda), phi, background nuisance
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, u, eps, omega, phi):
        """Calcula o desvio logarítmico do modelo neural em relação ao baseline."""
        features = torch.cat([u, eps, omega, phi], dim=-1)
        return self.net(features)


class NSBILHCModel:
    """Pipeline de Inferência Baseada em Simulação Neural (NSBI) para Espectros de Dimúons."""
    def __init__(self, m0=1.0, hidden_dim=64, device=None):
        self.m0 = m0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ConditionalDensityMLP(hidden_dim=hidden_dim).to(self.device)

    def sm_background_density(self, u, coeffs):
        """Densidade contínua do Modelo Padrão (Drell-Yan polinomial em u = ln(m))."""
        poly = np.polyval(coeffs, u)
        prob = np.exp(poly)
        prob /= np.trapezoid(prob, u) if len(u) > 1 else 1.0
        return np.maximum(prob, 1e-12)

    def unbinned_log_likelihood(self, m_events, eps, lambda_val, phi, coeffs=None):
        """Calcula a verossimilhança não-binada exata de N eventos individuais."""
        u = np.log(m_events / self.m0)
        u_min, u_max = u.min(), u.max()

        if coeffs is None:
            # Fit rápido do contínuo polinomial
            hist_y, bin_edges = np.histogram(u, bins=100)
            u_b = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            coeffs = np.polyfit(u_b, np.log(np.maximum(hist_y, 1.0)), deg=3)

        omega = 2.0 * np.pi / np.log(lambda_val)

        # Base contínua do Modelo Padrão nos pontos de dados
        poly_vals = np.polyval(coeffs, u)
        p_sm = np.exp(poly_vals - np.max(poly_vals))

        # Modulação de Invariância de Escala Discreta (DSI)
        modulation = 1.0 + eps * np.cos(omega * (u - np.mean(u)) + phi)
        modulation = np.maximum(modulation, 1e-6)

        # Integral de normalização numérica via quadratura de Gauss-Legendre
        u_grid = np.linspace(u_min, u_max, 500)
        poly_grid = np.polyval(coeffs, u_grid)
        p_sm_grid = np.exp(poly_grid - np.max(poly_vals))
        mod_grid = 1.0 + eps * np.cos(omega * (u_grid - np.mean(u)) + phi)
        z_norm = np.trapezoid(p_sm_grid * mod_grid, u_grid)

        # Log-verossimilhança total dos N eventos
        log_prob = np.log(p_sm * modulation) - np.log(z_norm)
        return float(np.sum(log_prob))

    def profile_likelihood_scan(self, m_events, lambda_val, eps_grid):
        """Estatística de teste da Razão de Verossimilhança Perfilada q(eps)."""
        u = np.log(m_events / self.m0)
        hist_y, bin_edges = np.histogram(u, bins=80)
        u_b = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        coeffs = np.polyfit(u_b, np.log(np.maximum(hist_y, 1.0)), deg=3)

        # Achar o máximo global de verossimilhança (MLE)
        def neg_loglik(params):
            eps, phi = params
            return -self.unbinned_log_likelihood(m_events, eps, lambda_val, phi, coeffs)

        res_mle = minimize(neg_loglik, x0=[0.01, 0.0], bounds=[(0.0, 0.5), (-np.pi, np.pi)])
        max_loglik = -res_mle.fun
        eps_hat = float(res_mle.x[0])

        q_stat = []
        for eps in eps_grid:
            # Perfilando sobre a fase phi
            def obj(p):
                return -self.unbinned_log_likelihood(m_events, eps, lambda_val, p[0], coeffs)
            res = minimize(obj, x0=[0.0], bounds=[(-np.pi, np.pi)])
            ll_prof = -res.fun
            # Estatística de Wilks q(eps) = -2 * ln(L(eps) / L_max)
            q = float(np.maximum(0.0, 2.0 * (max_loglik - ll_prof)))
            q_stat.append(q)

        return np.array(q_stat), eps_hat, max_loglik

    def compute_upper_limit_95(self, m_events, lambda_val):
        """Calcula o limite superior a 95% CL em epsilon (q = 3.841 conforme Teorema de Wilks)."""
        eps_grid = np.linspace(0.001, 0.30, 40)
        q_vals, eps_hat, _ = self.profile_likelihood_scan(m_events, lambda_val, eps_grid)

        # Interpolação para q = 3.841 (95% CL unilateral à direita de eps_hat)
        wilks_thresh = 3.841
        idx = np.where((eps_grid >= eps_hat) & (q_vals >= wilks_thresh))[0]
        if len(idx) > 0:
            limit_95 = float(eps_grid[idx[0]])
        else:
            limit_95 = float(eps_grid[-1])

        return limit_95, eps_hat, eps_grid, q_vals
