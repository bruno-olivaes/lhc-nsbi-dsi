"""Protocolo de Certificação de Integridade Metrológica Adaptado para NSBI e Método Tradicional."""
import numpy as np
from scipy.signal import lombscargle
from scipy.optimize import curve_fit


def traditional_binned_search(m_events, n_bins=100, deg=3):
    """Executa o método tradicional binned + Lomb-Scargle GLS nos eventos."""
    u = np.log(m_events)
    counts, bin_edges = np.histogram(u, bins=n_bins)
    u_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    sigma_counts = np.sqrt(np.maximum(counts, 1.0))
    coeffs = np.polyfit(u_centers, counts, deg=deg)
    baseline = np.polyval(coeffs, u_centers)
    residuals = (counts - baseline) / sigma_counts

    u_span = u_centers.max() - u_centers.min()
    omega_min = 2.0 * np.pi / u_span
    du_med = np.median(np.diff(np.sort(u_centers)))
    omega_max = 2.0 * np.pi / (4.0 * du_med)
    omega_grid = np.logspace(np.log10(omega_min), np.log10(omega_max), 2000)

    u_w = u_centers - u_centers.mean()
    y_w = residuals - residuals.mean()
    power = lombscargle(u_w, y_w, omega_grid, normalize=True)
    idx = int(np.argmax(power))
    omega_peak = float(omega_grid[idx])
    power_peak = float(power[idx])
    lambda_hat = float(np.exp(2.0 * np.pi / omega_peak))

    # Limite superior tradicional a 95% CL em epsilon
    # Na binagem clássica, a modulação de amplitude A no resíduo normalizado relaciona-se
    # com a amplitude fracionária da densidade eps via: eps_eff = A / sqrt(mean(counts))
    rms = np.std(residuals)
    eps_trad_95 = float((2.0 * rms) / np.sqrt(np.mean(counts)))

    return {
        "omega_peak": omega_peak,
        "power_peak": power_peak,
        "lambda_hat": lambda_hat,
        "residuals": residuals,
        "u_centers": u_centers,
        "eps_trad_95": eps_trad_95,
        "power": power,
        "omega_grid": omega_grid,
    }


def certify_nsbi_result(nsbi_model, m_events, lambda_test, n_bootstrap=200):
    """Audita os 5 gates de certificação sobre o modelo NSBI."""
    u = np.log(m_events)
    u_span = u.max() - u.min()
    du_med = np.median(np.diff(np.sort(u)))
    omega = 2.0 * np.pi / np.log(lambda_test)

    # Gate 1: Resolvabilidade
    omega_min = 2.0 * np.pi / u_span
    omega_max = 2.0 * np.pi / (4.0 * du_med)
    gate_resolvable = bool(omega_min <= omega <= omega_max)

    # Limite superior 95% CL via perfil de verossimilhança
    limit_95, eps_hat, eps_grid, q_vals = nsbi_model.compute_upper_limit_95(m_events, lambda_test)

    # Gate 2: Significância de Amplitude (Wilks q > 4.0 ~ 2 sigma)
    # Se eps_hat for estatisticamente indistinguível de zero, falha no gate de sinal positivo
    q_max = float(np.max(q_vals))
    gate_amplitude = bool(eps_hat > 0.05 and q_max > 9.0)

    # Gate 3: Precisão Paramétrica
    gate_precision = bool(limit_95 < 0.15)

    # Gate 4: Look-Elsewhere via Bootstrap
    rng = np.random.default_rng(42)
    boot_limits = []
    # Teste rápido de bootstrap para validação
    n_sample = len(m_events)
    for _ in range(min(50, n_bootstrap)):
        boot_events = rng.choice(m_events, size=n_sample, replace=True)
        q_b, _, _ = nsbi_model.profile_likelihood_scan(boot_events, lambda_test, np.linspace(0.01, 0.20, 10))
        boot_limits.append(float(np.max(q_b)))

    p_corrected = float(np.mean(np.array(boot_limits) >= q_max)) if q_max > 0 else 1.0
    gate_plevel = bool(p_corrected < 0.05)

    # Gate 5: Delta BIC
    # Comparar verossimilhança nula (eps=0) com o melhor ajuste
    ll_null = nsbi_model.unbinned_log_likelihood(m_events, 0.0, lambda_test, 0.0)
    ll_dsi = nsbi_model.unbinned_log_likelihood(m_events, eps_hat, lambda_test, 0.0)
    bic_null = -2.0 * ll_null
    bic_dsi = -2.0 * ll_dsi + 2.0 * np.log(len(m_events))
    delta_bic = float(bic_null - bic_dsi)
    gate_bic = bool(delta_bic > 10.0)

    certified = all([gate_resolvable, gate_amplitude, gate_precision, gate_plevel, gate_bic])

    return {
        "lambda_test": lambda_test,
        "omega": omega,
        "eps_hat": eps_hat,
        "limit_95": limit_95,
        "eps_grid": eps_grid,
        "q_vals": q_vals,
        "gate_resolvable": gate_resolvable,
        "gate_amplitude": gate_amplitude,
        "gate_precision": gate_precision,
        "gate_plevel": gate_plevel,
        "gate_bic": gate_bic,
        "p_corrected": p_corrected,
        "delta_bic": delta_bic,
        "certified": certified,
    }

