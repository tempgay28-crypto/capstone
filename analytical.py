"""IS 456:2000 flexural capacity of corroded RC beams.

Same formulas as "Parameters (input, output & condition).xlsx", plus the IS 456
Annex G limit for over-reinforced sections, which the spreadsheet omits.
Corrosion is modelled only as loss of steel area: Ast_res = Ast * (1 - corr/100).
"""
import numpy as np

# xu,max / d from IS 456 Cl. 38.1 (depends on steel grade)
_XU_MAX_RATIO = {250: 0.53, 415: 0.48, 500: 0.46}


def xu_max_ratio(fy):
    """Limiting neutral-axis depth ratio; interpolates between the tabulated grades."""
    grades = sorted(_XU_MAX_RATIO)
    return np.interp(fy, grades, [_XU_MAX_RATIO[g] for g in grades])


def steel_area(bar_dia_mm, n_bars):
    return n_bars * np.pi * bar_dia_mm ** 2 / 4.0


def effective_depth(depth_mm, cover_mm, bar_dia_mm):
    return depth_mm - cover_mm - bar_dia_mm / 2.0


def moment_capacity_knm(b, d, ast, fck, fy):
    """Ultimate moment of resistance Mu (kN·m), singly reinforced rectangular section.

    Mu = 0.87 fy Ast d (1 - Ast fy / (b d fck)), capped at Mu,lim when the
    section is over-reinforced (xu > xu,max).
    """
    b, d, ast, fck, fy = (np.asarray(x, dtype=float) for x in (b, d, ast, fck, fy))
    mu = 0.87 * fy * ast * d * (1.0 - (ast * fy) / (b * d * fck))
    k = xu_max_ratio(fy)
    mu_lim = 0.36 * k * (1.0 - 0.42 * k) * fck * b * d ** 2
    xu = 0.87 * fy * ast / (0.36 * fck * b)
    mu = np.where(xu > k * d, mu_lim, mu)
    return mu / 1e6


def point_load_kn(mu_knm, span_mm):
    """Central point load that produces Mu on a simply supported span: P = 4 Mu / L."""
    return 4.0 * mu_knm / (np.asarray(span_mm, dtype=float) / 1000.0)


def corroded_capacity(b, depth, cover, bar_dia, n_bars, fck, corr_pct, span, fy):
    """Return dict with uncorroded/corroded Mu, Pu and residual strength %."""
    d = effective_depth(depth, cover, bar_dia)
    ast0 = steel_area(bar_dia, n_bars)
    ast_res = ast0 * (1.0 - np.asarray(corr_pct, dtype=float) / 100.0)
    mu0 = moment_capacity_knm(b, d, ast0, fck, fy)
    mu_corr = moment_capacity_knm(b, d, ast_res, fck, fy)
    return {
        "d_eff_mm": d,
        "ast0_mm2": ast0,
        "ast_res_mm2": ast_res,
        "mu0_knm": mu0,
        "mu_corr_knm": mu_corr,
        "pu0_kn": point_load_kn(mu0, span),
        "pu_corr_kn": point_load_kn(mu_corr, span),
        "residual_pct": 100.0 * mu_corr / mu0,
    }
