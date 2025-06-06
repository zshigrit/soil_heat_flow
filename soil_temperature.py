import numpy as np
from tridiagonal_solver import tridiagonal_solver
from phase_change import phase_change


def soil_temperature(physcon, soilvar, tsurf, dt):
    """Solve for soil temperature using an implicit formulation."""
    nsoi = soilvar['nsoi']
    tsoi0 = soilvar['tsoi'].copy()

    tk_plus_onehalf = np.zeros(nsoi - 1)
    for i in range(nsoi - 1):
        numerator = soilvar['tk'][i] * soilvar['tk'][i + 1] * (soilvar['z'][i] - soilvar['z'][i + 1])
        denominator = (
            soilvar['tk'][i] * (soilvar['z_plus_onehalf'][i] - soilvar['z'][i + 1])
            + soilvar['tk'][i + 1] * (soilvar['z'][i] - soilvar['z_plus_onehalf'][i])
        )
        tk_plus_onehalf[i] = numerator / denominator

    a = np.zeros(nsoi)
    b = np.zeros(nsoi)
    c = np.zeros(nsoi)
    d = np.zeros(nsoi)

    i = 0
    m = soilvar['cv'][i] * soilvar['dz'][i] / dt
    a[i] = 0.0
    c[i] = -tk_plus_onehalf[i] / soilvar['dz_plus_onehalf'][i]
    b[i] = m - c[i] + soilvar['tk'][i] / (0.0 - soilvar['z'][i])
    d[i] = m * soilvar['tsoi'][i] + soilvar['tk'][i] / (0.0 - soilvar['z'][i]) * tsurf

    for i in range(1, nsoi - 1):
        m = soilvar['cv'][i] * soilvar['dz'][i] / dt
        a[i] = -tk_plus_onehalf[i - 1] / soilvar['dz_plus_onehalf'][i - 1]
        c[i] = -tk_plus_onehalf[i] / soilvar['dz_plus_onehalf'][i]
        b[i] = m - a[i] - c[i]
        d[i] = m * soilvar['tsoi'][i]

    i = nsoi - 1
    m = soilvar['cv'][i] * soilvar['dz'][i] / dt
    a[i] = -tk_plus_onehalf[i - 1] / soilvar['dz_plus_onehalf'][i - 1]
    c[i] = 0.0
    b[i] = m - a[i]
    d[i] = m * soilvar['tsoi'][i]

    soilvar['tsoi'] = tridiagonal_solver(a, b, c, d)

    soilvar['gsoi'] = soilvar['tk'][0] * (tsurf - soilvar['tsoi'][0]) / (0.0 - soilvar['z'][0])

    if soilvar['method'] == 'apparent-heat-capacity':
        soilvar['hfsoi'] = 0.0
    elif soilvar['method'] == 'excess-heat':
        soilvar = phase_change(physcon, soilvar, dt)

    edif = np.sum(soilvar['cv'] * soilvar['dz'] * (soilvar['tsoi'] - tsoi0) / dt)
    err = edif - soilvar['gsoi'] - soilvar.get('hfsoi', 0.0)
    if abs(err) > 1e-3:
        raise RuntimeError('Soil temperature energy conservation error')

    return soilvar
