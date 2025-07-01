import numpy as np


def phase_change(physcon, soilvar, dt):
    """Adjust temperatures for phase change using excess heat."""
    soilvar['hfsoi'] = 0.0
    nsoi = soilvar['nsoi']

    for i in range(nsoi):
        wliq0 = soilvar['h2osoi_liq'][i]
        wice0 = soilvar['h2osoi_ice'][i]
        wmass0 = wliq0 + wice0
        tsoi0 = soilvar['tsoi'][i]

        imelt = 0
        if soilvar['h2osoi_ice'][i] > 0 and soilvar['tsoi'][i] > physcon.tfrz:
            imelt = 1
            soilvar['tsoi'][i] = physcon.tfrz
        if soilvar['h2osoi_liq'][i] > 0 and soilvar['tsoi'][i] < physcon.tfrz:
            imelt = 2
            soilvar['tsoi'][i] = physcon.tfrz

        if imelt > 0:
            heat_flux_pot = (soilvar['tsoi'][i] - tsoi0) * soilvar['cv'][i] * soilvar['dz'][i] / dt
        else:
            heat_flux_pot = 0.0

        if imelt == 1:
            heat_flux_max = -soilvar['h2osoi_ice'][i] * physcon.hfus / dt
        elif imelt == 2:
            heat_flux_max = soilvar['h2osoi_liq'][i] * physcon.hfus / dt

        if imelt > 0:
            ice_flux = heat_flux_pot / physcon.hfus
            soilvar['h2osoi_ice'][i] = wice0 + ice_flux * dt
            soilvar['h2osoi_ice'][i] = max(0.0, soilvar['h2osoi_ice'][i])
            soilvar['h2osoi_ice'][i] = min(wmass0, soilvar['h2osoi_ice'][i])
            soilvar['h2osoi_liq'][i] = max(0.0, (wmass0 - soilvar['h2osoi_ice'][i]))
            heat_flux = physcon.hfus * (soilvar['h2osoi_ice'][i] - wice0) / dt
            soilvar['hfsoi'] += heat_flux
            residual = heat_flux_pot - heat_flux
            soilvar['tsoi'][i] = soilvar['tsoi'][i] - residual * dt / (soilvar['cv'][i] * soilvar['dz'][i])

            if abs(heat_flux) > abs(heat_flux_max) + 1e-12:
                raise RuntimeError('Soil temperature energy conservation error: phase change')

            if imelt == 2:
                constraint = min(heat_flux_pot, heat_flux_max)
                err = heat_flux - constraint
                if abs(err) > 1e-3:
                    raise RuntimeError('Soil temperature energy conservation error: freezing energy flux')
                err = (soilvar['h2osoi_ice'][i] - wice0) - constraint / physcon.hfus * dt
                if abs(err) > 1e-3:
                    raise RuntimeError('Soil temperature energy conservation error: freezing ice flux')

            if imelt == 1:
                constraint = max(heat_flux_pot, heat_flux_max)
                err = heat_flux - constraint
                if abs(err) > 1e-3:
                    raise RuntimeError('Soil temperature energy conservation error: thawing energy flux')
                err = (soilvar['h2osoi_ice'][i] - wice0) - constraint / physcon.hfus * dt
                if abs(err) > 1e-3:
                    raise RuntimeError('Soil temperature energy conservation error: thawing ice flux')

    return soilvar
