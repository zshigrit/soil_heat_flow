# Python translation of MATLAB soil heat flow model
# Implements soil thermal properties, soil temperature updates, and phase change
# from MATLAB scripts: soil_thermal_properties.m, soil_temperature.m, phase_change.m,
# tridiagonal_solver.m, and sp_05_02.m combined into one file.

from dataclasses import dataclass, field
from typing import List
import math

@dataclass
class Physcon:
    tfrz: float      # Freezing point of water (K)
    cwat: float      # Specific heat of water (J/kg/K)
    cice: float      # Specific heat of ice (J/kg/K)
    rhowat: float    # Density of water (kg/m3)
    rhoice: float    # Density of ice (kg/m3)
    cvwat: float     # Heat capacity of water (J/m3/K)
    cvice: float     # Heat capacity of ice (J/m3/K)
    tkwat: float     # Thermal conductivity of water (W/m/K)
    tkice: float     # Thermal conductivity of ice (W/m/K)
    hfus: float      # Heat of fusion for water at 0 C (J/kg)

@dataclass
class SoilVar:
    # Soil texture lists are indexed by texture class (1..11 in original code)
    silt: List[float]
    sand: List[float]
    clay: List[float]
    watsat: List[float]

    nsoi: int                         # number of soil layers
    dz: List[float]                   # layer thickness
    z_plus_onehalf: List[float]
    z: List[float]
    dz_plus_onehalf: List[float]
    tsoi: List[float]                 # temperature per layer
    h2osoi_liq: List[float]           # liquid water per layer
    h2osoi_ice: List[float]           # ice per layer
    tk: List[float]                   # thermal conductivity per layer
    cv: List[float]                   # heat capacity per layer

    soil_texture: int                 # texture class index
    method: str                       # 'excess-heat' or 'apparent-heat-capacity'

    gsoi: float = 0.0                 # energy flux into soil
    hfsoi: float = 0.0                # phase change energy flux

# --- Utility function -------------------------------------------------------

def tridiagonal_solver(a: List[float], b: List[float], c: List[float],
                        d: List[float], n: int) -> List[float]:
    """Solve a tridiagonal system using forward/back substitution."""
    e = [0.0] * n
    f = [0.0] * n
    u = [0.0] * n
    e[0] = c[0] / b[0]
    for i in range(1, n-1):
        e[i] = c[i] / (b[i] - a[i] * e[i-1])
    f[0] = d[0] / b[0]
    for i in range(1, n):
        f[i] = (d[i] - a[i] * f[i-1]) / (b[i] - a[i] * e[i-1])
    u[n-1] = f[n-1]
    for i in range(n-2, -1, -1):
        u[i] = f[i] - e[i] * u[i+1]
    return u

# --- Soil thermal properties ------------------------------------------------

def soil_thermal_properties(physcon: Physcon, soil: SoilVar) -> None:
    for i in range(soil.nsoi):
        k = soil.soil_texture - 1  # zero based index
        watliq = soil.h2osoi_liq[i] / (physcon.rhowat * soil.dz[i])
        watice = soil.h2osoi_ice[i] / (physcon.rhoice * soil.dz[i])
        fliq = 0.0
        if (watliq + watice) > 0:
            fliq = watliq / (watliq + watice)
        s = min((watliq + watice) / soil.watsat[k], 1.0)
        bd = 2700 * (1 - soil.watsat[k])
        tkdry = (0.135 * bd + 64.7) / (2700 - 0.947 * bd)
        tk_quartz = 7.7
        quartz = soil.sand[k] / 100.0
        tko = 2.0 if quartz > 0.2 else 3.0
        tksol = tk_quartz ** quartz * tko ** (1 - quartz)
        tksat = (tksol ** (1 - soil.watsat[k]) *
                 physcon.tkwat ** (fliq * soil.watsat[k]) *
                 physcon.tkice ** (soil.watsat[k] - fliq * soil.watsat[k]))
        tksat_u = (tksol ** (1 - soil.watsat[k]) *
                   physcon.tkwat ** soil.watsat[k])
        tksat_f = (tksol ** (1 - soil.watsat[k]) *
                   physcon.tkice ** soil.watsat[k])
        if soil.sand[k] < 50:
            ke_u = math.log10(max(s, 0.1)) + 1.0
        else:
            ke_u = 0.7 * math.log10(max(s, 0.05)) + 1.0
        ke_f = s
        if soil.tsoi[i] >= physcon.tfrz:
            ke = ke_u
        else:
            ke = ke_f
        soil.tk[i] = (tksat - tkdry) * ke + tkdry
        tku = (tksat_u - tkdry) * ke_u + tkdry
        tkf = (tksat_f - tkdry) * ke_f + tkdry
        cvsol = 1.926e06
        soil.cv[i] = ((1 - soil.watsat[k]) * cvsol +
                      physcon.cvwat * watliq + physcon.cvice * watice)
        cvu = (1 - soil.watsat[k]) * cvsol + physcon.cvwat * (watliq + watice)
        cvf = (1 - soil.watsat[k]) * cvsol + physcon.cvice * (watliq + watice)
        if soil.method == 'apparent-heat-capacity':
            tinc = 0.5
            ql = physcon.hfus * (physcon.rhowat * watliq + physcon.rhoice * watice)
            if soil.tsoi[i] > physcon.tfrz + tinc:
                soil.cv[i] = cvu
                soil.tk[i] = tku
            if physcon.tfrz - tinc <= soil.tsoi[i] <= physcon.tfrz + tinc:
                soil.cv[i] = (cvf + cvu) / 2.0 + ql / (2 * tinc)
                soil.tk[i] = tkf + (tku - tkf) * (
                    soil.tsoi[i] - physcon.tfrz + tinc) / (2 * tinc)
            if soil.tsoi[i] < physcon.tfrz - tinc:
                soil.cv[i] = cvf
                soil.tk[i] = tkf

# --- Phase change -----------------------------------------------------------

def phase_change(physcon: Physcon, soil: SoilVar, dt: float) -> None:
    soil.hfsoi = 0.0
    for i in range(soil.nsoi):
        wliq0 = soil.h2osoi_liq[i]
        wice0 = soil.h2osoi_ice[i]
        wmass0 = wliq0 + wice0
        tsoi0 = soil.tsoi[i]
        imelt = 0
        if soil.h2osoi_ice[i] > 0 and soil.tsoi[i] > physcon.tfrz:
            imelt = 1
            soil.tsoi[i] = physcon.tfrz
        if soil.h2osoi_liq[i] > 0 and soil.tsoi[i] < physcon.tfrz:
            imelt = 2
            soil.tsoi[i] = physcon.tfrz
        if imelt > 0:
            heat_flux_pot = (soil.tsoi[i] - tsoi0) * soil.cv[i] * soil.dz[i] / dt
        else:
            heat_flux_pot = 0.0
        heat_flux_max = 0.0
        if imelt == 1:
            heat_flux_max = -soil.h2osoi_ice[i] * physcon.hfus / dt
        if imelt == 2:
            heat_flux_max = soil.h2osoi_liq[i] * physcon.hfus / dt
        if imelt > 0:
            ice_flux = heat_flux_pot / physcon.hfus
            soil.h2osoi_ice[i] = wice0 + ice_flux * dt
            soil.h2osoi_ice[i] = max(0.0, soil.h2osoi_ice[i])
            soil.h2osoi_ice[i] = min(wmass0, soil.h2osoi_ice[i])
            soil.h2osoi_liq[i] = max(0.0, (wmass0 - soil.h2osoi_ice[i]))
            heat_flux = physcon.hfus * (soil.h2osoi_ice[i] - wice0) / dt
            soil.hfsoi += heat_flux
            residual = heat_flux_pot - heat_flux
            soil.tsoi[i] = soil.tsoi[i] - residual * dt / (soil.cv[i] * soil.dz[i])
            if abs(heat_flux) > abs(heat_flux_max) + 1e-12:
                raise RuntimeError('Soil temperature energy conservation error: phase change')
            if imelt == 2:
                constraint = min(heat_flux_pot, heat_flux_max)
                err = heat_flux - constraint
                if abs(err) > 1e-3:
                    raise RuntimeError('Soil temperature energy conservation error: freezing energy flux')
                err = (soil.h2osoi_ice[i] - wice0) - constraint / physcon.hfus * dt
                if abs(err) > 1e-3:
                    raise RuntimeError('Soil temperature energy conservation error: freezing ice flux')
            if imelt == 1:
                constraint = max(heat_flux_pot, heat_flux_max)
                err = heat_flux - constraint
                if abs(err) > 1e-3:
                    raise RuntimeError('Soil temperature energy conservation error: thawing energy flux')
                err = (soil.h2osoi_ice[i] - wice0) - constraint / physcon.hfus * dt
                if abs(err) > 1e-3:
                    raise RuntimeError('Soil temperature energy conservation error: thawing ice flux')

# --- Soil temperature update ------------------------------------------------

def soil_temperature(physcon: Physcon, soil: SoilVar, tsurf: float, dt: float) -> None:
    tsoi0 = soil.tsoi.copy()
    tk_plus_onehalf = [0.0] * (soil.nsoi - 1)
    for i in range(soil.nsoi - 1):
        num = soil.tk[i] * soil.tk[i + 1] * (soil.z[i] - soil.z[i + 1])
        den = (soil.tk[i] * (soil.z_plus_onehalf[i] - soil.z[i + 1]) +
               soil.tk[i + 1] * (soil.z[i] - soil.z_plus_onehalf[i]))
        tk_plus_onehalf[i] = num / den
    a = [0.0] * soil.nsoi
    b = [0.0] * soil.nsoi
    c = [0.0] * soil.nsoi
    d = [0.0] * soil.nsoi
    i = 0
    m = soil.cv[i] * soil.dz[i] / dt
    a[i] = 0.0
    c[i] = -tk_plus_onehalf[i] / soil.dz_plus_onehalf[i]
    b[i] = m - c[i] + soil.tk[i] / (0.0 - soil.z[i])
    d[i] = m * soil.tsoi[i] + soil.tk[i] / (0.0 - soil.z[i]) * tsurf
    for i in range(1, soil.nsoi - 1):
        m = soil.cv[i] * soil.dz[i] / dt
        a[i] = -tk_plus_onehalf[i - 1] / soil.dz_plus_onehalf[i - 1]
        c[i] = -tk_plus_onehalf[i] / soil.dz_plus_onehalf[i]
        b[i] = m - a[i] - c[i]
        d[i] = m * soil.tsoi[i]
    i = soil.nsoi - 1
    m = soil.cv[i] * soil.dz[i] / dt
    a[i] = -tk_plus_onehalf[i - 1] / soil.dz_plus_onehalf[i - 1]
    c[i] = 0.0
    b[i] = m - a[i]
    d[i] = m * soil.tsoi[i]
    soil.tsoi = tridiagonal_solver(a, b, c, d, soil.nsoi)
    soil.gsoi = soil.tk[0] * (tsurf - soil.tsoi[0]) / (0.0 - soil.z[0])
    if soil.method == 'apparent-heat-capacity':
        soil.hfsoi = 0.0
    elif soil.method == 'excess-heat':
        phase_change(physcon, soil, dt)
    edif = 0.0
    for i in range(soil.nsoi):
        edif += soil.cv[i] * soil.dz[i] * (soil.tsoi[i] - tsoi0[i]) / dt
    err = edif - soil.gsoi - soil.hfsoi
    if abs(err) > 1e-3:
        raise RuntimeError('Soil temperature energy conservation error')

# --- Main program translated from sp_05_02.m --------------------------------

def main():
    physcon = Physcon(
        tfrz=273.15,
        cwat=4188.0,
        cice=2117.27,
        rhowat=1000.0,
        rhoice=917.0,
        cvwat=4188.0 * 1000.0,
        cvice=2117.27 * 917.0,
        tkwat=0.57,
        tkice=2.29,
        hfus=0.3337e6
    )
    soil = SoilVar(
        silt=[5.0, 12.0, 32.0, 70.0, 39.0, 15.0, 56.0, 34.0, 6.0, 47.0, 20.0],
        sand=[92.0, 82.0, 58.0, 17.0, 43.0, 58.0, 10.0, 32.0, 52.0, 6.0, 22.0],
        clay=[3.0, 6.0, 10.0, 13.0, 18.0, 27.0, 34.0, 34.0, 42.0, 47.0, 58.0],
        watsat=[0.395, 0.410, 0.435, 0.485, 0.451, 0.420, 0.477, 0.476, 0.426, 0.492, 0.482],
        nsoi=120,
        dz=[0.025]*120,
        z_plus_onehalf=[0.0]*120,
        z=[0.0]*120,
        dz_plus_onehalf=[0.0]*120,
        tsoi=[0.0]*120,
        h2osoi_liq=[0.0]*120,
        h2osoi_ice=[0.0]*120,
        tk=[0.0]*120,
        cv=[0.0]*120,
        soil_texture=1,
        method='excess-heat'
    )
    soil.z_plus_onehalf[0] = -soil.dz[0]
    for i in range(1, soil.nsoi):
        soil.z_plus_onehalf[i] = soil.z_plus_onehalf[i-1] - soil.dz[i]
    soil.z[0] = 0.5 * soil.z_plus_onehalf[0]
    for i in range(1, soil.nsoi):
        soil.z[i] = 0.5 * (soil.z_plus_onehalf[i-1] + soil.z_plus_onehalf[i])
    for i in range(soil.nsoi - 1):
        soil.dz_plus_onehalf[i] = soil.z[i] - soil.z[i+1]
    soil.dz_plus_onehalf[soil.nsoi-1] = 0.5 * soil.dz[soil.nsoi-1]
    for i in range(soil.nsoi):
        soil.tsoi[i] = physcon.tfrz + 2.0
        h2osoi_sat = soil.watsat[soil.soil_texture-1] * physcon.rhowat * soil.dz[i]
        if soil.tsoi[i] > physcon.tfrz:
            soil.h2osoi_ice[i] = 0.0
            soil.h2osoi_liq[i] = 0.8 * h2osoi_sat
        else:
            soil.h2osoi_liq[i] = 0.0
            soil.h2osoi_ice[i] = 0.8 * h2osoi_sat
    tmean = physcon.tfrz + 15.0
    trange = 10.0
    dt = 1800.0
    nday = 200
    ntim = round(86400.0/dt)
    hour_vec = []
    z_vec = []
    tsoi_vec = []
    hour_out = [0.0]*ntim
    z_out = [0.0]*(soil.nsoi+1)
    tsoi_out = [[0.0]*ntim for _ in range(soil.nsoi+1)]
    for iday in range(1, nday+1):
        print(f"day = {iday:6.0f}")
        for itim in range(ntim):
            hour = (itim+1) * (dt/86400.0*24.0)
            tsurf = tmean + 0.5 * trange * math.sin(2*math.pi/24.0 * (hour-8.0))
            soil_thermal_properties(physcon, soil)
            soil_temperature(physcon, soil, tsurf, dt)
            if iday == nday:
                hour_vec.append(hour)
                z_vec.append(0.0)
                tsoi_vec.append(tsurf - physcon.tfrz)
                hour_out[itim] = hour
                z_out[0] = 0.0
                tsoi_out[0][itim] = tsurf - physcon.tfrz
                idx = 1
                for j in range(soil.nsoi):
                    if soil.z[j] > -1.0:
                        hour_vec.append(hour)
                        z_vec.append(soil.z[j]*100.0)
                        tsoi_vec.append(soil.tsoi[j] - physcon.tfrz)
                        z_out[idx] = soil.z[j]*100.0
                        tsoi_out[idx][itim] = soil.tsoi[j] - physcon.tfrz
                        idx += 1
    with open('data.txt','w') as f:
        f.write('{:12s} {:12s} {:12s}\n'.format('hour','z','tsoi'))
        for h,zv,tv in zip(hour_vec,z_vec,tsoi_vec):
            f.write('{:12.3f} {:12.3f} {:12.3f}\n'.format(h,zv,tv))
    # Plotting not included (would require matplotlib)

if __name__ == '__main__':
    main()
