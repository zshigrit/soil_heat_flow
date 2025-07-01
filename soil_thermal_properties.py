import numpy as np


def soil_thermal_properties(physcon, soilvar):
    """Calculate soil thermal conductivity and heat capacity."""
    nsoi = soilvar['nsoi']
    texture = soilvar['soil_texture'] - 1  # zero-based index

    for i in range(nsoi):
        watliq = soilvar['h2osoi_liq'][i] / (physcon.rhowat * soilvar['dz'][i])
        watice = soilvar['h2osoi_ice'][i] / (physcon.rhoice * soilvar['dz'][i])

        if watliq + watice > 0:
            fliq = watliq / (watliq + watice)
        else:
            fliq = 0.0

        s = min((watliq + watice) / soilvar['watsat'][texture], 1.0)

        bd = 2700 * (1 - soilvar['watsat'][texture])
        tkdry = (0.135 * bd + 64.7) / (2700 - 0.947 * bd)

        tk_quartz = 7.7
        quartz = soilvar['sand'][texture] / 100.0
        tko = 2.0 if quartz > 0.2 else 3.0
        tksol = tk_quartz ** quartz * tko ** (1 - quartz)

        tksat = (
            tksol ** (1 - soilvar['watsat'][texture])
            * physcon.tkwat ** (fliq * soilvar['watsat'][texture])
            * physcon.tkice ** (soilvar['watsat'][texture] - fliq * soilvar['watsat'][texture])
        )
        tksat_u = tksol ** (1 - soilvar['watsat'][texture]) * physcon.tkwat ** soilvar['watsat'][texture]
        tksat_f = tksol ** (1 - soilvar['watsat'][texture]) * physcon.tkice ** soilvar['watsat'][texture]

        if soilvar['sand'][texture] < 50:
            ke_u = np.log10(max(s, 0.1)) + 1.0
        else:
            ke_u = 0.7 * np.log10(max(s, 0.05)) + 1.0
        ke_f = s

        ke = ke_u if soilvar['tsoi'][i] >= physcon.tfrz else ke_f

        soilvar['tk'][i] = (tksat - tkdry) * ke + tkdry
        tku = (tksat_u - tkdry) * ke_u + tkdry
        tkf = (tksat_f - tkdry) * ke_f + tkdry

        cvsol = 1.926e6
        soilvar['cv'][i] = (
            (1 - soilvar['watsat'][texture]) * cvsol
            + physcon.cvwat * watliq
            + physcon.cvice * watice
        )
        cvu = (1 - soilvar['watsat'][texture]) * cvsol + physcon.cvwat * (watliq + watice)
        cvf = (1 - soilvar['watsat'][texture]) * cvsol + physcon.cvice * (watliq + watice)

        if soilvar['method'] == 'apparent-heat-capacity':
            tinc = 0.5
            ql = physcon.hfus * (physcon.rhowat * watliq + physcon.rhoice * watice)
            if soilvar['tsoi'][i] > physcon.tfrz + tinc:
                soilvar['cv'][i] = cvu
                soilvar['tk'][i] = tku
            elif physcon.tfrz - tinc <= soilvar['tsoi'][i] <= physcon.tfrz + tinc:
                soilvar['cv'][i] = (cvf + cvu) / 2 + ql / (2 * tinc)
                soilvar['tk'][i] = tkf + (tku - tkf) * (soilvar['tsoi'][i] - physcon.tfrz + tinc) / (2 * tinc)
            else:
                soilvar['cv'][i] = cvf
                soilvar['tk'][i] = tkf

    return soilvar
