import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from soil_thermal_properties import soil_thermal_properties
from soil_temperature import soil_temperature


@dataclass
class PhysCon:
    tfrz: float = 273.15
    cwat: float = 4188.0
    cice: float = 2117.27
    rhowat: float = 1000.0
    rhoice: float = 917.0
    tkwat: float = 0.57
    tkice: float = 2.29
    hfus: float = 0.3337e6

    @property
    def cvwat(self):
        return self.cwat * self.rhowat

    @property
    def cvice(self):
        return self.cice * self.rhoice


def main():
    physcon = PhysCon()

    soilvar = {
        'silt': np.array([5.0, 12.0, 32.0, 70.0, 39.0, 15.0, 56.0, 34.0, 6.0, 47.0, 20.0]),
        'sand': np.array([92.0, 82.0, 58.0, 17.0, 43.0, 58.0, 10.0, 32.0, 52.0, 6.0, 22.0]),
        'clay': np.array([3.0, 6.0, 10.0, 13.0, 18.0, 27.0, 34.0, 34.0, 42.0, 47.0, 58.0]),
        'watsat': np.array([0.395, 0.410, 0.435, 0.485, 0.451, 0.420, 0.477, 0.476, 0.426, 0.492, 0.482]),
        'soil_texture': 1,
        'method': 'excess-heat',
    }

    tmean = physcon.tfrz + 15.0
    trange = 10.0
    dt = 1800
    nday = 200

    nsoi = 120
    soilvar['nsoi'] = nsoi
    soilvar['dz'] = np.full(nsoi, 0.025)

    soilvar['z_plus_onehalf'] = np.zeros(nsoi)
    soilvar['z_plus_onehalf'][0] = -soilvar['dz'][0]
    for i in range(1, nsoi):
        soilvar['z_plus_onehalf'][i] = soilvar['z_plus_onehalf'][i - 1] - soilvar['dz'][i]

    soilvar['z'] = np.zeros(nsoi)
    soilvar['z'][0] = 0.5 * soilvar['z_plus_onehalf'][0]
    for i in range(1, nsoi):
        soilvar['z'][i] = 0.5 * (soilvar['z_plus_onehalf'][i - 1] + soilvar['z_plus_onehalf'][i])

    soilvar['dz_plus_onehalf'] = np.zeros(nsoi)
    for i in range(nsoi - 1):
        soilvar['dz_plus_onehalf'][i] = soilvar['z'][i] - soilvar['z'][i + 1]
    soilvar['dz_plus_onehalf'][-1] = 0.5 * soilvar['dz'][-1]

    soilvar['tsoi'] = np.full(nsoi, physcon.tfrz + 2.0)
    h2osoi_sat = soilvar['watsat'][soilvar['soil_texture'] - 1] * physcon.rhowat * soilvar['dz']
    soilvar['h2osoi_liq'] = np.where(soilvar['tsoi'] > physcon.tfrz, 0.8 * h2osoi_sat, 0.0)
    soilvar['h2osoi_ice'] = np.where(soilvar['tsoi'] <= physcon.tfrz, 0.8 * h2osoi_sat, 0.0)

    soilvar['tk'] = np.zeros(nsoi)
    soilvar['cv'] = np.zeros(nsoi)
    soilvar['gsoi'] = 0.0
    soilvar['hfsoi'] = 0.0

    ntim = round(86400 / dt)
    hour_vec = []
    z_vec = []
    tsoi_vec = []
    hour_out = np.zeros(ntim)
    z_out = np.zeros(nsoi + 1)
    tsoi_out = np.zeros((nsoi + 1, ntim))

    for iday in range(1, nday + 1):
        print(f"day = {iday:6.0f}")
        for itim in range(1, ntim + 1):
            hour = itim * (dt / 86400 * 24)
            tsurf = tmean + 0.5 * trange * np.sin(2 * np.pi / 24 * (hour - 8.0))

            soilvar = soil_thermal_properties(physcon, soilvar)
            soilvar = soil_temperature(physcon, soilvar, tsurf, dt)

            if iday == nday:
                hour_vec.append(hour)
                z_vec.append(0.0)
                tsoi_vec.append(tsurf - physcon.tfrz)
                hour_out[itim - 1] = hour
                tsoi_out[0, itim - 1] = tsurf - physcon.tfrz
                z_out[0] = 0.0
                for i in range(nsoi):
                    if soilvar['z'][i] > -1.0:
                        hour_vec.append(hour)
                        z_vec.append(soilvar['z'][i] * 100.0)
                        tsoi_vec.append(soilvar['tsoi'][i] - physcon.tfrz)
                        z_out[i + 1] = soilvar['z'][i] * 100.0
                        tsoi_out[i + 1, itim - 1] = soilvar['tsoi'][i] - physcon.tfrz

    A = np.vstack([hour_vec, z_vec, tsoi_vec]).T
    np.savetxt('data.txt', A, fmt='%12.3f', header='hour z tsoi', comments='')

    plt.contour(hour_out, z_out, tsoi_out, levels=15)
    plt.title('Soil Temperature (°C)')
    plt.xlabel('Time of day (hours)')
    plt.ylabel('Soil depth (cm)')
    plt.show()


if __name__ == '__main__':
    main()
