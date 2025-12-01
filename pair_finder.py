import numpy as np
from matplotlib import pyplot as plt
import swiftsimio
import time
import unyt
import os
from multiprocessing import Pool
from functools import partial
import sys

def calc_distances(data, coordinates, coord):
    distances_wrapped = np.abs(coordinates - coord)
    distances_wrapped -= 0.5 * data.metadata.boxsize[0]
    distances_wrapped = 0.5 * data.metadata.boxsize[0] - np.abs(distances_wrapped)
    return distances_wrapped

snapshot = sys.argv[1]                # snapshot id, e.g. '0127'
size = sys.argv[2]                    # Boxsize [Mpc]
resolution = sys.argv[3]              # 'm7', 'm6' or 'm5'
cutoff_min_smass = int(sys.argv[4])
n_jobs = int(sys.argv[5])

home = '/cosma/home/do019/dc-vanz1/'
filename_data = home+'hdf5_links/L'+size+resolution+'/halo_properties_'+snapshot+'.hdf5'
output_file = f'{home}GalaxyPairs/3D{cutoff_min_smass}/L{size}{resolution}/{snapshot}.npy'

data = swiftsimio.load(filename_data)
print(f'z = {data.metadata.redshift:.2f}')

stellar_mass_all = data.exclusive_sphere_50kpc.stellar_mass.to('Msun')

print('max smass:', np.log10(np.max(stellar_mass_all)))

mask = stellar_mass_all > 10**cutoff_min_smass

cutoff_secondary = cutoff_min_smass - 1
mask_secondary = stellar_mass_all > 10**cutoff_secondary

halo_centers_all = data.input_halos.halo_centre.to_physical()

smass = stellar_mass_all[mask]                                                      #Stellar mass
smass_secondary = stellar_mass_all[mask_secondary]
halo_centers = halo_centers_all[mask]                                               #Halo center coordinates
halo_centers_secondary = halo_centers_all[mask_secondary]

Ngalaxies = len(smass)
Nsecondary = np.sum(mask_secondary)

print('Suitable galaxies:', Ngalaxies)
print('Possible secondaries:', Nsecondary)

time0 = time.time()

x, y, z = np.transpose(halo_centers_secondary)  #coordinates of possible secondary galaxies

def function(i):
    if i%1000 == 0:
        print(f'{i}/{Ngalaxies} -- {100*i/Ngalaxies:.1f} %')
    suitable_smass = smass_secondary > 0.1*smass[i]
    x, y, z = np.transpose(halo_centers_secondary[suitable_smass])  #coordinates of possible secondary galaxies
    xdistances = calc_distances(data, x, halo_centers[i][0])
    ydistances = calc_distances(data, y, halo_centers[i][1])
    zdistances = calc_distances(data, z, halo_centers[i][2])
    distances2 = xdistances**2 + ydistances**2 + zdistances**2
    current_galaxy = np.argmin(distances2)
    distances2[current_galaxy] = np.inf
        #This loop locates the nearest galaxy within 10 and 100 percent of its mass
    NNs = [0, 0]
    iis = [0, 0]
    N2 = np.sum(distances2 < 2**2)
    for j in range(2):
        distances_index_local = np.argmin(distances2)
        if distances2[distances_index_local] < np.inf:
            distances_index = np.flatnonzero(suitable_smass)[distances_index_local]
            iis[j] = distances_index
            NNs[j] = distances2[distances_index_local]
            distances2[distances_index_local] = np.inf
        if NNs[1] != 0:
            break
    else:
        if NNs[0] == 0:
            iis[0] = i
            NNs[0] = np.inf
        iis[1] = i
        NNs[1] = np.inf

    return NNs, iis, [N2, 0]

if __name__ == '__main__':
    with Pool(processes=n_jobs) as pool:
        y = np.arange(0, Ngalaxies)
        results = pool.map(function, y)

interacting_indices = np.array([results[i][1] for i in range(len(results))], dtype = int)
interacting_indices = np.array([results[i][1] for i in range(len(results))], dtype = int)
distanceNN = np.array(results)[:,0]**.5
N2 = np.array(results)[:,2,0].astype(int)

halo_centers_all_np = halo_centers_all.to_value()
halo_index_map = {tuple(halo_centers_all_np[i]): i for i in range(halo_centers_all_np.shape[0])}

shape = interacting_indices.shape
interacting_indices = interacting_indices.flatten()
indices_secondary = np.array([halo_index_map[tuple(coord.to_value())] for coord in halo_centers_secondary[interacting_indices]])
indices_secondary = indices_secondary.reshape(*shape)

dictionary = {
    'Indices': np.arange(0, len(mask), 1)[mask],
    'DistanceNN': distanceNN,
    'Interacting_Index': indices_secondary,
    'N2': N2,
    'Boxsize': data.metadata.boxsize,
    'Redshift': data.metadata.redshift
}

np.save(output_file, dictionary)