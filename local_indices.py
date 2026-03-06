import numpy as np
import swiftsimio

def snapshot_list(start, end):

    def zero_padding(x):
        zeros = 3-int(np.log10(x))
        return zeros*'0'+str(x)
        
    snapshot_ids = np.arange(start, end+1)
    snapshot_str = [zero_padding(i) for i in snapshot_ids]
    return snapshot_str[::-1]

def wrapped_distances(coord_array1, coord_array2):
    distances_wrapped = np.abs(coord_array1 - coord_array2)
    distances_wrapped -= 0.5 * boxsize
    distances_wrapped = 0.5 * boxsize - np.abs(distances_wrapped)
    return distances_wrapped

def calc_separation(coord_array1, coord_array2):
    distances_squared = np.sum(wrapped_distances(coord_array1, coord_array2)**2, axis=1)
    return np.sqrt(distances_squared)

def save_ssfr(snapshot):
    data = swiftsimio.load(f'../hdf5_links/L200m6/halo_properties_{snapshot}.hdf5')
    trackids_new_all = np.array(data.input_halos_hbtplus.track_id)

    index_map = {v: i for i, v in enumerate(trackids_new_all)}
    
    indices_interacting = np.array([index_map[x] if x in index_map else np.nan for x in trackids_main_interacting])
    indices_isolated = np.array([index_map[x] if x in index_map else np.nan for x in trackids_main_isolated])
    indices_secondary = np.array([index_map[x] if x in index_map else np.nan for x in trackids_main_secondary])

    halo_centers = np.array(data.input_halos.halo_centre.to('kpc'))
    smass = np.array(data.exclusive_sphere_50kpc.stellar_mass.to('Msun'))
    sfr = np.array(data.exclusive_sphere_50kpc.star_formation_rate.to('Msun/yr'))
    ssfr = sfr / smass * 1e9

    nonnan = ~np.isnan(indices_interacting) & ~np.isnan(indices_isolated) & ~np.isnan(indices_secondary)
    separation = np.zeros(len(indices_interacting))
    separation[nonnan] = calc_separation(halo_centers[indices_interacting], halo_centers[indices_secondary])
    separation[~nonnan] = np.nan
    np.savez(f'../personal_storage/time_evolution/{snapshot}', interacting=ssfr[indices_interacting], isolated=ssfr[indices_isolated], separation=separation, redshift=data.metadata.redshift)

boxsize = 200_000 # kpc
print('Loading data (0127)')
data = swiftsimio.load(f'../hdf5_links/L200m6/halo_properties_0127.hdf5')
trackids_main_all = np.array(data.input_halos_hbtplus.track_id)

interacting = np.load('../personal_storage/GalaxyProperties/3D/L200m6/0127/s0.01/Interacting.npz', allow_pickle=True)
idx_interacting = interacting['indices']
r = interacting['r']
trackids_main_interacting = trackids_main_all[idx_interacting[r < .05]]

isolated = np.load('../personal_storage/GalaxyProperties/3D/L200m6/0127/s0.01/Isolated.npz', allow_pickle=True)
idx_isolated = isolated['indices']
trackids_main_isolated = trackids_main_all[idx_isolated[r < .05]]

secondary = np.load('../personal_storage/GalaxyProperties/3D/L200m6/0127/s0.01/Secondary.npz', allow_pickle=True)
idx_secondary = secondary['indices']
trackids_main_secondary = trackids_main_all[idx_secondary[r < .05]]

snapshots = snapshot_list(1, 69)
for s in snapshots:
    print(f'Save sSFRs {s}')
    save_ssfr(s)
