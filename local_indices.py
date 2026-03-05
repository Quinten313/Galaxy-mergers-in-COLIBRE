import numpy as np
import swiftsimio

def snapshot_list(start, end):

    def zero_padding(x):
        zeros = 3-int(np.log10(x))
        return zeros*'0'+str(x)
        
    snapshot_ids = np.arange(start, end+1)
    snapshot_str = [zero_padding(i) for i in snapshot_ids]
    return snapshot_str[::-1]

def local_indices(trackids_main, trackids_new_all):
    index_map = {v: i for i, v in enumerate(trackids_new_all)}
    indices = np.array([index_map[x] if x in index_map else np.nan for x in trackids_main])
    return indices

def save_ssfr(snapshot):
    data = swiftsimio.load(f'../hdf5_links/L200m6/halo_properties_{snapshot}.hdf5')
    trackids_new_all = np.array(data.input_halos_hbtplus.track_id)
    
    indices_interacting = local_indices(trackids_main_interacting, trackids_new_all)
    indices_isolated = local_indices(trackids_main_isolated, trackids_new_all)

    smass = np.array(data.exclusive_sphere_50kpc.stellar_mass.to('Msun'))
    sfr = np.array(data.exclusive_sphere_50kpc.star_formation_rate.to('Msun/yr'))
    ssfr = sfr / smass * 1e9
    np.savez(f'../personal_storage/time_evolution/{snapshot}', interacting=ssfr[indices_interacting], isolated=ssfr[indices_isolated], redshift=data.metadata.redshift)

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

snapshots = snapshot_list(92, 127)
for s in snapshots:
    print(f'Save sSFRs {s}')
    save_ssfr(s)
