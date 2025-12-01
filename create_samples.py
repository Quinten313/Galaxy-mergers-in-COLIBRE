import numpy as np
import swiftsimio
import time
import unyt
import os
import sys

epsilon_r = 0.1                 # Radiative efficiency
c = 3e10                        # Speed of light to cm/s
Msun = 1.988e33                 # Solar mass to grams
yr = 365.25*24*3600             # year to seconds
Ledd = 1.26e38                  # Eddington luminosity per Msun (erg/s)

# Calculates the black hole luminosity, based on the black hole accretion rate
def bh_luminosity(bhar):
    return np.array(epsilon_r*bhar*Msun*c**2/yr)

# Calculates the specific black hole acretion rate given the black hole mass and accretion rate. Returns zero if BH mass == 0
def sbhar(bhmass, bhar):
    Medd = Ledd / c**2 / epsilon_r * yr / Msun * bhmass  # Eddington mass accretion rate [Msun/yr]
    sbhar = np.array(bhar / Medd)
    sbhar[bhmass == 0] = 0
    return sbhar

class Analyse:

    def __init__(self, filename_simulation, filename_dictionary, simulation, snapshot, tag, mass_cutoff = [10, 12], sfr_thresholds = [0,0]):
        try:
            for attr in univs[simulation+'_'+snapshot].__dict__:
                setattr(self, attr, getattr(univs[simulation+'_'+snapshot], attr))
            print('Reusing data')
        except:
            print('Importing data')
            self.data = swiftsimio.load(filename_simulation)
            self.load_dictionary(filename_dictionary)
            self.load_data()
            self.ssfr_half_mass_radius()
        self.path = f'/cosma8/data/do019/dc-vanz1/GalaxyProperties/3D/{simulation}/'
        self.snapshot = snapshot
        self.tag = tag
        self.mass_cutoff = mass_cutoff
        self.sfr_thresholds = sfr_thresholds
        self.interacting_sample()
        self.isolated_sample()
        self.save_file('Interacting.npz', self.interacting)
        self.save_file('Isolated.npz', self.isolated)
        self.save_file('Secondary.npz', self.interacting_indices[self.interacting])
        np.savez(f'{self.path}{snapshot}/{tag}/samples.npz',interacting=self.interacting, isolated=self.isolated, secondary=self.interacting_indices[self.interacting])

    def load_dictionary(self, filename_dictionary):
        dictionary = np.load(filename_dictionary, allow_pickle=True).item()

        self.indices = dictionary['Indices']
        self.distanceNN = dictionary['DistanceNN']
        self.interacting_indices_local = dictionary['Interacting_Index']
        self.z = dictionary['Redshift']
        self.N2_local = dictionary['N2']
        self.boxsize = dictionary['Boxsize'][0]
        
        print(f"Boxsize: {dictionary['Boxsize'][0]}\nRedshift: " + str(round(dictionary['Redshift'], 2)))

    def load_data(self):
        # Loads all properties used in this analysis
        self.smass = self.data.exclusive_sphere_50kpc.stellar_mass.to('Msun')
        self.smass1 = self.data.exclusive_sphere_1kpc.stellar_mass.to('Msun')
        self.smass3 = self.data.exclusive_sphere_3kpc.stellar_mass.to('Msun')
        self.smass10 = self.data.exclusive_sphere_10kpc.stellar_mass.to('Msun')
        self.smass30 = self.data.exclusive_sphere_30kpc.stellar_mass.to('Msun')
        
        self.sfr1 = self.data.exclusive_sphere_1kpc.star_formation_rate.to('Msun/yr')
        self.sfr3 = self.data.exclusive_sphere_3kpc.star_formation_rate.to('Msun/yr')
        self.sfr10 = self.data.exclusive_sphere_10kpc.star_formation_rate.to('Msun/yr')
        self.sfr30 = self.data.exclusive_sphere_30kpc.star_formation_rate.to('Msun/yr')
        self.sfr = self.data.exclusive_sphere_50kpc.star_formation_rate.to('Msun/yr')
        
        self.ssfr1 = (self.sfr1 / self.smass1).to('1/Gyr')
        self.ssfr3 = (self.sfr3 / self.smass3).to('1/Gyr')
        self.ssfr10 = (self.sfr10 / self.smass10).to('1/Gyr')
        self.ssfr30 = (self.sfr30 / self.smass30).to('1/Gyr')
        self.ssfr50 = (self.sfr / self.smass).to('1/Gyr')

        self.gmass = self.data.exclusive_sphere_50kpc.gas_mass.to('Msun')
        self.halo_centers = self.data.input_halos.halo_centre.to_physical()
        
        self.hmass = self.data.spherical_overdensity_200_crit.total_mass.to('Msun')
        halo_mass_sattelites = self.data.bound_subhalo.total_mass.to("Msun")
        satellite_mask = self.data.input_halos.is_central == 0
        self.hmass[satellite_mask] = halo_mass_sattelites[satellite_mask]
        
        self.is_central = self.data.input_halos.is_central
        self.v = np.array(self.data.exclusive_sphere_50kpc.centre_of_mass_velocity[:, 0].to('km/s')) + \
                 np.array(self.data.input_halos.halo_centre[:, 0].to_physical() * \
                 unyt.unyt_array(self.data.metadata.cosmology_raw['H [internal units]'][0], 'km/s/Mpc'))
        self.bhmass = self.data.exclusive_sphere_50kpc.most_massive_black_hole_mass.to('Msun')
        self.bhar = self.data.exclusive_sphere_50kpc.most_massive_black_hole_accretion_rate.to('Msun/yr')
        self.stellar_half_mass_radius = self.data.exclusive_sphere_50kpc.half_mass_radius_stars.to_physical().to('Mpc')

        interacting_indices = np.zeros(len(self.smass))
        self.r = np.zeros(len(self.smass))
        self.r2 = np.zeros(len(self.smass))
        N2 = np.zeros(len(self.smass))
        for i in range(len(self.indices)):
            interacting_indices[self.indices[i]] = self.interacting_indices_local[i, 0]
            self.r[self.indices[i]] = self.distanceNN[i, 0]
            self.r2[self.indices[i]] = self.distanceNN[i, 1]
            N2[self.indices[i]] = self.N2_local[i]
        self.interacting_indices = interacting_indices.astype(int)
        self.N2 = N2.astype(int)

    def ssfr_half_mass_radius(self):          
        time0 = time.time()

        half_mass_radius = self.stellar_half_mass_radius.to('kpc')[self.indices].value

        ssfrs = [self.ssfr3[self.indices], self.ssfr10[self.indices], self.ssfr30[self.indices], self.ssfr50[self.indices]]
        smasses = [self.smass3[self.indices], self.smass10[self.indices], self.smass30[self.indices], self.smass[self.indices]]
        sfrs = [self.sfr3[self.indices], self.sfr10[self.indices], self.sfr30[self.indices], self.sfr[self.indices]]
        radii = [3,10,30,50]
        
        ssfr = []
        smass_avg = []
        for i in range(len(self.indices)):
            if half_mass_radius[i] < radii[0]:    # If the SHMR is smaller than the smallest allowed aperture, set the sSFR to the smallest aperture with nonzero sSFR
                for j in range(len(radii)-1):  
                    if ssfrs[j][i] > 0:
                        ssfr.append(ssfrs[j][i])
                        break
                else:
                    ssfr.append(ssfrs[-1][i])
            else:
                for j in range(len(radii)-1):              # If the sSFR in the corresponding aperture is zero, set the sSFR to the smallest aperture
                    if half_mass_radius[i] < radii[j+1]:   # with nonzero sSFR that is larger than the current aperture
                        if ssfrs[j][i] == 0:
                            for k in range(j+1, len(ssfrs)-1):
                                if ssfrs[k][i] > 0:
                                    ssfr.append(ssfrs[k][i])
                                    break
                            else:
                                ssfr.append(ssfrs[-1][i])
                            break
                        else:                                                                        # Use interpolation between apertures to calculate the sSFR
                            x = (half_mass_radius[i] - radii[j]) / (radii[j+1] - radii[j])
                            sfr = 10**(x*np.log10(sfrs[j+1][i])+(1-x)*np.log10(sfrs[j][i]))
                            smass = 10**(x*np.log10(smasses[j+1][i])+(1-x)*np.log10(smasses[j][i]))
                            ssfr.append(sfr/smass*1e9)
                        break
                else:
                    raise Exception()

        self.ssfr = np.zeros(len(self.smass))
        for i, index in enumerate(self.indices):
            self.ssfr[index] = ssfr[i]
        print(f'sSFR half mass calculation: {time.time()-time0:.1f} s')

        fiber_angular_radius = (3*unyt.arcsec).to('rad').value / 2
        delta_t = (self.data.metadata.cosmology.age(0) - self.data.metadata.cosmology.age(self.z)).value*unyt.Gyr
        distance = (delta_t * unyt.c).to('kpc')
        fiber_aperture = (fiber_angular_radius * distance).value
        if fiber_aperture < radii[0]:
            print('aperture < 3 kpc', fiber_aperture)
            ssfr_fiber = ssfrs[0]
        else:
            for j in range(len(radii)-1):
                if fiber_aperture < radii[j+1]:
                    x = (fiber_aperture - radii[j]) / (radii[j+1] - radii[j])
                    sfr_fiber = 10**(x*np.log10(sfrs[j+1])+(1-x)*np.log10(sfrs[j]))
                    smass_fiber = 10**(x*np.log10(smasses[j+1])+(1-x)*np.log10(smasses[j]))
                    ssfr_fiber = sfr_fiber/smass_fiber*1e9
                    break
        self.ssfr_fiber = np.zeros(len(self.smass))
        for i, index in enumerate(self.indices):
            self.ssfr_fiber[index] = ssfr_fiber[i]
                        
    def interacting_sample(self):
        #This piece of code creates the interacting galaxy sample.
        #The code also creates an array of isolated galaxy candidates, but the final selection is done later.

        mask_int1 = self.smass[self.indices] > 10**(self.mass_cutoff[0]+.05)                            # Mstar threshold
        mask_int2 = self.smass[self.indices] < 10**(self.mass_cutoff[1]-.05)                            # Mstar maximum     
        mask_int3 = self.sfr[self.indices] >= self.sfr_thresholds[0]                                    # SFR threshold
        mask_int4 = self.ssfr[self.indices] >= self.sfr_thresholds[1]                                   # sSFR threshold
        mask_int5 = self.smass[self.indices] > 0.1*self.smass[self.interacting_indices_local[:, 0]]     # mu < 10
        mask_int6 = self.distanceNN[:, 1] < 1.5                                                         # At least 2 galaxies within 1.5 Mpc
        
        mask_int = mask_int1 & mask_int2 & mask_int3 & mask_int4 & mask_int5 & mask_int6
        
        mask_iso1 = self.smass[self.indices] > 10**self.mass_cutoff[0]                                  # Mstar threshold
        mask_iso2 = self.smass[self.indices] < 10**self.mass_cutoff[1]                                  # Mstar maximum
        mask_iso3 = self.sfr[self.indices] >= self.sfr_thresholds[0]                                    # SFR threshold
        mask_iso4 = self.ssfr[self.indices] >= self.sfr_thresholds[1]                                   # sSFR threshold
        mask_iso5 = self.smass[self.indices] > 0.1*self.smass[self.interacting_indices_local[:, 0]]     # mu < 10
        mask_iso6 = self.distanceNN[:, 0] < 2                                                           # At least 1 galaxy within 2 Mpc
        
        self.mask_iso = mask_iso1 & mask_iso2 & mask_iso3 & mask_iso4 & mask_iso5 & mask_iso6
        self.interacting_local = np.arange(len(mask_int))[mask_int]
        
        print('Host galaxies:', len(self.interacting_local))
        print('Control group candidates:', np.sum(self.mask_iso))

    @staticmethod
    def weight(y, y0, tolerance):
        return 1 - np.abs(y-y0) / tolerance

    def weights(self, index_interacting, mask_isolated, x):
        w1 = self.__class__.weight(np.log10(self.smass[self.indices[index_interacting]]), np.log10(self.smass[self.indices[mask_isolated]]), x/2)
        w2 = self.__class__.weight(self.N2_local[index_interacting], self.N2_local[mask_isolated], x*self.N2_local[index_interacting])
        w3 = self.__class__.weight(
            self.distanceNN[index_interacting, 1], self.distanceNN[mask_isolated, 0], x*self.distanceNN[index_interacting, 1]
        )
        return w1*w2*w3

    def isolated_sample(self):
        time0 = time.time()
        no_match, isolated, tolerance = [], [], []
        for i in self.interacting_local:
            x = 0.1
            while True:
                mask1 = (10**(-x/2)*self.smass[self.indices[i]] < self.smass[self.indices]) & \
                        (10**(x/2)*self.smass[self.indices[i]] > self.smass[self.indices])
                mask2 = ((1-x)*self.N2_local[i] < self.N2_local) & ((1+x)*self.N2_local[i] > self.N2_local)
                mask3 = ((1-x)*self.distanceNN[i,1] < self.distanceNN[:, 0]) & ((1+x)*self.distanceNN[i,1] > self.distanceNN[:, 0])
                mask = mask1&mask2&mask3&self.mask_iso
                mask[i] = False
                for j in range(2):
                    if np.isin(self.interacting_indices_local[i,j], self.indices[mask]):
                        mask[np.where(self.interacting_indices_local[i,j]==self.indices)[0][0]] = False
                if np.sum(mask) >= 1:
                    candidates = self.indices[mask]
                    weight = self.weights(i, mask, x)
                    isolated.append(candidates[np.argmax(weight)])
                    tolerance.append(x)
                    break
                else:
                    if x == .2:
                        no_match.append(i)
                        break
                    x += .05

        mask_match = ~np.isin(self.interacting_local, no_match)
        interacting_matched = self.interacting_local[mask_match]
        self.interacting = self.indices[interacting_matched]
        self.isolated = isolated
        self.tolerance = tolerance
        print('Matched pairs:', len(self.interacting))
        print(f'{time.time()-time0:.1f} s')

    def save_file(self, file, sample):
        path1 = self.path+self.snapshot
        if not os.path.exists(path1):
            os.system('mkdir '+path1)
        path2 = path1 + '/' + self.tag + '/'
        if not os.path.exists(path2):
            os.system('mkdir '+path2)
        
        dictionary = {
            'smass': self.smass[sample],
            'gmass': self.gmass[sample],
            'halo_centers': self.halo_centers[sample],
            'sfr': self.sfr[sample],
            'ssfr': self.ssfr[sample],
            'ssfr_fiber': self.ssfr_fiber[sample],
            'hmass': self.hmass[sample],
            'is_central': self.is_central[sample],
            'v': self.v[sample],
            'r': self.r[sample],
            'r2': self.r2[sample],
            'N2': self.N2[sample],
            'bh_luminosity': bh_luminosity(self.bhar[sample]),
            'sbhar': sbhar(self.bhmass[sample], self.bhar[sample]),
            'shmr': self.stellar_half_mass_radius[sample],
            'tolerance': self.tolerance,
            'mass_ratio': self.smass[sample] / self.smass[self.interacting_indices[sample]],
            'boxsize': self.boxsize,
            'redshift': self.z,
            'ssfr1': self.ssfr1[sample],
            'ssfr3': self.ssfr3[sample],
            'ssfr10': self.ssfr10[sample],
            'ssfr30': self.ssfr30[sample],
            'ssfr50': self.ssfr50[sample],
            'ssfr1p': (self.data.projected_aperture_1kpc_projx.star_formation_rate[sample] / self.data.projected_aperture_1kpc_projx.stellar_mass[sample]).to('1/Gyr'),
            'ssfr3p': (self.data.projected_aperture_3kpc_projx.star_formation_rate[sample] / self.data.projected_aperture_3kpc_projx.stellar_mass[sample]).to('1/Gyr'),
            'ssfr10p': (self.data.projected_aperture_10kpc_projx.star_formation_rate[sample] / self.data.projected_aperture_10kpc_projx.stellar_mass[sample]).to('1/Gyr'),
            'ssfr30p': (self.data.projected_aperture_30kpc_projx.star_formation_rate[sample] / self.data.projected_aperture_30kpc_projx.stellar_mass[sample]).to('1/Gyr'),
            'ssfr50p': (self.data.projected_aperture_50kpc_projx.star_formation_rate[sample] / self.data.projected_aperture_50kpc_projx.stellar_mass[sample]).to('1/Gyr'),
        }
        try:
            dictionary['ssfr3avg10'] = (self.data.exclusive_sphere_3kpc.averaged_star_formation_rate[sample, 1] / self.smass3[sample]).to('1/Gyr')
            dictionary['ssfr3avg100'] = (self.data.exclusive_sphere_3kpc.averaged_star_formation_rate[sample, 0] / self.smass3[sample]).to('1/Gyr')
        except AttributeError:
            print('Unavailable in snipshot')
        np.savez(path2+file, **dictionary)

def add_univ(snapshot, size, resolution, masses=[10, 12], sfr=0, ssfr=0):
    size = str(size)
    simulation = 'L'+size+resolution
    tag = 's'+str(ssfr)
    if masses != [10, 12]:
        tag += '_'+str(masses[0])+str(masses[1])
    if sfr != 0:
        tag +=f'_sfr{sfr}'
    home = '/cosma/home/do019/dc-vanz1/'
    npy_file = f'{home}GalaxyPairs/3D{cutoff_min_smass}/L{size}{resolution}/{snapshot}.npy'
    hdf5_file = home+'hdf5_links/L'+size+resolution+'/halo_properties_'+snapshot+'.hdf5'
    univs[simulation+'_'+snapshot] = Analyse(
        hdf5_file, npy_file, simulation, snapshot, tag, mass_cutoff = masses, sfr_thresholds = [sfr, ssfr]
    )

if __name__ == '__main__':
    univs = {}
    
    snapshot = sys.argv[1]
    boxsize = sys.argv[2]
    resolution = sys.argv[3]
    cutoff_min_smass = sys.argv[4]
    
    add_univ(snapshot, boxsize, resolution)
    add_univ(snapshot, boxsize, resolution, ssfr=0.01)
    if cutoff_min_smass == '8':
        add_univ(snapshot, boxsize, resolution, masses=[8, 12])
        add_univ(snapshot, boxsize, resolution, ssfr=0.01, masses=[8, 12])
