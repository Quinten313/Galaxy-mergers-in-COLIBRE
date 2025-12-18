import numpy as np
from scipy.stats import binned_statistic
import os

#------------GENERAL FUNCTIONS-------------

def calc_distances(coordinate_array1, coordinate_array2, boxsize):
    distances_wrapped_3D = np.abs(coordinate_array1 - coordinate_array2)
    distances_wrapped_3D -= 0.5 * boxsize
    distances_wrapped_3D = 0.5 * boxsize - np.abs(distances_wrapped_3D)
    distances_wrapped = np.sum(distances_wrapped_3D**2, axis=1)**.5
    return distances_wrapped
    

# -----------LOADING THE DATA--------------

# Loads in all properties from the data file
def load_data(sample, path):
    data = np.load(path)
    for key in data:
        sample[key] = data[key]

# Calls LoadData() for the host and control galaxy sample
def add_sample(run, path):
    run['interacting'] = {}
    load_data(run['interacting'], path + 'Interacting.npz')
    run['isolated'] = {}
    load_data(run['isolated'], path + 'Isolated.npz')
    run['secondary'] = {}
    load_data(run['secondary'], path + 'Secondary.npz')

# Loads in a specific run
def add_run(simulation, snapshot, run):
    try:
        simulation[snapshot][run] = {}
    except:
        simulation[snapshot] = {}
        simulation[snapshot][run] = {}
    path = f'/cosma8/data/do019/dc-vanz1/GalaxyProperties/{simulation['dimension']}/{simulation['simulation']}/{snapshot}/{run}/'
    add_sample(simulation[snapshot][run], path)

# Loads in all available snapshots of a list of runs for one simulational volume
def add_all_runs(simulation, runs):
    snapshots = os.listdir(f'/cosma8/data/do019/dc-vanz1/GalaxyProperties/{simulation['dimension']}/{simulation['simulation']}/')
    print('Available snapshots:', sorted(snapshots))
    for snapshot in snapshots:
        for run in runs:
            add_run(simulation, snapshot, run)


#------------COMBINING SNAPSHOTS-------------

# Combines snapshots for better statistics, returns an array containing property x of all galaxies in the input snapshots
def add(simulation, snapshots, run, sample, x):
    big_array = simulation[snapshots[0]][run][sample][x]
    if len(snapshots) > 1:
        for snapshot in snapshots[1:]:
            big_array = np.concatenate([big_array, simulation[snapshot][run][sample][x]])
    return big_array

def combine_snapshots_masked(simulation, snapshots, run, x, y, key, maximum, inversed=False):
    x_int, x_iso, y_int, y_iso = [], [], [], []
    for s in snapshots:
        mask =  simulation[s][run]['interacting'][key] < maximum
        x_int = np.concatenate([x_int, simulation[s][run]['interacting'][x][mask]])
        x_iso = np.concatenate([x_iso, simulation[s][run]['isolated'][x][mask]])
        y_int = np.concatenate([y_int, simulation[s][run]['interacting'][y][mask]])
        y_iso = np.concatenate([y_iso, simulation[s][run]['isolated'][y][mask]])
    return x_int, x_iso, y_int, y_iso

def apply_mask(simulation, snapshots, run, sample, key, edges):
    snapshots_combined = add(simulation, snapshots, run, sample, key)
    return (snapshots_combined >= edges[0]) & (snapshots_combined < edges[1])

def mask_overlapping_half_mass_radii(simulation, snapshots, run):
    centers_hosts = add(simulation, snapshots, run, 'interacting', 'halo_centers')
    centers_secondaries = add(simulation, snapshots, run, 'secondary', 'halo_centers')
    boxsize = simulation[snapshots[0]][run]['interacting']['boxsize']
    distances = calc_distances(centers_hosts, centers_secondaries, boxsize)

    shmr_host = add(simulation, snapshots, run, 'interacting', 'shmr')
    shmr_secondary = add(simulation, snapshots, run, 'secondary', 'shmr')
    return ~(distances < shmr_host + shmr_secondary)


#-------------PLOTTING THE DATA--------------

# Generates bins in linear or logarithmic space
def set_bins(bins, x_min, x_max, log_bins):
    bin_edges = np.linspace(x_min, x_max, bins+1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if log_bins:
        return 10**bin_edges, 10**bin_centers
    return bin_edges, bin_centers

# Calculates the number of counts in each bin
def calc_counts(x, bins, x_min, x_max, log_bins=False):
    bin_edges, bin_centers = set_bins(bins, x_min, x_max, log_bins)
    count = binned_statistic(x=x, values=x, bins=bin_edges, statistic='count')[0]
    return bin_centers, count

# Plot the counts in each bin
def plot_counts(ax, x, bins, x_min, x_max, log_bins = False, c='black', label=None):
    bin_centers, count = calc_counts(x, bins, x_min, x_max, log_bins)

    ax.plot(bin_centers, count, label = label, c = c)
    ax.set_xlim(x_min, x_max)

# Calculates the binned mean of property y, along with its bootstrapping error and bin centers as a function of x
def calc_mean(x, y, bins, x_min, x_max, log_bins=False, N=100):
    bin_edges, bin_centers = set_bins(bins, x_min, x_max, log_bins)
    
    mean, _, indices = binned_statistic(x=x, values=y, statistic = 'mean', bins=bin_edges)
    
    bootstrapping_error = []
    for i in range(1, len(bin_edges)):
        y_in_bin = y[indices == i]
        if len(y_in_bin) < 2:
            bootstrapping_error.append(np.nan)
            continue
        resamples = np.random.randint(0, len(y_in_bin), [N, len(y_in_bin)])
        resampled_means = [np.mean(y_in_bin[resample]) for resample in resamples]
        bootstrapping_error.append(np.std(resampled_means))
    return mean, bootstrapping_error, bin_centers

# Plot the binned mean of property y, along with its error as a function of x
def plot_mean(ax, x, y, bins, x_min, x_max, log_bins=False, N=100, c='black', label=None):
    mean, err, bin_centers = calc_mean(x, y, bins, x_min, x_max, log_bins, N)

    ax.plot(bin_centers, mean, label = label, c = c)
    ax.fill_between(bin_centers, mean-err, mean+err, alpha = .3, color = c)
    ax.set_xlim(x_min, x_max)

# Calculates the ratio of 2 binned means of property y, along with its bootstrapping error and bin centers as a function of x
def calc_mean_ratio(x, y_int, y_iso, bins, x_min, x_max, log_bins=False, N=100):
    bin_edges, bin_centers = set_bins(bins, x_min, x_max, log_bins)

    mean_int, _, indices = binned_statistic(x=x, values=y_int, statistic = lambda x: np.nanmean(x), bins=bin_edges)
    mean_iso, _, _ = binned_statistic(x=x, values=y_iso, statistic = lambda x: np.nanmean(x), bins=bin_edges)
    ratio = mean_int / mean_iso
    
    bootstrapping_error = []
    for i in range(1, len(bin_edges)):
        y_int_in_bin = y_int[indices == i]
        y_iso_in_bin = y_iso[indices == i]
        if len(y_int_in_bin) < 2:
            bootstrapping_error.append(np.nan)
            continue
        resamples = np.random.randint(0, len(y_int_in_bin), [N, len(y_int_in_bin)])
        resampled_means_int = [np.mean(y_int_in_bin[resample]) for resample in resamples]
        resampled_means_iso = [np.mean(y_iso_in_bin[resample]) for resample in resamples]
        resampled_ratios = np.array(resampled_means_int) / np.array(resampled_means_iso)
        bootstrapping_error.append(np.std(resampled_ratios))
        
    return ratio, bootstrapping_error, bin_centers

# Plots the ratio of 2 binned means of property y, along with its error as a function of x
def plot_mean_ratio(ax, x, y_int, y_iso, bins, x_min, x_max, log_bins=False, N=100, add_line=True, c='black', label=None, show_error=True):
    ratio, err_ratio, bin_centers = calc_mean_ratio(x, y_int, y_iso, bins, x_min, x_max, log_bins, N)
    
    ax.plot(bin_centers, ratio, label = label, c = c)
    if add_line:
        ax.axhline(1, color='grey', linestyle='--', label='No enhancement', alpha = .7)
    if show_error:
        ax.fill_between(bin_centers, ratio-err_ratio, ratio+err_ratio, alpha = .3, color = c)
    ax.set_xlim(x_min, x_max)

# Calculates the binned median of property y, along with its percentiles and bin centers as a function of x
def calc_median(x, y, bins, x_min, x_max, log_bins=False):
    bin_edges, bin_centers = set_bins(bins, x_min, x_max, log_bins)
    
    median = binned_statistic(x=x, values=y, statistic = 'median', bins=bin_edges)[0]
    percentile16 = binned_statistic(x=x, values=y, bins=bin_edges, statistic=lambda x: np.percentile(x, 16))[0]
    percentile84 = binned_statistic(x=x, values=y, bins=bin_edges, statistic=lambda x: np.percentile(x, 84))[0]

    return median, percentile16, percentile84, bin_centers

# Plot the binned median of property y, along with its percentiles as a function of x
def plot_median(ax, x, y, bins, x_min, x_max, log_bins=False, c='black', label=None):
    median, percentile16, percentile84, bin_centers = calc_median(x, y, bins, x_min, x_max, log_bins)

    ax.plot(bin_centers, median, label = label, c = c)
    ax.fill_between(bin_centers, percentile16, percentile84, alpha = .3, color = c)
    ax.set_xlim(x_min, x_max)
    if log_bins:
        ax.set_xlim(10**x_min, 10**x_max)

# Calculates the ratio of 2 binned medians of property y, along with the bin centers as a function of x
def calc_median_ratio(x, y_int, y_iso, bins, x_min, x_max, log_bins):
    median_int, _, _, bin_centers = calc_median(x, y_int, bins, x_min, x_max, log_bins)
    median_iso, _, _, _ = calc_median(x, y_iso, bins, x_min, x_max, log_bins)

    ratio = median_int / median_iso

    return ratio, bin_centers

# Plots the ratio of 2 binned medians of property y as a function of x
def plot_median_ratio(ax, x, y_int, y_iso, bins, x_min, x_max, log_bins=False, add_line=True, c='black', label=None):
    ratio, bin_centers = calc_median_ratio(x, y_int, y_iso, bins, x_min, x_max, log_bins)
    
    ax.plot(bin_centers, ratio, label = label, c = c)
    if add_line:
        ax.axhline(1, color='grey', linestyle='--', label='No enhancement', alpha = .7)
    ax.set_xlim(x_min, x_max)