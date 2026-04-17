import numpy as np
from scipy.stats import binned_statistic
import os
import matplotlib.patheffects as pe
from matplotlib import colors

#------------GENERAL FUNCTIONS-------------

PATH_EFFECTS = [pe.Stroke(linewidth=3, foreground="k"), pe.Normal()]
LW = 2

def calc_distances(coordinate_array1, coordinate_array2, boxsize):
    distances_wrapped_3D = np.abs(coordinate_array1 - coordinate_array2)
    distances_wrapped_3D -= 0.5 * boxsize
    distances_wrapped_3D = 0.5 * boxsize - np.abs(distances_wrapped_3D)
    distances_wrapped = np.sum(distances_wrapped_3D**2, axis=1)**.5
    return distances_wrapped

# This function has as input the indices that require additional plotting constraints
# and outputs the start and end indices for each chain that can be used for plotting
def bins_with_neighbors(x, x_min, x_max, n_bins, log_bins, binmin):
    bin_edges = np.linspace(x_min, x_max, n_bins+1)
    if log_bins:
        bin_edges = 10**bin_edges
    count = np.histogram(x, bins=bin_edges)[0]
    indices = np.arange(n_bins)[count < binmin]
    is_neighbor = np.isin(np.arange(n_bins), (list((set(indices) | set(indices-1) | set(indices+1)) - set([-1]) - set([n_bins]))))
    chains = []
    if is_neighbor[0]:
        start = 0
    else:
        start = None
    for i in range(1, n_bins-1):
        if not is_neighbor[i] and is_neighbor[i-1]:
            chains.append([start, i])
        if is_neighbor[i] and not is_neighbor[i-1]:
            start = i
    if is_neighbor[-1]:
        chains.append([start, n_bins])
    chains = np.array(chains)
    chains_regular = []
    if np.min(chains) > 0:
        chains_regular.append([0, np.min(chains)+1])
    for i in range(len(chains)-1):
        chains_regular.append([chains[i, 1]-1, chains[i+1,0]+1])
    if np.max(chains) < n_bins-1:
        chains_regular.append([np.max(chains)-1, n_bins-1])
    return chains, np.array(chains_regular)
    

# -----------LOADING THE DATA--------------

# Loads in all properties from the data file
def load_data(sample, path):
    data = np.load(path, allow_pickle=True)
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
def add_run(simulation, snapshot, run, local_density=None):
    try:
        simulation[snapshot][run] = {}
    except:
        simulation[snapshot] = {}
        simulation[snapshot][run] = {}
    suffix = ''
    if local_density:
        suffix = '_'+local_density
    path = f'/cosma8/data/do019/dc-vanz1/GalaxyProperties/{simulation['dimension']}{suffix}/{simulation['simulation']}/{snapshot}/{run}/'
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
def plot_counts(ax, x, bins, x_min, x_max, log_bins=False, c='black', label=None):
    bin_centers, count = calc_counts(x, bins, x_min, x_max, log_bins)
    
    ax.plot(bin_centers, count, label=label, c=c)
    ax.set_xlim(x_min, x_max)
    if log_bins:
        ax.set(xlim=[10**x_min, 10**x_max], xscale='log')

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
def plot_mean(ax, x, y, bins, x_min, x_max, log_bins=False, N=100, c='black', label=None, binmin=None, linestyle='-'):
    mean, err, bin_centers = calc_mean(x, y, bins, x_min, x_max, log_bins, N)
    if binmin:
        bin_edges = np.linspace(x_min, x_max, bins+1)
        if log_bins:
            bin_edges = 10**bin_edges
        count = np.histogram(x, bins=bin_edges)[0]
        mask = count < binmin
        plot_mask = mask.copy()
        plot_mask[np.argmin(plot_mask==False)-1] = True
        if np.sum(mask) > 0:
            ax.plot(bin_centers[plot_mask], mean[plot_mask], c=c, linestyle=':')
            err = np.array(err)
            mean[mask], err[mask], bin_centers[mask] = np.nan, np.nan, np.nan
    
    ax.plot(bin_centers, mean, label=label, c=c, linestyle=linestyle)
    ax.fill_between(bin_centers, mean-err, mean+err, alpha=.3, color=c)
    ax.set_xlim(x_min, x_max)
    if log_bins:
        ax.set(xlim=[10**x_min, 10**x_max], xscale='log')

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
def plot_mean_ratio(ax, x, y_int, y_iso, bins, x_min, x_max, log_bins=False, N=100, add_line=True, c='black', label=None, show_error=True, binmin=None, linestyle='-'):
    ratio, err_ratio, bin_centers = calc_mean_ratio(x, y_int, y_iso, bins, x_min, x_max, log_bins, N)
    if binmin:
        bin_edges = np.linspace(x_min, x_max, bins+1)
        if log_bins:
            bin_edges = 10**bin_edges
        count = np.histogram(x, bins=bin_edges)[0]
        mask = count < binmin
        plot_mask = mask.copy()
        plot_mask[np.argmin(plot_mask==False)-1] = True
        if np.sum(mask) > 0:
            ax.plot(bin_centers[plot_mask], ratio[plot_mask], c=c, linestyle=':')
            err_ratio = np.array(err_ratio)
            ratio[mask], err_ratio[mask], bin_centers[mask] = np.nan, np.nan, np.nan
    ax.plot(bin_centers, ratio, label=label, c=c, linestyle=linestyle)
    if add_line:
        ax.axhline(1, color='grey', linestyle='--', label='No enhancement', alpha = .7)
    if show_error:
        ax.fill_between(bin_centers, ratio-err_ratio, ratio+err_ratio, alpha = .3, color=c)
    ax.set_xlim(x_min, x_max)
    if log_bins:
        ax.set(xlim=[10**x_min, 10**x_max], xscale='log')

# Calculates the binned median of property y, along with its percentiles and bin centers as a function of x
def calc_median(x, y, bins, x_min, x_max, log_bins=False):
    bin_edges, bin_centers = set_bins(bins, x_min, x_max, log_bins)
    
    median = binned_statistic(x=x, values=y, statistic = 'median', bins=bin_edges)[0]
    percentile16 = binned_statistic(x=x, values=y, bins=bin_edges, statistic=lambda x: np.percentile(x, 16))[0]
    percentile84 = binned_statistic(x=x, values=y, bins=bin_edges, statistic=lambda x: np.percentile(x, 84))[0]

    return median, percentile16, percentile84, bin_centers

# Plot the binned median of property y, along with its percentiles as a function of x
def plot_median(ax, x, y, bins, x_min, x_max, log_bins=False, c='black', label=None, binmin=None, show_error=True):
    median, percentile16, percentile84, bin_centers = calc_median(x, y, bins, x_min, x_max, log_bins)

    if binmin:
        chains1, chains2 = bins_with_neighbors(x, x_min, x_max, bins, log_bins, binmin)
        for (start, end) in chains1:
            ax.plot(bin_centers[start:end], median[start:end], c=c, linestyle=':')
        for i, (start, end) in enumerate(chains2):
            label_ = None
            if i == len(chains1)-1:
                label_ = label
            ax.plot(bin_centers[start:end], median[start:end], label=label_, c=c, path_effects=PATH_EFFECTS, lw=LW)
            if show_error:
                ax.fill_between(bin_centers[start:end], percentile16[start:end], percentile84[start:end], alpha=.3, color=c, zorder=-10)
    else:
        ax.plot(bin_centers, median, label=label, c=c, path_effects=PATH_EFFECTS, lw=LW)
        if show_error:
            ax.fill_between(bin_centers, percentile16, percentile84, alpha=.3, color=c, zorder=-10)
    ax.set_xlim(x_min, x_max)
    if log_bins:
        ax.set(xlim=[10**x_min, 10**x_max], xscale='log')

# Calculates the ratio of 2 binned medians of property y, along with the bin centers as a function of x
def calc_median_ratio(x, y_int, y_iso, bins, x_min, x_max, log_bins):
    median_int, _, _, bin_centers = calc_median(x, y_int, bins, x_min, x_max, log_bins)
    median_iso, _, _, _ = calc_median(x, y_iso, bins, x_min, x_max, log_bins)

    ratio = median_int / median_iso

    return ratio, bin_centers

# Plots the ratio of 2 binned medians of property y as a function of x
def plot_median_ratio(ax, x, y_int, y_iso, bins, x_min, x_max, log_bins=False, add_line=True, c='black', label=None, binmin=None):
    ratio, bin_centers = calc_median_ratio(x, y_int, y_iso, bins, x_min, x_max, log_bins)
    if binmin:
        chains1, chains2 = bins_with_neighbors(x, x_min, x_max, bins, log_bins, binmin)
        for (start, end) in chains1:
            ax.plot(bin_centers[start:end], ratio[start:end], c = c, linestyle=':')
        for i, (start, end) in enumerate(chains2):
            label_ = None
            if i == len(chains1)-1:
                label_ = label
            ax.plot(bin_centers[start:end], ratio[start:end], label=label_, c=c)
    else: 
        ax.plot(bin_centers, ratio, label=label, c=c)
    if add_line:
        ax.axhline(1, color='grey', linestyle='--', label='No enhancement', alpha = .7)
    ax.set_xlim(x_min, x_max)
    if log_bins:
        ax.set(xlim=[10**x_min, 10**x_max], xscale='log')


#-------------POLISHING PLOTS--------------

# Adds major and minor ticks to all borders
def polish(ax):
    ax.tick_params(
        which='both',
        top=True,
        right=True,
        direction='in'
    )
    ax.minorticks_on()

# Makes a blue legend
def legend(ax, loc=0, ncols=1):
    leg = ax.legend(
        loc=loc,
        ncols=ncols,
        fancybox=True,
        framealpha=1
    )
    frame = leg.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(1.2)
    frame.set_facecolor((0.9, 0.95, 1.0))

# Make a frameless legend
def legend_frameless(ax, loc=0, ncols=1):
    leg = ax.legend(
        loc=loc,
        ncols=ncols,
        frameon=False
    )

# Creates a red and blue colormap
def red_blue_colors(n):
    cmap = colors.LinearSegmentedColormap.from_list('', ['#bb0000', '#00aaff'])
    return cmap(np.linspace(0, 1, n))
