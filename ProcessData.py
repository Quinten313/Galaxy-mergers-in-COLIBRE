import numpy as np
from scipy.stats import binned_statistic

# -----------LOADING THE DATA--------------

# Loads in all properties from the data file
def LoadData(sample, path):
    data = np.load(path)
    for key in data:
        sample[key] = data[key]

# Calls LoadData() for the host and control galaxy sample
def AddSample(run, path):
    run['interacting'] = {}
    LoadData(run['interacting'], path + 'Interacting.npz')
    run['isolated'] = {}
    LoadData(run['isolated'], path + 'Isolated.npz')

# Loads in a specific run
def AddRun(simulation, snapshot, run):
    try:
        simulation[snapshot][run] = {}
    except:
        simulation[snapshot] = {}
        simulation[snapshot][run] = {}
    path = '/cosma/home/do019/dc-vanz1/Runs'+simulation['dimension']+'/'+simulation['simulation']+'/'+snapshot+'/'+run+'/'
    AddSample(simulation[snapshot][run], path)


#------------COMBINING SNAPSHOTS-------------

# Combines snapshots for better statistics, returns an array containing property x of all galaxies in the input snapshots
def Add(simulation, snapshots, run, sample, x):
    big_array = []
    for i in range(len(snapshots)):
        big_array = np.concatenate([big_array, simulation[snapshots[i]][run][sample][x]])
    return big_array

def CombineSnapshotsMasked(simulation, snapshots, run, x, y, key, maximum, inversed=False):
    x_int, x_iso, y_int, y_iso = [], [], [], []
    for s in snapshots:
        mask =  simulation[s][run]['interacting'][key] < maximum
        x_int = np.concatenate([x_int, simulation[s][run]['interacting'][x][mask]])
        x_iso = np.concatenate([x_iso, simulation[s][run]['isolated'][x][mask]])
        y_int = np.concatenate([y_int, simulation[s][run]['interacting'][y][mask]])
        y_iso = np.concatenate([y_iso, simulation[s][run]['isolated'][y][mask]])
    return x_int, x_iso, y_int, y_iso


#-------------PLOTTING THE DATA--------------

# Generates bins in linear or logarithmic space
def Bins(bins, x_min, x_max, log_bins):
    bin_edges = np.linspace(x_min, x_max, bins+1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if log_bins:
        return 10**bin_edges, 10**bin_centers
    return bin_edges, bin_centers

# Calculates the binned mean of property y, along with its error and bin centers as a function of x
def CalcMean(x, y, bins, x_min, x_max, log_bins):
    bin_edges, bin_centers = Bins(bins, x_min, x_max, log_bins)
    
    mean = binned_statistic(x=x, values=y, statistic = 'mean', bins=bin_edges)[0]
    std = binned_statistic(x=x, values=y, bins=bin_edges, statistic='std')[0]
    count = binned_statistic(x=x, values=y, bins=bin_edges, statistic='count')[0]

    err = std/count**.5
    return mean, err, bin_centers

# Plot the binned mean of property y, along with its error as a function of x
def PlotMean(ax, x, y, bins, x_min, x_max, log_bins = False, c='black', label=None):
    mean, err, bin_centers = CalcMean(x, y, bins, x_min, x_max, log_bins)

    ax.plot(bin_centers, mean, label = label, c = c)
    ax.fill_between(bin_centers, mean-err, mean+err, alpha = .3, color = c)
    ax.set_xlim(x_min, x_max)

# Calculates the ratio of 2 binned means of property y, along with its error and bin centers as a function of x
def CalcMeanRatio(x, y_int, y_iso, bins, x_min, x_max, log_bins):
    mean_int, err_int, bin_centers = CalcMean(x, y_int, bins, x_min, x_max, log_bins)
    mean_iso, err_iso, _ = CalcMean(x, y_iso, bins, x_min, x_max, log_bins)

    ratio = mean_int / mean_iso
    err_ratio = ((err_int/mean_iso)**2+(mean_int*err_iso/mean_iso**2)**2)**.5
    return ratio, err_ratio, bin_centers

# Plots the ratio of 2 binned means of property y, along with its error as a function of x
def PlotMeanRatio(ax, x, y_int, y_iso, bins, x_min, x_max, log_bins = False, add_line=True, c='black', label=None):
    ratio, err_ratio, bin_centers = CalcMeanRatio(x, y_int, y_iso, bins, x_min, x_max, log_bins)
    
    ax.plot(bin_centers, ratio, label = label, c = c)
    if add_line:
        ax.axhline(1, color='grey', linestyle='--', label='No enhancement', alpha = .7)

    ax.fill_between(bin_centers, ratio-err_ratio, ratio+err_ratio, alpha = .3, color = c)
    ax.set_xlim(x_min, x_max)

# Calculates the binned median of property y, along with its percentiles and bin centers as a function of x
def CalcMedian(x, y, bins, x_min, x_max, log_bins):
    bin_edges, bin_centers = Bins(bins, x_min, x_max, log_bins)
    
    median = binned_statistic(x=x, values=y, statistic = 'median', bins=bin_edges)[0]
    percentile16 = binned_statistic(x=x, values=y, bins=bin_edges, statistic=lambda x: np.percentile(x, 16))[0]
    percentile84 = binned_statistic(x=x, values=y, bins=bin_edges, statistic=lambda x: np.percentile(x, 84))[0]

    return median, percentile16, percentile84, bin_centers

# Plot the binned median of property y, along with its percentiles as a function of x
def PlotMedian(ax, x, y, bins, x_min, x_max, log_bins = False, c='black', label=None):
    median, percentile16, percentile84, bin_centers = CalcMedian(x, y, bins, x_min, x_max, log_bins)

    ax.plot(bin_centers, median, label = label, c = c)
    ax.fill_between(bin_centers, percentile16, percentile84, alpha = .3, color = c)
    ax.set_xlim(x_min, x_max)
    if log_bins:
        ax.set_xlim(10**x_min, 10**x_max)

# Calculates the ratio of 2 binned medians of property y, along with the bin centers as a function of x
def CalcMedianRatio(x, y_int, y_iso, bins, x_min, x_max, log_bins):
    median_int, _, _, bin_centers = CalcMedian(x, y_int, bins, x_min, x_max, log_bins)
    median_iso, _, _, _ = CalcMedian(x, y_iso, bins, x_min, x_max, log_bins)

    ratio = median_int / median_iso
    return ratio, bin_centers
