import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def multi_fractal(x, d1, d2, c, t):
    y = np.piecewise(x, [x <t, x>=t],
                     [lambda x: -d1*x+c, lambda x: -d2*(x-t)-d1*t+c])
    return y

def calculate_fractal_dimension(channel_map, max_size, random_baseline=False):
    channel_mask = channel_map.flatten()
    box_sizes = np.arange(1, int(max_size)+1)
    box_counts = np.zeros(box_sizes.shape)
    if random_baseline:
        channel_pixels = channel_mask.sum()
        random_channel_mask = np.zeros(channel_map.size)
        random_counts = np.zeros(box_sizes.shape)
        indices = (np.random.sample(int(channel_pixels))*len(channel_mask)).astype(int)
        random_channel_mask[indices] = 1
        random_channel_map = random_channel_mask.reshape(channel_map.shape)
    for i, k in enumerate(box_sizes):
        box_covering = np.add.reduceat(np.add.reduceat(channel_map,np.arange(0, channel_map.shape[0],k), axis=0),
                                       np.arange(0, channel_map.shape[1], k), axis=1)
        box_counts[i] = len(np.where(box_covering > 0)[0])
        if random_baseline:
            random_box_covering = np.add.reduceat(np.add.reduceat(random_channel_map, np.arange(0, random_channel_map.shape[0],k), axis=0),
                                                                  np.arange(0, random_channel_map.shape[1], k), axis=1)
            random_counts[i] = len(np.where(random_box_covering > 0)[0])
    if random_baseline:
        return box_sizes, box_counts, random_counts
    else:
        return box_sizes, box_counts

def get_fractals(grid, field, threshold, max_size, random_baseline=True):
    channel_mask = np.where(grid.at_node[field]>=threshold, 1, 0)
    channel_map = channel_mask.reshape(grid.shape)
    return calculate_fractal_dimension(channel_map, max_size, random_baseline)

def fit_fractals(grid, field, samples=100, space_function=np.linspace):
    max_size = int(min(grid.shape)/4)
    minimum = grid.at_node[field][np.nonzero(grid.at_node[field])].min()
    #minimum = grid.at_node[field].min()
    thresholds = space_function(minimum, grid.at_node[field].max(), samples)
    sizes = []
    counts = []
    types = []
    d1s = []
    d2s = []
    ts = []
    cs = []
    fields = []
    threshold_vs = []
    for threshold in thresholds:
        box_sizes, box_counts, random_counts = get_fractals(grid, field, threshold, max_size, True)
        d1, d2, c, t = curve_fit(multi_fractal, np.log(box_sizes), np.log(box_counts))[0]
        rd1, rd2, rc, rt = curve_fit(multi_fractal, np.log(box_sizes), np.log(random_counts))[0]
        sizes += list(box_sizes)
        counts += list(box_counts)
        d1s += [d1]*len(box_counts)
        d2s += [d2]*len(box_counts)
        ts  += [t]*len(box_counts)
        cs += [c]*len(box_counts)
        fields += [field]*len(box_counts)
        types += ['true']*len(box_counts)
        threshold_vs += [threshold]*len(box_counts)
        sizes += list(box_sizes)
        counts += list(random_counts)
        d1s += [rd1]*len(box_counts)
        d2s += [rd2]*len(box_counts)
        ts += [rt]*len(box_counts)
        cs += [rc]*len(box_counts)
        fields += [field]*len(box_counts)
        types += ['random']*len(box_counts)
        threshold_vs += [threshold]*len(box_counts)
    return sizes, counts, types, d1s, d2s, ts, cs, fields, threshold_vs

def make_fractal_df(grid, fields, samples=100, space_function=np.linspace):
    sizes = []
    counts = []
    types = []
    d1s = []
    d2s = []
    ts = []
    cs = []
    field_vs = []
    thresholds = []
    for field in fields:
        new_sizes, new_counts, new_types, new_d1s, new_d2s, new_ts, new_cs, new_fields, new_thresholds = fit_fractals(grid, field, samples, space_function)
        sizes += new_sizes
        counts += new_counts
        types += new_types
        d1s += new_d1s
        d2s += new_d2s
        ts += new_ts
        cs += new_cs
        field_vs += new_fields
        thresholds += new_thresholds
    #print(len(sizes), len(counts), len(types), len(d1s), len(d2s), len(ts), len(thresholds), len(fields))
    df = pd.DataFrame({'box size': sizes, 'box_count': counts, 'type': types,
                       'd1': d1s, 'd2': d2s, 't': ts, 'c': cs, 'threshold': thresholds, 'field': field_vs})
    return df
