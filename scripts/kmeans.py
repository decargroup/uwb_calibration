# %%
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.computecorrectedrange import ComputeCorrectedRange
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use('Qt5Agg')

sns.set_theme()

def append_dfs(df):
    df_all = df_all = pd.concat(df,ignore_index=True).copy()
    df_all['bias'] = df_all['calib_range'] - df_all['gt']
    df_all['fpp_avg'] = 0.5 * (df_all['fpp1'] + df_all['fpp2'])
    df_all['rxp_avg'] = 0.5 * (df_all['rxp1'] + df_all['rxp2'])
    df_all['std_avg'] = 0.5 * (df_all['std1'] + df_all['std2'])

    return df_all

def plot_histograms(df, labels, classes, show=True):
    fig, axs = plt.subplots(4,1)

    for i,class_name in enumerate(classes):    
        idx = labels==class_name
        axs[0].hist(df['bias'][idx],\
                    bins=np.linspace(-0.25,1,100), 
                    density=True, alpha=0.5, 
                    label=r'Cluster ' + str(i))

        axs[1].hist(df['fpp_avg'][idx], 
                    bins=np.linspace(-100,-78,100), 
                    density=True, 
                    alpha=0.5)

        axs[2].hist(df['rxp_avg'][idx], 
                    bins=np.linspace(-82,-77,100), 
                    density=True, 
                    alpha=0.5)
    
        axs[3].hist(df['std_avg'][idx], 
                    bins=np.linspace(25,90,66), 
                    density=True, 
                    alpha=0.5)
    

    axs[0].set_xlabel(r'Bias [m]')
    axs[1].set_xlabel(r'Average FPP [dBm]')
    axs[2].set_xlabel(r'Average RXP [dBm]')
    axs[3].set_xlabel(r'Average STD [?]')
    axs[0].legend()
    

    if show:
        plt.show(block=True)

# %%
# Read ranging data
tag_ids={'ifo001': [1,2],
         'ifo002': [3,4],
         'ifo003': [5,6]}
moment_arms={'ifo001': [np.idx([0.13189,-0.17245,-0.05249]), 
                        np.array([-0.17542,0.15712,-0.05307])],
             'ifo002': [np.array([0.16544,-0.15085,-0.03456]), 
                        np.array([-0.15467,0.16972,-0.01680])],
             'ifo003': [np.array([0.16685,-0.18113,-0.05576]), 
                        np.array([-0.13485,0.15468,-0.05164])]}
raw_obj = PostProcess("datasets/2022_08_03/bias_calibration_new2/merged.bag",
                      tag_ids,
                      moment_arms,
                      num_meas=-1)

# %%
# Calibrate
df = {}
correct_range = ComputeCorrectedRange(in_ns=True, filename="calib_results_new.pickle")
for pair in raw_obj.ts_data:
    # Copy data to a new dataframe
    df[pair] = pd.DataFrame(raw_obj.ts_data[pair],
                            columns=['t_nsecs',
                                     't_secs', 
                                     'range', 
                                     'tx1', 
                                     'rx1', 
                                     'tx2', 
                                     'rx2', 
                                     'tx3', 
                                     'rx3', 
                                     'fpp1', 
                                     'fpp2', 
                                     'rxp1', 
                                     'rxp2', 
                                     'std1', 
                                     'std2',])
    df[pair]['gt'] = raw_obj.time_intervals[pair]['r_gt']

    # Calibrate range measurements
    n = len(df[pair])
    df[pair]["from_id"] = pair[0]*np.ones((n,))
    df[pair]["to_id"] = pair[1]*np.ones((n,))
    df_calib = correct_range.get_corrected_range(df[pair])

    df[pair]["calib_range"] = df_calib["range"]
    df[pair]["calib_std"] = df_calib["std"]

df_all = append_dfs(df).copy()

# %% Drop outliers
df_all.drop(df_all[np.abs(df_all['bias'])>3].index, inplace=True)

# %%
# Fit K-means model
n_clusters=15

df_scale = df_all.copy()
# df_scale['bias'] = np.abs(df_scale['bias'])
training_data = np.array(df_scale.loc[:,['fpp_avg', 'rxp_avg', 'std_avg']])
training_data -= np.mean(training_data,axis=0)
training_data = normalize(training_data,axis=0)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(training_data)

# %%
# Visualize K-means clusters
plot_histograms(df_all, kmeans.labels_, np.linspace(0,n_clusters-1,n_clusters))
# %%
