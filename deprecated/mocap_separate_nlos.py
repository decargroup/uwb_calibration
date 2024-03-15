# %%
from pyuwbcalib.postprocess import PostProcess
from deprecated.computecorrectedrange import ComputeCorrectedRange
from scipy.interpolate import interp1d, BSpline
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from pymlg import SO3 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use('Qt5Agg')

sns.set_theme()

def append_dfs(df):
    df_all = df_all = pd.concat(df,ignore_index=True).copy()
    df_all['bias'] = df_all['range'] - df_all['gt']
    df_all['fpp_avg'] = 0.5 * (df_all['fpp1'] + df_all['fpp2'])
    df_all['rxp_avg'] = 0.5 * (df_all['rxp1'] + df_all['rxp2'])
    df_all['std_avg'] = 0.5 * (df_all['std1'] + df_all['std2'])

    return df_all

def plot_histograms(df, los_idx, nlos1_idx, nlos2_idx, show=True):
    fig, axs = plt.subplots(4,1)
    axs[0].hist(df['bias'][los_idx],\
                bins=np.linspace(-0.25,1,100), 
                density=True, alpha=0.5, 
                label=r'LOS')
    axs[0].hist(df['bias'][nlos1_idx], 
                bins=np.linspace(-0.25,1,100), 
                density=True, 
                alpha=0.5, 
                label=r'NLOS - 1 obstacle')
    axs[0].hist(df['bias'][nlos2_idx], 
                bins=np.linspace(-0.25,1,100), 
                density=True, 
                alpha=0.5, 
                label=r'NLOS - 2 obstacles')
    axs[0].set_xlabel(r'Bias [m]')
    axs[0].legend()

    axs[1].hist(df['fpp_avg'][los_idx], 
                bins=np.linspace(-100,-78,100), 
                density=True, 
                alpha=0.5)
    axs[1].hist(df['fpp_avg'][nlos1_idx], 
                bins=np.linspace(-100,-78,100), 
                density=True, 
                alpha=0.5)
    axs[1].hist(df['fpp_avg'][nlos2_idx], 
                bins=np.linspace(-100,-78,100), 
                density=True, 
                alpha=0.5)
    axs[1].set_xlabel(r'Average FPP [dBm]')

    axs[2].hist(df['rxp_avg'][los_idx], 
                bins=np.linspace(-82,-77,100), 
                density=True, 
                alpha=0.5)
    axs[2].hist(df['rxp_avg'][nlos1_idx], 
                bins=np.linspace(-82,-77,100), 
                density=True, 
                alpha=0.5)
    axs[2].hist(df['rxp_avg'][nlos2_idx], 
                bins=np.linspace(-82,-77,100), 
                density=True, 
                alpha=0.5)
    axs[2].set_xlabel(r'Average RXP [dBm]')

    axs[3].hist(df['std_avg'][los_idx], 
                bins=np.linspace(25,90,66), 
                density=True, 
                alpha=0.5)
    axs[3].hist(df['std_avg'][nlos1_idx], 
                bins=np.linspace(25,90,66), 
                density=True, 
                alpha=0.5)
    axs[3].hist(df['std_avg'][nlos2_idx], 
                bins=np.linspace(25,90,66), 
                density=True, 
                alpha=0.5)
    axs[3].set_xlabel(r'Average STD [?]')

    fig, axs = plt.subplots(1,3,sharex='all',sharey='all')
    axs[0].set_ylabel(r'Bias [m]')
    axs[0].scatter(df['fpp_avg'][los_idx],
                df['bias'][los_idx], 
                # s=1, 
                label=r'LOS', 
                alpha=0.5)
    
    axs[1].set_xlabel(r'Average First Path Power [dBm]')
    axs[1].scatter(df['fpp_avg'][nlos1_idx],
                df['bias'][nlos1_idx], 
                # s=1, 
                label=r'NLOS - 1 obstacle', 
                alpha=0.5,
                color='darkorange')
    
    axs[2].scatter(df['fpp_avg'][nlos2_idx],
                df['bias'][nlos2_idx], 
                # s=1, 
                label=r'NLOS - 2 obstacles', 
                alpha=0.5,
                color='g')

    fig.legend()

    if show:
        plt.show(block=True)

class MocapSeparateNlos(object):
    def __init__(self, raw_obj, tag_ids, moment_arms, calibrate=False) -> None:
        self.tag_ids = tag_ids
        self.moment_arms = moment_arms
        
        self.r = raw_obj.r
        self.rot = raw_obj.rot
        self.t_r = raw_obj.t_r

        self.data = {}
        for pair in raw_obj.ts_data:
            self.data[pair] = pd.DataFrame(raw_obj.ts_data[pair],
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
            self.data[pair].drop(self.data[pair][self.data[pair]['range'] > 8].index, inplace=True)

        self._get_tags_trajectory()
        self._fit_splines() # would need polygon of each robot whenever
                            # any uwb measurement occurs
        self._compute_gt_distance()

        self._compute_drones_convex_hull()

        if calibrate:
            self._calibrate()

        self._drop_timestamps()

        self.identify_nlos()

    def _calibrate(self):
        correct_range = ComputeCorrectedRange(in_ns=True)
        for pair in self.data:
            data_iter = self.data[pair].copy()
            n = len(data_iter)
            data_iter["from_id"] = pair[0]*np.ones((n,))
            data_iter["to_id"] = pair[1]*np.ones((n,))
            calib_iter = correct_range.get_corrected_range(data_iter)
            self.data[pair]["uncalib_range"] = self.data[pair]["range"].copy()
            self.data[pair]["range"] = calib_iter['range']
            self.data[pair]["std"] = calib_iter['std']

    def _drop_timestamps(self):
        for pair in self.data:
            self.data[pair].drop(axis=1,
                                 columns = ['tx1', 
                                            'rx1',
                                            'tx2',
                                            'rx2',
                                            'tx3',
                                            'rx3'])

    def _compute_drones_convex_hull(self):
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')

        self.ch = {}
        
        x_pad = 20/1000
        y_pad = 20/1000
        z_pad = 120/1000*1.25

        ## IFO 001
        r1 = np.array([-66.111, -10.754, 19.430])/1000
        r2 = np.array([22.317, -35.613, 21.300])/1000
        r3 = np.array([50.500, -7.650, 20.516])/1000
        
        a = r1 + np.array([-x_pad, 25+y_pad, 0])
        b = np.array([r1[0]-x_pad, r2[1]-y_pad, r1[2]])
        c = np.array([r3[0]+x_pad, r2[1]-y_pad, r3[2]])
        d = np.array([r3[0]+x_pad, a[1], r3[2]])
        
        z_offset = np.array([0,0,-z_pad])
        e = a + z_offset
        f = b + z_offset
        g = c + z_offset
        h = d + z_offset

        vertices = np.array([a,b,c,d,e,f,g,h])
        
        # for point in vertices:
        #     ax.scatter(point[0],point[1],point[2],color='b')

        self.ch['ifo001'] = ConvexHull(vertices)

        ## IFO 002
        r1 = np.array([66.101, 29.598, 44.038])/1000
        r2 = np.array([-38.727, 33.397, 46.074])/1000
        r3 = np.array([16.735, -40.649, 22.850])/1000
        
        a = r2 + np.array([-x_pad, y_pad, 0])
        b = np.array([a[0], r3[1]-y_pad/2, r2[2]])
        c = np.array([r1[0]+x_pad, b[1], r1[2]])
        d = r1 + np.array([x_pad, y_pad, 0])
        
        z_offset = np.array([0,0,-z_pad])
        e = a + z_offset
        f = b + z_offset
        g = c + z_offset
        h = d + z_offset

        vertices = np.array([a,b,c,d,e,f,g,h])

        # for point in vertices:
        #     ax.scatter(point[0],point[1],point[2],color='r')

        self.ch['ifo002'] = ConvexHull(vertices)

        ## IFO 003
        r1 = np.array([-26.671, 9.990, 15.755])/1000
        r2 = np.array([-29.042, -40.149, 12.696])/1000
        r3 = np.array([84.337, -13.887, 17.257])/1000
        
        a = r1 + np.array([-x_pad, y_pad, 0])
        b = r2 + np.array([-x_pad, -y_pad, 0])
        c = np.array([r3[0]+x_pad, b[1], r3[2]])
        d = np.array([r3[0]+x_pad, a[1], r3[2]])
        
        z_offset = np.array([0,0,-z_pad])
        e = a + z_offset
        f = b + z_offset
        g = c + z_offset
        h = d + z_offset

        vertices = np.array([a,b,c,d,e,f,g,h])

        # for point in vertices:
        #     ax.scatter(point[0],point[1],point[2],color='g')

        self.ch['ifo003'] = ConvexHull(vertices)
        
        # plt.show(block=True)

    def identify_nlos(self):
        for pair in self.data:
            num_collisions = np.empty(0, dtype=int)
            for _, row in self.data[pair].iterrows():
                collisions = self._check_collision(row,pair)
                num_collisions = np.hstack((num_collisions, len(collisions)))

            self.data[pair]['num_collisions'] = num_collisions

            print(r'Completed identifying obstacles for pair ' + str(pair))

    def _check_collision(self, row, pair) -> list:
        t = row['t_nsecs']

        # Get convex inequalities for the three drones' convex hulls
        polyh_drones = {}
        for machine in self.tag_ids:
            polyh_drones[machine] = self._generate_drone_polyhedron(t, machine)

        # Sample points in the straight line between the two ranging tags
        points = self._sample_from_range_vector(t, pair)

        collisions = []
        for machine in polyh_drones:
            is_inside = self._check_points_in_polyhedron(points,
                                                         polyh_drones[machine]['A'],
                                                         polyh_drones[machine]['b'])
            if np.sum(is_inside) > 0:
                collisions = collisions + [machine]

        return collisions

    def _generate_drone_polyhedron(self, t, machine):
        C, r = self._get_pose(t, machine)
        A, b = self._get_convex_inequalities(self.ch[machine], C.T, r)

        return {'A': A, 'b': b}

    def _sample_from_range_vector(self, t, pair):
        n = 1000
        
        r0 = self._get_tag_position(t, pair[0])
        r1 = self._get_tag_position(t, pair[1])

        return np.vstack((np.linspace(r0[0], r1[0], n),
                          np.linspace(r0[1], r1[1], n),
                          np.linspace(r0[2], r1[2], n))).T

    def _get_tag_position(self, t, tag):
        return np.array([self.r_tags_spl[tag][0](t),
                         self.r_tags_spl[tag][1](t),
                         self.r_tags_spl[tag][2](t)])

    def _get_pose(self, t, machine):
        q = np.array([self.q_spl[machine][0](t),
                      self.q_spl[machine][1](t),
                      self.q_spl[machine][2](t),
                      self.q_spl[machine][3](t)])
        phi = R.from_quat(q).as_rotvec() 

        r = np.array([self.r_spl[machine][0](t),
                      self.r_spl[machine][1](t),
                      self.r_spl[machine][2](t)])
        
        return SO3.Exp(phi), r.reshape((-1, 1))

    def _compute_gt_distance(self):
        for pair in self.data:
            t = self.data[pair]['t_nsecs']
            r0 = self._get_tag_position(t, pair[0])
            r1 = self._get_tag_position(t, pair[1])

            self.data[pair]['gt'] = np.linalg.norm(r1-r0, axis=0)
            self.data[pair].drop(self.data[pair][self.data[pair]['gt'] > 8].index, inplace=True)

    def _get_tags_trajectory(self):
        r_tags = {} # position
        for machine in self.tag_ids:
            # Iterate through tags for every machine
            for i,tag in enumerate(self.tag_ids[machine]):
                r_tags[tag] = self.r[machine] \
                              + (self.rot[machine].as_matrix() 
                                 @ self.moment_arms[machine][i]).T

        self.r_tags = r_tags

    def _fit_splines(self):
        # Create position splines for all tags
        self.r_tags_spl = {}
        for machine in self.r:
            for tag in self.tag_ids[machine]:
                self.r_tags_spl[tag] \
                    = [BSpline(self.t_r[machine], self.r_tags[tag][0,:], k=3),
                       BSpline(self.t_r[machine], self.r_tags[tag][1,:], k=3),
                       BSpline(self.t_r[machine], self.r_tags[tag][2,:], k=3),]

        # Create position splines for all machines
        self.r_spl = {}
        for machine in self.r:
            self.r_spl[machine] \
                = [BSpline(self.t_r[machine], self.r[machine][0,:], k=3),
                   BSpline(self.t_r[machine], self.r[machine][1,:], k=3),
                   BSpline(self.t_r[machine], self.r[machine][2,:], k=3),]

        # Create quaternion splines for all machines
        self.q_spl = {}
        for machine in self.rot:
            q = self.rot[machine].as_quat().T
            self.q_spl[machine] \
                    = [BSpline(self.t_r[machine], q[0,:], k=3),
                       BSpline(self.t_r[machine], q[1,:], k=3),
                       BSpline(self.t_r[machine], q[2,:], k=3),
                       BSpline(self.t_r[machine], q[3,:], k=3),]
        
    @staticmethod
    def _get_convex_inequalities(ch: ConvexHull, C_ab, r_zw_a):
        """
        Returns the two matrices A, b that specify a convex polyhedron. The 
        polyhedron's internal space is defined by all positions r_a that satisfy 

            A @ r_a <= b

        where r_a is resolved in frame "a".

        PARAMETERS
        ----------

        ch: scipy.spatial.ConvexHull made from vertices supplied in the body frame

        C_ab: np.ndarray with shape (3, 3)
            attitude of frame a relative to frame b

        r_zw_a: np.ndarray with size 3 
            position of frame b reference point relative to frame a reference point 


        RETURNS:
        --------
        A: np.ndarray with shape (M, 3)
        b: np.ndarray with shape (M, 1)
        """

        A_b = ch.equations[:, 0:3]
        b_b = -ch.equations[:, 3].reshape((-1, 1))
        # Apple the pose transformation to the linear inequalities
        A = A_b @ C_ab.T 
        b = b_b + A @ r_zw_a.reshape((-1, 1))
        return A, b 

    @staticmethod
    def _check_points_in_polyhedron(test_points, A, b):
        """
        Checks if points lie within a convex polyhedron supplied by linear 
        inequality matrices A, b.


        PARAMETERS:
        -----------
        test_points: np.ndarray with shape (N, 3) of coordinates
        A: np.ndarray with shape (M, 3)
        b: np.ndarray with shape (M, 1)

        RETURNS:
        --------
        is_inside: np.ndarray with shape (N,) of boolean values corresponding to 
        each test_point.
        """
        N = test_points.shape[0]
        is_inside = np.array([False] * N)
        b = b.reshape((-1,1))
        is_inside = np.all((A @ test_points.T - b) <= 0, axis=0)
        return is_inside

    def visualize(self):
        # Interpolation results
        pass


# %%
tag_ids={'ifo001': [1,2],
         'ifo002': [3,4],
         'ifo003': [5,6]}
moment_arms={'ifo001': [np.array([0.13189,-0.17245,-0.05249]), 
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
sep_nlos_obj = MocapSeparateNlos(raw_obj, tag_ids, moment_arms, calibrate=True)

# %%
# Plotting 

# Plot range vs. gt
n = len(sep_nlos_obj.data)
num_rows = 4
num_cols = int(np.ceil(n/4))
fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)

for i,pair in enumerate(sep_nlos_obj.data):
    row = i//num_cols
    col = np.mod(i,num_cols)

    df = sep_nlos_obj.data[pair]
    axs[row,col].plot(df['t_nsecs'], df['gt'])

    los_idx = np.array(df['num_collisions'] == 0)
    nlos1_idx = np.array(df['num_collisions'] == 1)
    nlos2_idx = np.array(df['num_collisions'] == 2)

    axs[row,col].scatter(df['t_nsecs'][los_idx], df['range'][los_idx])
    axs[row,col].scatter(df['t_nsecs'][nlos1_idx], df['range'][nlos1_idx])
    axs[row,col].scatter(df['t_nsecs'][nlos2_idx], df['range'][nlos2_idx])
    axs[row,col].set_title(r''+str(pair))
    axs[row,col].set_ylim(0,10)

# Append all data into one dataframe
df_all = append_dfs(sep_nlos_obj.data)

# Plot normalized histograms
los_idx = np.array(df_all['num_collisions'] == 0)
nlos1_idx = np.array(df_all['num_collisions'] == 1)
nlos2_idx = np.array(df_all['num_collisions'] == 2)
plot_histograms(df_all, los_idx, nlos1_idx, nlos2_idx)

# %%
# Manual random forest attempt
# thresh_fpp = -84.5
# thresh_rxp = -78.5
# thresh_std = 59
thresh_fpp = -85
thresh_rxp = -79
thresh_std = 60
print(np.sum((df_all['std_avg'][los_idx] < thresh_std) 
              & (df_all['fpp_avg'][los_idx]  < thresh_fpp) 
              & (df_all['rxp_avg'][los_idx] < thresh_rxp)) 
      / np.sum(los_idx))
print(np.sum((df_all['std_avg'][nlos1_idx] < thresh_std) 
              & (df_all['fpp_avg'][nlos1_idx] < thresh_fpp) 
              & (df_all['rxp_avg'][nlos1_idx] < thresh_rxp)) 
      / np.sum(nlos1_idx))
print(np.sum((df_all['std_avg'][nlos2_idx] < thresh_std) 
              & (df_all['fpp_avg'][nlos2_idx] < thresh_fpp) 
              & (df_all['rxp_avg'][nlos2_idx] < thresh_rxp)) 
      / np.sum(nlos2_idx))
# %%
# Train random forest
X = np.vstack((df_all['fpp_avg'],
               df_all['rxp_avg'],
               df_all['fpp_avg'] - df_all['rxp_avg'],
               df_all['std_avg'])).T
y = np.array(df_all['num_collisions'])

n0 = np.sum(los_idx)
n1 = np.sum(nlos1_idx)
n2 = np.sum(nlos2_idx)
n = n0+n1+n2

n0 *= 1
# n1 *= 0.4
n2 *= 1

clf = RandomForestClassifier(class_weight={0:n/n0,1:n/n1,2:n/n2},max_depth=15)

clf.fit(X,y)

# %%
# Evaluate random forest on train data
print(RandomForestClassifier.score(clf,X,y))

class_pred = clf.predict(X).astype(int)

los_idx = np.array(class_pred == 0)
nlos1_idx = np.array(class_pred == 1)
nlos2_idx = np.array(class_pred == 2)
plot_histograms(df_all ,los_idx, nlos1_idx, nlos2_idx)

plt.show(block=True)
# %%
# Load test data
raw_obj_test = PostProcess("datasets/2022_08_03/bias_calibration_new2/merged.bag",
                           tag_ids,
                           moment_arms,
                           num_meas=-1)

sep_nlos_obj_test = MocapSeparateNlos(raw_obj_test, tag_ids, moment_arms, calibrate=True)

# Append all data into one dataframe
df_all_test = append_dfs(sep_nlos_obj_test.data)

# %%
# Evaluate random forest on test data
X_test = np.vstack((df_all_test['fpp_avg'],
                    df_all_test['rxp_avg'],
                    df_all_test['fpp_avg'] - df_all_test['rxp_avg'],
                    df_all_test['std_avg'])).T
y_test = np.array(df_all_test['num_collisions'])
class_pred_test = clf.predict(X_test).astype(int)

print(RandomForestClassifier.score(clf,X_test,y_test))

class_pred_test = clf.predict(X_test).astype(int)

# los_idx_test = np.array(class_pred_test == 0)
# nlos1_idx_test = np.array(class_pred_test == 1)
# nlos2_idx_test = np.array(class_pred_test == 2)
# plot_histograms(df_all_test, los_idx_test, nlos1_idx_test, nlos2_idx_test)

# %%
los_idx_ytest = np.array(y_test == 0)
nlos1_idx_ytest = np.array(y_test == 1)
nlos2_idx_ytest = np.array(y_test == 2)

print(np.sum(class_pred_test[los_idx_ytest] == 0) / np.sum(los_idx_ytest)) 

print((np.sum(class_pred_test[nlos1_idx_ytest] == 1) + np.sum(class_pred_test[nlos1_idx_ytest] == 2)) / np.sum(nlos1_idx_ytest)) 

print((np.sum(class_pred_test[nlos2_idx_ytest] == 1) + np.sum(class_pred_test[nlos2_idx_ytest] == 2)) / np.sum(nlos2_idx_ytest)) 

los_idx = np.array(df_all_test['num_collisions'] == 0)
nlos1_idx = np.array(df_all_test['num_collisions'] == 1)
nlos2_idx = np.array(df_all_test['num_collisions'] == 2)
plot_histograms(df_all_test, los_idx, nlos1_idx, nlos2_idx, show=False)

los_idx_test = np.array(class_pred_test == 0)
nlos1_idx_test = np.array(class_pred_test == 1)
nlos2_idx_test = np.array(class_pred_test == 2)
plot_histograms(df_all_test, los_idx_test, nlos1_idx_test, nlos2_idx_test)
# %%
