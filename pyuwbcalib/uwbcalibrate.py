import numpy as np
from .postprocess import PostProcess
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
import pickle
import pandas as pd

class UwbCalibrate(PostProcess):
    _c = 299702547  # speed of light
    _inherited = [
                  'df',
                  'machine_ids',
                  'tag_ids',
                  'ds_twr',
                  'fpp_exists',
                 ]
    
    def __init__(self, 
                 data, 
                 rm_static = False, 
                 f_lift=lambda x: 10**((x + 82)/10)):
        """
        Constructor
        """
        self._data = data
        
        if rm_static:
            self._remove_static_regions()

        self.lift = f_lift

    def __getattr__(self, attr):
        if attr in self._inherited:
            return getattr(self._data, attr)

    def _remove_static_regions(self):
        '''
        Remove the static region in the extremes.
        TODO: extensively test this with a few datasets.
        '''        
        thresh = 0.1
        window = 10 # size of windows
        
        lower_idx = 0
        upper_idx = np.Inf

        for machine_i in self.machine_ids:
            r_iw_a = self.get_machine_pos(machine_i, as_numpy = True)
            
            # Lower bound
            x = r_iw_a[:,0]
            y = r_iw_a[:,1]
            z = r_iw_a[:,2]
            
            x_lower, x_upper = self._find_static_extremes(x, thresh, window)
            y_lower, y_upper = self._find_static_extremes(y, thresh, window)
            z_lower, z_upper = self._find_static_extremes(z, thresh, window)
            
            lowest_low_idx = np.min([x_lower, y_lower, z_lower])
            highest_upper_idx = np.max([x_upper, y_upper, z_upper])
            
            if lowest_low_idx > lower_idx:
                lower_idx = lowest_low_idx
            
            if highest_upper_idx < upper_idx:
                upper_idx = highest_upper_idx
            
        n = len(self.df)
        self.df.drop(np.linspace(0,lower_idx,lower_idx+1), inplace=True)
        self.df.drop(np.linspace(upper_idx, n-1, n-upper_idx), inplace=True)
        # self.df.reset_index(inplace=True, drop=True)
        
    @staticmethod
    def _find_static_extremes(r, thresh, window):
        rolling_mean = pd.DataFrame(r).rolling(window, center=True).mean()
        rolling_mean = rolling_mean.fillna(method="bfill").fillna(method="ffill")
        rolling_mean = np.array(rolling_mean)
        
        diff_lower = np.abs(rolling_mean[1:] - rolling_mean[0])
        diff_upper = np.abs(np.flip(rolling_mean)[1:] - rolling_mean[-1])
        
        lower_idx = np.argmax(diff_lower>thresh)
        upper_idx = len(diff_lower) - np.argmax(np.flip(diff_upper)>thresh) - 1
        
        return lower_idx, upper_idx

    def calibrate_antennas(
                            self, 
                            loss='cauchy', 
                            tx_rx_split={'tx':60, 'rx':40},
                          ):
        """
        Calibrate the antenna delays by formulating and solving a linear least-squares problem.

        RETURNS:
        --------
        dict: Dictionary with 3 fields each for tag z \in {i,j,k}
            Module i: (float)
                Antenna delay for tag i
        """
        tags = list(np.concatenate(list(self.tag_ids.values())).flat)
        n = len(self.df)
        
        from_idx = [tags.index(x) for x in np.array(self.df["from_id"])]
        to_idx = [tags.index(x) for x in np.array(self.df["to_id"])]
        rows = np.linspace(0,n-1,n).astype(int)
        
        if self.ds_twr:
            K = self.df["del_t3"] / self.df["del_t4"]
        else:
            K = 1
        
        A = np.zeros((n,len(tags)))
        A[rows, from_idx] += 0.5
        A[rows, to_idx] += 0.5 * K
        
        b = 1 / self._c * self.df["gt_range"] * 1e9 \
            - 0.5 * self.df["del_t1"] \
            + 0.5 * K * self.df["del_t2"]
        b = np.array(b)

        x = self._solve_for_antenna_delays(A, b, loss)['x']
        x = x.flatten()

        self.delays = {id: x[i] for i,id in enumerate(tags)}
        
        # self._correct_antenna_delays(tx_rx_split)

    def _correct_antenna_delay(self, tx_rx_split):
        """
        Modifies the data of this object to correct for the antenna delay of a
        specific module.

        PARAMETERS:
        -----------
        id: int
            Module ID whose antenna delay is to be corrected.
        delay: float
            The amount of antenna delay, in nanoseconds.

        TODO: What about D1 and D2? This seems to be a problem.
              We might have to calibrate for TX and RX delays separately
              if we are to proceed with Kalman filtering with this architecture.
        TODO: tof1, tof2, and tof3 as well, once individual delays are taken into consideration.
        TODO: access delays from self object
        """
        
        # TODO: DO SOMETHING SIMILAR TO LINES 106, 107, 116, 117
        
        for id in delays_dict.keys():
            delay = delays_dict[id]
            for pair in self.time_intervals:
                if pair[0] == id:
                    self.time_intervals[pair]["Ra1"] += delay
                elif pair[1] == id:
                    self.time_intervals[pair]["Db1"] -= delay

    def _solve_for_antenna_delays(self, A, b, loss):
        """
        Solves the linear least-squares problem.

        PARAMETERS:
        -----------
        A: 2D np.array
            The A matrix.
        b: np.array
            The b vector.

        RETURNS:
        --------
        np.array: The solution to the Ax=b problem.
        """
        # return np.linalg.lstsq(A, b)
        n = A.shape[1]
        return least_squares(self._cost_func, np.zeros(n), loss=loss, f_scale=0.1, args=(A,b.T))

    @staticmethod
    def _cost_func(x,A,b):
        return (A@x - b).reshape(-1,)

    @staticmethod
    def _rolling_window(a, window):
        '''
        Copied from 
        https://stackoverflow.com/questions/27427618/how-can-i-simply-calculate-the-rolling-moving-variance-of-a-time-series-in-pytho
        '''
        pad = np.ones(len(a.shape), dtype=np.int32)
        pad[-1] = window-1
        pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
        a = np.pad(a, pad,mode='reflect')
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _reject_outliers(self, bias, lifted_pr, std_window, chi_thresh, axs, axs_std):
        '''
        keep fitting models and rejecting outliers until no more outliers
        '''
        outlier_bias = np.empty(0,)
        outlier_lifted_pr = np.empty(0,)
        while True:
            # Fit 
            windows = self._rolling_window(bias.ravel(), std_window)
            bias_std = np.std(windows - np.mean(windows).reshape(-1,1), axis=-1)
            std_spl = UnivariateSpline(lifted_pr, bias_std, k=4)
            bias_std_new = std_spl(lifted_pr)

            # Fit spline
            spl = UnivariateSpline(lifted_pr, bias, k=3)
            bias_fit = spl(lifted_pr)

            # Remove outliers
            norm_e_squared = (bias - bias_fit)**2 / bias_std_new**2
            # outliers = norm_e_squared > chi_thresh
            outliers = np.abs(bias - bias_fit) > chi_thresh
            if np.sum(outliers):
                outlier_bias = np.hstack((outlier_bias, bias[outliers]))
                bias = bias[~outliers]

                outlier_lifted_pr = np.hstack((outlier_lifted_pr, lifted_pr[outliers]))
                lifted_pr = lifted_pr[~outliers] 
            else:
                break

        # PLOTTING
        try:
            axs.scatter(lifted_pr, bias)
            axs.scatter(outlier_lifted_pr, outlier_bias)
            axs.set_xlabel(r"$f(P_r)$")
            axs.set_ylabel(r"Bias [m]")
        except:
            pass
        
        try:
            axs_std.scatter(lifted_pr, bias_std)
            axs_std.scatter(lifted_pr, bias_std_new)
            axs_std.set_xlabel(r"$f(P_r)$")
            axs_std.set_ylabel(r"Bias Standard Deviation [m]")
        except:
            pass

        return spl, std_spl, bias, lifted_pr

    def fit_model(self, std_window=50, chi_thresh=10.8, merge_pairs=False):
        
        if merge_pairs:
            sorted_pairs = [tuple(sorted(i)) for i in self.tag_pairs]
            addressed_pairs = list(set(sorted_pairs))
            for i, pair in enumerate(addressed_pairs):
                if pair not in self.tag_pairs:
                    addressed_pairs[i] = pair[::-1]
        else:
            addressed_pairs = self.tag_pairs

        num_pairs = len(addressed_pairs)
        # fig, axs = plt.subplots(3,num_pairs,sharey='row')
        fig, axs = plt.subplots(4,int(np.ceil((num_pairs+1)/4)),sharey='all',sharex='all')
        fig2, axs2 = plt.subplots(2,1) 
        fig3, axs3 = plt.subplots(4,int(np.ceil((num_pairs+1)/4)),sharey='all',sharex='all') 
        fig4, axs4 = plt.subplots(4,int(np.ceil((num_pairs+1)/4)),sharey='all',sharex='all') 
        fig3.suptitle(r"Outlier rejection")
        fig4.suptitle(r"Standard Deviation Fit")
        axs[0,0].set_ylabel(r"Bias [m]")

        self.mean_spline = {pair:[] for pair in addressed_pairs}
        self.std_spline = {pair:[] for pair in addressed_pairs}

        self._all_spline_data = {'lifted_pr_trunc': np.empty(0),
                                 'lifted_pr_trunc_STDFIT': np.empty(0),
                                 'lifted_pr': np.empty(0),
                                 'bias': np.empty(0),
                                 'bias_STDFIT': np.empty(0),
                                 'std': np.empty(0)}

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for lv0, pair in enumerate(addressed_pairs):
            range = self.compute_range_meas(pair)
            bias = range - self.time_intervals[pair]["r_gt"]
            lifted_pr = 0.5*self.lift(self.ts_data[pair][:,self.fpp1_idx]) \
                        + 0.5*self.lift(self.ts_data[pair][:,self.fpp2_idx])
            r_gt_unsorted = self.time_intervals[pair]["r_gt"]

            if merge_pairs and pair[::-1] in self.tag_pairs:
                opposite_pair = pair[::-1]
                range_new = self.compute_range_meas(opposite_pair)
                range = np.append(range, range_new)
                bias = np.append(bias, range_new - self.time_intervals[opposite_pair]["r_gt"])
                lifted_pr = np.append(lifted_pr, 
                                      0.5*self.lift(self.ts_data[opposite_pair][:,self.fpp1_idx]) \
                                      + 0.5*self.lift(self.ts_data[opposite_pair][:,self.fpp2_idx]))
                r_gt_unsorted = np.append(r_gt_unsorted, self.time_intervals[opposite_pair]["r_gt"])

            pr_thresh = 1.8
            bias = bias[lifted_pr < pr_thresh]
            r_gt_unsorted = r_gt_unsorted[lifted_pr < pr_thresh]
            lifted_pr = lifted_pr[lifted_pr < pr_thresh]   

            # rolling var along last axis
            sort_pr = np.argsort(lifted_pr)
            bias = bias[sort_pr]
            lifted_pr = lifted_pr[sort_pr]
            r_gt = r_gt_unsorted[sort_pr]

            row = np.mod(lv0,4)
            col = int(np.floor(lv0/4))
            _, std_spl, bias_trunc_STDFIT, lifted_pr_trunc_STDFIT \
                        = self._reject_outliers(bias, 
                                                lifted_pr, 
                                                std_window, 
                                                3, 
                                                [],
                                                axs4[row,col])
            spl, _, bias_trunc, lifted_pr_trunc \
                        = self._reject_outliers(bias, 
                                                lifted_pr, 
                                                std_window, 
                                                0.3, 
                                                axs3[row,col],
                                                [])
            self.mean_spline[pair] = spl
            self.std_spline[pair] = std_spl
            bias_std = std_spl(lifted_pr)
            bias_fit = spl(lifted_pr)

            # Save the fit data 
            self._all_spline_data['lifted_pr_trunc'] = np.append(self._all_spline_data['lifted_pr_trunc'], lifted_pr_trunc)
            self._all_spline_data['lifted_pr_trunc_STDFIT'] = np.append(self._all_spline_data['lifted_pr_trunc_STDFIT'], lifted_pr_trunc_STDFIT)
            self._all_spline_data['lifted_pr'] = np.append(self._all_spline_data['lifted_pr'], lifted_pr)
            self._all_spline_data['bias'] = np.append(self._all_spline_data['bias'], bias_trunc)
            self._all_spline_data['bias_STDFIT'] = np.append(self._all_spline_data['bias_STDFIT'], bias_trunc_STDFIT)
            self._all_spline_data['std'] = np.append(self._all_spline_data['std'], bias_std)

            ### PLOT 1 ###
            axs[np.mod(lv0,4),int(np.floor(lv0/4))].scatter(lifted_pr_trunc_STDFIT, bias_trunc_STDFIT, label=r"Raw data", linestyle="dotted", s=1)
            axs[np.mod(lv0,4),int(np.floor(lv0/4))].plot(lifted_pr, bias_fit, label=r"Fit")
            axs[np.mod(lv0,4),int(np.floor(lv0/4))].fill_between(
                lifted_pr.ravel(),
                bias_fit - 3 * bias_std,
                bias_fit + 3 * bias_std,
                alpha=0.5,
                label=r"99.7% confidence interval",
            )
            axs[np.mod(lv0,4),int(np.floor(lv0/4))].set_xlabel(r"$f(P_r)$")
            axs[np.mod(lv0,4),int(np.floor(lv0/4))].set_title(str(pair))
            
            # ## Visualize std vs. distance
            # axs[2,lv0].scatter(r_gt, bias_std, s=1)
            # axs[2,lv0].set_xlabel(r"Ground truth distance [m]")

            ### PLOT 2 ### Plot with all splines
            if lv0 == 0:
                axs2[0].plot(lifted_pr, 
                             bias_fit*100, 
                             label="Individual Pairs", 
                             linewidth=1, 
                             color='gray',
                             alpha=0.5,)
            else:
                axs2[0].plot(lifted_pr, 
                             bias_fit*100, 
                             linewidth=1, 
                             color='gray',
                             alpha=0.5,)
            # fig2.suptitle(r"Bias-Power Fit")

            axs2[1].plot(lifted_pr, 
                         bias_std*100, 
                         linewidth=1, 
                         color='gray',
                         alpha=0.5,)

        axs[0,-1].legend()

        # Sort stored fit data
        sort_pr = np.argsort(self._all_spline_data['lifted_pr_trunc'])
        self._all_spline_data['lifted_pr_trunc'] = self._all_spline_data['lifted_pr_trunc'][sort_pr]
        self._all_spline_data['bias'] = self._all_spline_data['bias'][sort_pr]
        self._all_spline_data['std'] = self._all_spline_data['std'][sort_pr]
        
        sort_pr = np.argsort(self._all_spline_data['lifted_pr_trunc_STDFIT'])
        self._all_spline_data['lifted_pr_trunc_STDFIT'] = self._all_spline_data['lifted_pr_trunc_STDFIT'][sort_pr]
        self._all_spline_data['bias_STDFIT'] = self._all_spline_data['bias_STDFIT'][sort_pr]

        sort_pr = np.argsort(self._all_spline_data['lifted_pr'])
        self._all_spline_data['lifted_pr'] = self._all_spline_data['lifted_pr'][sort_pr]

        row = np.mod(lv0+1,4)
        col = int(np.floor((lv0+1)/4))
        _, std_spl, bias_trunc_STDFIT, lifted_pr_trunc_STDFIT \
                        = self._reject_outliers(self._all_spline_data['bias_STDFIT'], 
                                                self._all_spline_data['lifted_pr_trunc_STDFIT'], 
                                                std_window, 
                                                3, 
                                                [],
                                                axs4[row,col])
        spl, _, bias_trunc, lifted_pr_trunc \
                        = self._reject_outliers(self._all_spline_data['bias'], 
                                                self._all_spline_data['lifted_pr_trunc'], 
                                                std_window, 
                                                0.3, 
                                                axs3[row,col],
                                                [])

        bias_std = std_spl( self._all_spline_data['lifted_pr'])
        bias_fit = spl( self._all_spline_data['lifted_pr'])

        axs[np.mod(lv0+1,4),int(np.floor((lv0+1)/4))].scatter(lifted_pr_trunc_STDFIT,
                                                            bias_trunc_STDFIT, 
                                                            label=r"Raw data", 
                                                            linestyle="dotted", 
                                                            s=1)
        axs[np.mod(lv0+1,4),int(np.floor((lv0+1)/4))].plot( self._all_spline_data['lifted_pr'], bias_fit, label=r"Fit")
        axs[np.mod(lv0+1,4),int(np.floor((lv0+1)/4))].fill_between(
             self._all_spline_data['lifted_pr'].ravel(),
            bias_fit - 3 * bias_std,
            bias_fit + 3 * bias_std,
            alpha=0.5,
            label=r"99.97% confidence interval",
        )
        axs[np.mod(lv0+1,4),int(np.floor((lv0+1)/4))].set_xlabel(r"$f(P_r)$")

        axs2[0].plot(self._all_spline_data['lifted_pr'], bias_fit*100, label=r"Average", linewidth=8, color=colors[0])
        # axs2[0].legend(loc='upper right')
        # axs2[0].set_xlabel(r"$f(P_r)$")
        axs2[0].set_ylabel(r"Bias [cm]")
        # fig2.suptitle(r"Bias-Power Fit")
        lgnd = fig2.legend(ncol=2, loc='upper center', facecolor=[1,1,1])
        lgnd.legendHandles[0]._alpha = 0.9

        axs2[1].plot(self._all_spline_data['lifted_pr'], bias_std*100, linewidth=8, color=colors[0])
        axs2[1].set_xlabel(r"$\Psi\left( 0.5 (p_4^\mathrm{f} + p_2^\mathrm{f}) \right)$")
        axs2[1].set_ylabel(r"Bias Std [cm]")
        
        axs2[0].set_yticks([-10, -5, 0, 5, 10])
        axs2[1].set_yticks([0, 10, 20])

        self.spl = spl
        self.std_spl = std_spl
        
        fig2.subplots_adjust(bottom=0.15, hspace=0.3)
        fig2.savefig("figs/bias_power_fit.pdf", dpi=300)

    def compute_range_meas(self, pair=(1,2), visualize=False, owr = False):
        # TODO: Inherit this function from PostProcess?
        interv = self.time_intervals[pair]
        if owr and self.mult_twr:
            range = 1/2 * self._c / 1e9 \
                    * (abs(interv["tof1"]) \
                       + abs(interv["tof2"]) \
                       + 0*abs(interv["tof3"])) # TODO: why *0? Antenna delay?
        elif owr:
            range = 1/2 * self._c / 1e9 \
                    * (abs(interv["tof1"]) \
                       + abs(interv["tof2"]))
        elif self.mult_twr:
            range = 0.5 * self._c / 1e9 * \
                (interv["Ra1"] - (interv["Ra2"] / interv["Db2"]) * interv["Db1"])
        else:
            range = 0.5 * self._c / 1e9 * \
                (interv["Ra1"] - interv["Db1"])

        if visualize:
            fig, axs = plt.subplots(1)

            axs.plot(interv["t"]/1e9, range, label='Range Measurements')

            axs.set_ylabel("Distance [m]")
            axs.set_xlabel("t [s]")
            axs.set_ylim([-1, 5])

        return range

    def save_calib_results(self):
        calib_results = {
                        'delays': self.delays,
                        'bias_spl': self.spl,
                        'std_spl': self.std_spl,
                        }

        with open("calib_results.pickle","wb") as file:
            pickle.dump(calib_results, file)