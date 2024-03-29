import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd 
from scipy.interpolate import interp1d
from pymlg import SO3
from configparser import ConfigParser
from scipy.interpolate import UnivariateSpline
from typing import List

def save(obj, filename="file.pickle") -> None:
    """Save object using pickle.

    Parameters
    ----------
    obj : object
        Any object to be saved.
    filename : str, optional
        Name of pickle file, by default "file.pickle"
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file)

def load(filename='file.pickle') -> object:
    """Load an object using pickle.

    Parameters
    ----------
    filename : str, optional
        Name of pickle file, by default 'file.pickle'

    Returns
    -------
    object
        The loaded object.
    """
    with open(filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        
    return data

def merge_calib_results(
    calib_list: List[dict],
):
    """Merge calibration results from multiple calibration runs.

    Parameters
    ----------
    calib_list : list[dict]
        List of calibration results.
        
    Returns
    -------
    dict
        The merged calibration results.
    """
    
    final_result = {
        'delays': {},
        'bias_spl': {},
        'std_spl': {},
    }
    
    # First, merge the antenna delays by taking the mean of all the delays
    for calib in calib_list:
        for anchor_id in calib['delays']:
            if anchor_id not in final_result['delays']:
                final_result['delays'][anchor_id] = []
            final_result['delays'][anchor_id].append(calib['delays'][anchor_id])
            
    for anchor_id in final_result['delays']:
        final_result['delays'][anchor_id] = np.mean(final_result['delays'][anchor_id])
        
    # Second merge the bias_spl by sampling from each spline 
    # and fitting a spline to the mean of the samples
    samples_x = np.linspace(0.01, 1.8, 1000)
    samples_y_list = []
    for calib in calib_list:
        samples_y_list.append(calib['bias_spl'](samples_x))
        
    samples_y = np.mean(samples_y_list, axis=0)
    final_result['bias_spl'] = UnivariateSpline(samples_x, samples_y, k=3)
    
    # Third, merge the std_spl in a similar way
    samples_y_list = []
    for calib in calib_list:
        samples_y_list.append(calib['std_spl'](samples_x))
        
    samples_y = np.mean(samples_y_list, axis=0)
    final_result['std_spl'] = UnivariateSpline(samples_x, samples_y, k=4)
        
    return final_result
    

def read_anchor_positions(
    parser: ConfigParser,
):
    """Read anchor positions from the configuration file.
    
    Parameters
    ----------
    parser : ConfigParser
        The configuration parser.
        
    Returns
    -------
    dict
        keys: anchor id
        values: anchor position
    """
    anchor_positions = {int(id) : eval(parser['ANCHORS'][id]) for id in parser['ANCHORS']}
    return anchor_positions

def set_plotting_env() -> None:
    """Set the plotting environment
    """
    sns.set_palette("colorblind")
    sns.set_style('whitegrid')

    plt.rc('figure', figsize=(16, 9))
    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=35)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('legend', facecolor=[1,1,1])
    plt.rc('legend', fontsize=30)
    plt.rcParams['figure.constrained_layout.use'] = True

def interpolate(
        x, 
        t_old, 
        t_new
    ) -> np.ndarray:
        """Interpolate data from old timestamps to new timestamps.

        Parameters
        ----------
        x : np.ndarray
            Data to be interpolated.
        t_old : np.ndarray
            The old timestamps.
        t_new : _type_
            The new timestamps.

        Returns
        -------
        np.ndarray
            Interpolated data.
        """
        f = interp1d(t_old, x, kind='linear', fill_value='extrapolate', axis=0)

        return f(t_new)

def compute_range_meas(
    df: pd.DataFrame,
    max_value: float = 0,
) -> np.ndarray:
        """Compute the range measurement based on the timestamp intervals.

        Returns
        -------
        np.ndarray
            Range measurements.
        """
        # Speed of light
        c = 299702547 

        del_t1 = df['rx2'] - df['tx1']
        del_t2 = df['tx2'] - df['rx1']
        while ((del_t1 < 0).any() or (del_t2 < 0).any()) and max_value > 0:
            del_t1[del_t1 < 0] += max_value
            del_t2[del_t2 < 0] += max_value
        while ((del_t1 > max_value).any() or (del_t2 > max_value).any()) and max_value > 0:
            del_t1[del_t1 > max_value] -= max_value
            del_t2[del_t2 > max_value] -= max_value
        if 'tx3' in df.columns:
            del_t3 = df['rx3'] - df['rx2']
            del_t4 = df['tx3'] - df['tx2']
            while ((del_t3 < 0).any() or (del_t4 < 0).any()) and max_value > 0:
                del_t3[del_t3 < 0] += max_value
                del_t4[del_t4 < 0] += max_value
            while ((del_t3 > max_value).any() or (del_t4 > max_value).any()) and max_value > 0:
                del_t3[del_t3 > max_value] -= max_value
                del_t4[del_t4 > max_value] -= max_value
            range = 0.5 * c / 1e9 * \
                    (del_t1 - (del_t3 / del_t4) * del_t2)
        else:
            range = 0.5 * c / 1e9 * \
                    (del_t1 - del_t2)

        return range

def get_bias(df) -> float:
        """Get the ranging bias.

        Parameters
        ----------
        df: pd.dataframe
            Dataframe containing range and ground truth range (gt_range).

        Returns
        -------
        pd.dataframe
            The computed bias.
        """
        return df['range'] - df['gt_range']
    
def compute_distance_two_bodies(
    r_0w_a: np.ndarray,
    r_1w_a: np.ndarray,
    C_a0: np.ndarray = SO3.identity(),
    r_t0_0: np.ndarray = np.zeros(3),
    C_a1: np.ndarray = SO3.identity(),
    r_t1_1: np.ndarray = np.zeros(3),
) -> float:
    """Compute the distance between two points on rigid bodies with offsets from
    the body frames' origin.

    Parameters
    ----------
    r_0w_a : np.ndarray
        Position of the rigid body 0's origin in the world frame.
    r_1w_a : np.ndarray
        Position of the rigid body 1's origin in the world frame.
    C_a0 : np.ndarray, optional
        Orientation of Robot 0 parametrized as a DCM, by default SO3.identity()
    r_t0_0 : np.ndarray, optional
        The position of the point on Robot 0 in the body frame, by default np.zeros(3).
    C_a1 : np.ndarray, optional
        Orientation of Robot 1 parametrized as a DCM, by default SO3.identity()
    r_t1_1 : np.ndarray, optional
        The position of the point on Robot 1 in the body frame, by default np.zeros(3).

    Returns
    -------
    float
        Distance between the two points.
    """
    return np.linalg.norm(
        C_a0 @ r_t0_0
        + r_0w_a
        - r_1w_a
        - C_a1 @ r_t1_1
    )
    
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx