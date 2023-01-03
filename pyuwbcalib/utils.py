import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd 
from scipy.interpolate import interp1d

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
        if 'tx3' in df.columns:
            del_t3 = df['rx3'] - df['rx2']
            del_t4 = df['tx3'] - df['tx2']
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