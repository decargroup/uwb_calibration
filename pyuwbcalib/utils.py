import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
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