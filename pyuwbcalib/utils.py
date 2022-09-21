# Move interpolate function from postprocess to here
# SAME with unwrapping methods
# Move "find machine from tag" from postprocess to here
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def save(obj, filename="file.pickle") -> None:
    """_summary_

    Parameters
    ----------
    obj : _type_
        _description_
    filename : str, optional
        _description_, by default "file.pickle"
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file)

def load(filename='file.pickle') -> object:
    """_summary_

    Parameters
    ----------
    filename : str, optional
        _description_, by default 'file.pickle'

    Returns
    -------
    _type_
        _description_
    """
    with open(filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        
    return data

def set_plotting_env() -> None:
    """_summary_
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
