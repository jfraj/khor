import numpy as np


#import warnings
#warnings.simplefilter('ignore', np.RankWarning)


def get_linearfit_features(val_list):
    """Return features associated to the linear fit of a list."""
    if len(val_list) == 1:
        return [val_list[0], val_list[0]]
    return np.polyfit(range(len(val_list)), val_list, 1)

if __name__ == "__main__":
    #print(get_linearfit_features([-1, 0.3, 2.1, 3.2, 4.3, 5.4, 5.2]))
    print(get_linearfit_features([1, 2]))
