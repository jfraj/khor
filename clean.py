import numpy as np


#import warnings
#warnings.simplefilter('ignore', np.RankWarning)


def get_linearfit_features(val_list):
    """Return features associated to the linear fit of a list."""
    if len(val_list) == 1:
        return [-1, val_list[0], -1]
    fit_tup, res, rk, sv, rc = np.polyfit(range(len(val_list)),
                                          val_list, 1, full=True)
    try:
        res = res[0]
    except IndexError:
        res = -1
    return fit_tup[0], fit_tup[1], res

if __name__ == "__main__":
    print(get_linearfit_features([-1, 0.3, 2.1, 3.2, 4.3, 5.4, 5.2, 100000,1000000]))
    #print(get_linearfit_features([1, 2]))
