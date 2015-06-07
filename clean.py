import numpy as np
#from scipy import fft, arange
from scipy import stats as scistats

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

def get_fft_features(bidtimes):
    """Return basic features extracted from fft of the bid list"""
    ntimes=len(bidtimes)
    magnitudes = np.abs(np.fft.rfft(bidtimes)/ntimes)
    freqs = np.abs(np.fft.fftfreq(ntimes, 1.0)[:ntimes//2+1])
    cent = np.sum(magnitudes*freqs) / np.sum(magnitudes)
    spectral_flatness = scistats.gmean(magnitudes)/np.mean(magnitudes)
    return cent, np.std(freqs), spectral_flatness, np.ptp(magnitudes)


if __name__ == "__main__":
    time_list = [-1, 0.3, 2.1, 3.2, 4.3, 5.4, 5.2, 100000,1000000]
    #print(get_linearfit_features(time_list))
    #print(get_linearfit_features([1, 2]))
    print(get_fft_features(time_list))
