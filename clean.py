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
    if cent == np.nan:
        cent = -1
    spectral_flatness = scistats.gmean(magnitudes)/np.mean(magnitudes)
    if spectral_flatness == np.nan:
        spectral_flatness = -1
    ptp = np.ptp(magnitudes)
    if ptp == np.nan:
        ptp = -1
    freq_std = np.std(freqs)
    if freq_std == np.nan:
        freq_std = -1
    linfit_m, linfit_b, linfit_r = get_linearfit_features(magnitudes[:10])
    return cent, freq_std, spectral_flatness, ptp, linfit_m, linfit_b, linfit_r


if __name__ == "__main__":
    time_list = [-1, 0.3, 2.1, 3.2, 4.3, 5.4, 5.2, 100000,1000000]
    #print(get_linearfit_features(time_list))
    #print(get_linearfit_features([1, 2]))
    print(get_fft_features(time_list))
