"""
use Dynamic Time Warping to align a spectrum to a ref
"""

import dtw
import numpy as np

__all__ = ['dtwalign']


def dtwalign(pix, flux, wav_ref, flux_ref):
    """

    Parameters
    ----------
    pix
    flux
    wav_ref
    flux_ref

    Returns
    -------

    """
    # do the actual DTW matching... there's many settings I need to explore...
    alignment = dtw.dtw(flux / np.nanmean(flux), flux_ref / np.nanmean(flux_ref),
                        keep_internals=True, step_pattern='symmetric1',
                        open_begin=False, open_end=False)

    wav_guess = np.zeros_like(pix)
    for k in range(len(flux)):
        refOK = np.where((alignment.index1 == k))[0]
        if len(refOK) > 0:
            wav_guess[k] = np.mean(wav_ref[alignment.index2[refOK]])

    return wav_guess
