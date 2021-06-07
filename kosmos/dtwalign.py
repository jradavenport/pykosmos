"""
use Dynamic Time Warping to align a spectrum to a ref
"""

import dtw
import numpy as np
import matplotlib.pyplot as plt
from specutils import Spectrum1D

__all__ = ['dtwalign']


def dtwalign(arc, ref, display=False):
    """
    Align an arc lamp spectrum in pixel-units to a reference spectrum
    in wavelength units using Dynamic Time Warping (DTW).

    Notes: very simple, fairly robust, but has several key limitations:
    - resulting wavelength axis may not be smooth (mapping to reference)
    - DTW fixes the first/last pixel to the start/stop of reference spectrum.
        If reference is much wider than observed lamp, this is a big problem.
        Suggest using `dtwalign` for 1st-pass, but examine (pixel, wavelength)
        plot to see if diverges strongly!

    This function should probably be wrapped with something instrument-
    specific, to handle known limitations and input a sensible reference.

    Parameters
    ----------
    arc : Spectrum1D object
        the Arc-lamp spectrum, as returned by e.g. BoxcarExtract
    ref : Spectrum1D object
        reference spectrum to match to
    display : bool (optional, default=False)
        if set, produce a plot of pixel vs wavelength solution

    Returns
    -------
    spec : Spectrum1D object
        The same spectrum as the arc-lamp input, but swapping the
        pixel to wavelength axis
    """

    # use Dynamic Time Warping!
    # https://doi.org/10.18637/jss.v031.i07
    # Normalize by the mean, and use the "symmetric1" pattern, both seem to help.
    alignment = dtw.dtw(arc.flux.value / np.nanmean(arc.flux.value),
                        ref.flux.value / np.nanmean(ref.flux.value),
                        keep_internals=True, step_pattern='symmetric1',
                        open_begin=False, open_end=False)

    wav_guess = np.zeros_like(arc.spectral_axis.value)
    # brute force step through each pixel
    for k in range(len(wav_guess)):
        # how many reference wavelengths match (not 1-to-1 sometimes)
        refok = np.where((alignment.index1 == k))[0]
        if len(refok) > 0:
            # if there are multiple points that are matched, take average of wavelengths
            wav_guess[k] = np.nanmean(ref.spectral_axis.value[alignment.index2[refok]])

    # plot pixel observed vs wavelength matched
    if display:
        plt.plot(arc.spectral_axis.value, wav_guess)
        plt.xlabel(arc.spectral_axis.unit)
        plt.ylabel(ref.spectral_axis.unit)
        plt.show()

    # return a Spectrum1D object, but switch to the new spectral axis
    spec = Spectrum1D(spectral_axis=wav_guess * ref.spectral_axis.unit,
                      flux=arc.flux,
                      uncertainty=arc.uncertainty
                      )

    return spec
