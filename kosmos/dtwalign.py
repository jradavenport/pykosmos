"""
use Dynamic Time Warping to align a spectrum to a ref
"""

import dtw
import numpy as np
import matplotlib.pyplot as plt
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler, gaussian_smooth


__all__ = ['dtwalign']


def dtwalign(arc, ref, display=False, upsample=True, Ufactor=5,
             step_pattern='symmetric1',open_begin=False, open_end=False):
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
    upsample : bool (optional (default=True)
        do the DTW on an up-sampled version of the observed arc and reference
        spectra using a gaussian smooth. Linearlly down-sample result.
    Ufactor : int (default=5)
        the factor to up-sample both the ref and arc spectra by.
        UPGRADE IDEA: up-sample the arc and ref by different factors?
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
    # Normalize by the mean, seems to help if amplitudes are different.
    # x1, x2 = arc.flux.value / np.nanmean(arc.flux.value),  ref.flux.value / np.nanmean(ref.flux.value)

    if upsample:
        FCR = FluxConservingResampler()
        spec1 = FCR(arc, np.linspace(arc.spectral_axis.value.min(),
                                     arc.spectral_axis.value.max(),
                                     len(arc.spectral_axis.value) * Ufactor) * arc.spectral_axis.unit)
        x1 = gaussian_smooth(spec1, stddev=2)

        spec2 = FCR(ref, np.linspace(ref.spectral_axis.value.min(),
                                     ref.spectral_axis.value.max(),
                                     len(ref.spectral_axis.value) * Ufactor) * ref.spectral_axis.unit)
        x2 = gaussian_smooth(spec2, stddev=2)

    else:
        x1, x2 = arc, ref

    alignment = dtw.dtw(x1.flux.value / np.nanmean(x1.flux.value),
                        x2.flux.value / np.nanmean(x2.flux.value),
                        keep_internals=True, step_pattern=step_pattern,
                        open_begin=open_begin, open_end=open_end)

    wav_guess = np.zeros_like(x1.spectral_axis.value)
    # brute force step through each pixel
    for k in range(len(wav_guess)):
        # how many reference wavelengths match (not 1-to-1 sometimes)
        refok = np.where((alignment.index1 == k))[0]
        if len(refok) > 0:
            # if there are multiple points that are matched, take average of wavelengths
            wav_guess[k] = np.nanmean(x2.spectral_axis.value[alignment.index2[refok]])

    if upsample:
        # if upsample is being used, linearly downsample to original pixel axis
        wav_guess0 = wav_guess
        wav_guess = np.interp(arc.spectral_axis.value, x1.spectral_axis.value, wav_guess0)

    # plot pixel observed vs wavelength matched
    if display:
        plt.plot(arc.spectral_axis.value, wav_guess)
        plt.xlabel(arc.spectral_axis.unit)
        plt.ylabel(ref.spectral_axis.unit)
        plt.show()

    # return a Spectrum1D object, but switch to the new spectral axis
    spec = Spectrum1D(spectral_axis=wav_guess * ref.spectral_axis.unit,
                      flux=arc.flux,
                      uncertainty=arc.uncertainty)

    return spec
