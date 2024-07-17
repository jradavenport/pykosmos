import numpy as np
import matplotlib.pyplot as plt
from specutils import Spectrum1D
from astropy.nddata import CCDData
from astropy.nddata import StdDevUncertainty
from specreduce.tracing import FitTrace, ArrayTrace
from astropy.modeling import models
from specreduce.background import Background as Bkgd
from specreduce.extract import BoxcarExtract as BExtract
from warnings import warn

__all__ = ['trace', 'BoxcarExtract']


def _gaus(x, a, b, x0, sigma):
    """
    Define a simple Gaussian curve

    Could maybe be swapped out for astropy.modeling.models.Gaussian1D

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def trace(img, nbins=20, guess=None, window=None,
          Saxis=0, Waxis=1, display=False, ax=None, 
          trace_model=models.Spline1D(), return_trace=False):
    """
    Trace the spectrum aperture in an image

    Assumes wavelength axis is along the X, spatial axis along the Y.
    Chops image up in bins along the wavelength direction, fits a Gaussian
    within each bin to determine the spatial center of the trace. Finally,
    draws a cubic spline through the bins to up-sample trace along every X pixel.

    Parameters
    ----------
    img : 2d numpy array, or CCDData object
        This is the image to run trace over
    nbins : int, optional
        number of bins in wavelength (X) direction to chop image into. Use
        fewer bins if trace is having difficulty, such as with faint
        targets (default = 20, but minimum must be 4)
    guess : int, optional
        A guess at where the desired trace is in the spatial direction (Y). If set,
        overrides the normal max peak finder. Good for tracing a fainter source if
        multiple traces are present.
    window : int, optional
        If set, only fit the trace within a given region around the guess position.
        Useful for tracing faint sources if multiple traces are present, but
        potentially bad if the trace is substantially bent or warped.
    display : bool, optional
        If set to true display the trace over-plotted on the image
    Saxis : int, optional
        Set which axis is the spatial dimension. For DIS, Saxis=0
        (corresponds to NAXIS2 in header). For KOSMOS, Saxis=1.
        (Default is 0)
    Waxis : int, optional
        Set which axis is the wavelength dimension. For DIS, Waxis=1
        (corresponds to NAXIS1 in the header). For KOSMOS, Waxis=0.
        (Default is 1)
        NOTE: if Saxis is changed, Waxis will be updated, and visa versa.
    ax : matplotlib axes or subplot object, optional
        axes or subplot to be plotted onto. If not specified one will be 
        created. (Default is None)
    trace_model : astropy.modeling model, optional
        passed to FitTrace (Default=models.Spline1D())
    return_trace : bool, optional 
        If True, return the FitTrace object. If False, return the np.ndarray,
        as was done in previous versions. (Default is False)

    Returns
    -------
    my : array or trace
        The spatial (Y) positions of the trace, interpolated over the
        entire wavelength (X) axis

    """

    warn("PyKOSMOS trace is now a wrapper for functions within specreduce.tracing, and is kept for backwards compatibility. More features are available from specreduce.tracing.", 
         DeprecationWarning, stacklevel=2)

    # Require at least 4 big bins along the trace to define shape. Sometimes can get away with very few
    if nbins < 4:
        raise ValueError('nbins must be >= 4')

    img_to_trace = img

    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0
    
        # if swapped (Saxis=1,Waxis=0), need to transpose before pass to specreduce trace,
        # which doesn't support swapped axis
        if isinstance(img, (CCDData)):
            img_to_trace = img.data.T
        else:
            img_to_trace = img.T

    # The Specreduce FitTrace is based on the original version of this function.
    # Call this as a general purpose wrapper
    tt = FitTrace(img_to_trace, bins=nbins, peak_method='gaussian', guess=guess, window=window, 
                  trace_model=trace_model)
    my = tt.trace.data
    mx = np.arange(len(my))

    if display is True:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        im = ax.imshow(img, origin='lower', aspect='auto', cmap=plt.cm.Greys_r)
        im.set_clim(np.percentile(img, (5, 98)))
        if Waxis == 1:
            # ax.scatter(xbins, ybins, alpha=0.5)
            ax.plot(mx, my)
        if Waxis == 0:
            # ax.scatter(ybins, xbins, alpha=0.5)
            ax.plot(my, mx)
        plt.show()

    # if you want the output from FitTrace
    if return_trace:
        return tt
    # otherwise just return the array, as the old behavior
    else:
        return my


def BoxcarExtract(img, trace_line, apwidth=8, skysep=3, skywidth=7, skydeg=0,
                  Saxis=0, Waxis=1, display=False, ax=None,
                  bkgd_sub=True, return_extract=False, return_sky=False):
    """
    This is nearly identical to specreduce.extract.BoxcarExtract,
    because that was based on the same PyDIS source code as this.

    1. Extract the spectrum using the trace. Simply add up all the flux
    around the aperture within a specified +/- width.

    Note: implicitly assumes wavelength axis is perpendicular to
    the trace.

    2. Fits a polynomial to the sky at each column

    3. Computes the uncertainty in each pixel

    Parameters
    ----------
    img : CCDData object
        This is the image to run extract over
    trace_line : 1-d array or specreduce trace object
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from trace. Can be
        either np.ndarray or a specreduce trace object.
    apwidth : int, optional
        The width along the Y axis on either side of the trace to extract.
        Note: a fixed width is used along the whole trace.
        (default is 8 pixels, must be at least 1 pixel)
    skysep : int, optional
        The separation in pixels from the aperture to the sky window.
        (Default is 3, must be at least 1 pixel)
    skywidth : int, optional
        The width in pixels of the sky windows on either side of the
        aperture. (Default is 7, must be at least 1 pixel)
    skydeg : int, optional
        The polynomial order to fit between the sky windows.
        (Default is 0)
    Saxis : int, optional
        Set which axis is the spatial dimension. For DIS, Saxis=0
        (corresponds to NAXIS2 in header). For KOSMOS, Saxis=1.
        (Default is 0)
    Waxis : int, optional
        Set which axis is the wavelength dimension. For DIS, Waxis=1
        (corresponds to NAXIS1 in the header). For KOSMOS, Waxis=0.
        (Default is 1)
        NOTE: if Saxis is changed, Waxis will be updated, and visa versa.
    ax : matplotlib axes or subplot object, optional
        axes or subplot to be plotted onto. If not specified one will be 
        created. (Default is None)
    bkgd_sub : bool, optional
        If True, subtract the specreduce.background.Background from the 
        image before running the extraction. This means the sky does not 
        need to be subtracted later. (Default is True)
    return_extract: bool, optional
        If True, return the result of specreduce.extract.BoxcarExtract.
        If False, return Spectrum1D objects for the extracted spectrum
        and the skyspec estimated from Background. This is consistent
        with previous function behavior. 
        (Default is False)
        NOTE: If `bkgd_sub=True` and  `return_extract=False` (default behavior) 
        function will return skyspec, but user should probably NOT then 
        subtract the sky from the spectrum again.
    return_sky: bool, optional
        If True, return sky (background) as well as spectrum
        (Default is False)
        NOTE: this default has changed, possibly breaking old code!
        

    Returns
    -------
    spec : Spectrum1D object
        The extracted spectrum
    skyspec : Spectrum1D object
        The sky spectrum used in the extraction process

    """

    warn("PyKOSMOS BoxcarExtract is now a wrapper for functions within specreduce.extract.", 
                  DeprecationWarning, stacklevel=2)
    
    img_to_ext = img
    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0

        # if swapped (Saxis=1,Waxis=0), need to transpose before pass to specreduce functions,
        # which don't support swapped axis correctly yet
        if isinstance(img, (CCDData)):
            img_to_ext = img.data.T
        else:
            img_to_ext = img.T

    if apwidth < 1:
        raise ValueError('apwidth must be >= 1')
    if skysep < 1:
        raise ValueError('skysep must be >= 1')
    if skywidth < 1:
        raise ValueError('skywidth must be >= 1')

    # need to convert trace_line into a new trace object
    if isinstance(trace_line, (np.ndarray)):
        trace_obj = ArrayTrace(img_to_ext, trace_line)
        if len(trace_obj.trace.data) != len(trace_line):
            raise Exception('Error: Length of trace does not equal Waxis shape.')
    else:
        trace_obj = trace_line

    if len(trace_obj.trace.data) != img.shape[Waxis]:
        raise Exception('Error: Length of trace does not equal Waxis shape.')
    
    # disable disp_axis and crossdisp_axis, since they not being broadcast correctly yet
    bg = Bkgd.two_sided(img_to_ext, trace_obj, separation=skysep, width=skywidth)
                        # disp_axis=Waxis, crossdisp_axis=Saxis)

    skyspec = bg.bkg_spectrum()

    # should we subtract the background (sky) before extraction?
    if bkgd_sub:
        img_to_ext = img_to_ext - bg

    # disable disp_axis and crossdisp_axis, since they not being broadcast correctly yet
    extr = BExtract(img_to_ext, trace_obj, width=apwidth) 
                    # disp_axis=Waxis, crossdisp_axis=Saxis)

    # this is where we can add the Optimal Extraction option next

    # based on aperture phot err description by F. Masci, Caltech:
    # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
    sigB = 1 # stddev in the background data... not available yet from specreduce!
    N_B = float(skywidth * 2)  # number of bkgd pixels
    N_A = float(apwidth * 2) # number of aperture pixels
    fluxerr = np.sqrt(extr.spectrum.flux.value + (N_A + N_A**2. / N_B) * (sigB**2.))

    spec = Spectrum1D(flux=extr.spectrum.flux, 
                      uncertainty=StdDevUncertainty(fluxerr) 
                      )

    if return_extract:
        if return_sky:
            return extr, bg
        else:
            return extr
    else:
        if return_sky:
            return spec, skyspec
        else:
            return spec
