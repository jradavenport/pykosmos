"""
Functions that work to identify spectral features, and fit them for
wavelength calibration.

The work flow is: `identify` functions are used to either manually or
automatically find features (e.g. arclines at known wavelengths), and then
`fit_wavelength` simply interpolates.

IMPROVEMENT NEEDED: some form of reidentify, which takes a very close
solution and does simple (affine?) scaling.
"""

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, interp1d
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler, gaussian_smooth
from specutils.utils.wcs_utils import air_to_vac as a2v
import os
import pandas as pd

__all__ = ['identify_widget', 'loadlinelist', 'identify_nearest',
           'identify_dtw', 'find_peaks', 'fit_wavelength', 'air_to_vac']

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


def find_peaks(wave, flux, pwidth=10, pthreshold=0.97, minsep=1):
    """
    Given a slice thru an arclamp image, find the significant peaks.
    Originally from PyDIS

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Wavelength (could be approximate)
    flux : `~numpy.ndarray`
        Flux
    pwidth : float (default=10)
        the number of pixels around the "peak" to fit over
    pthreshold : float (default = 0.97)
        Peak threshold, between 0 and 1
    minsep : float (default=1)
        Minimum separation

    Returns
    -------
    Peak Pixels, Peak Wavelengths
    """
    # sort data, cut top x% of flux data as peak threshold
    flux_thresh = np.percentile(flux, pthreshold*100)

    # find flux above threshold
    high = np.where((flux >= flux_thresh))[0]

    # find  individual peaks (separated by > 1 pixel)
    # this is horribly ugly code... but i think works
    pk = high[1:][((high[1:]-high[:-1]) > minsep)]

    # offset from start/end of array by at least same # of pixels
    pk = pk[pk > pwidth]
    pk = pk[pk < (len(flux) - pwidth)]

    pcent_pix = np.zeros_like(pk, dtype='float')
    wcent_pix = np.zeros_like(pk, dtype='float')

    # for each peak, fit a gaussian to find center
    for i in range(len(pk)):
        xi = wave[pk[i] - pwidth:pk[i] + pwidth]
        yi = flux[pk[i] - pwidth:pk[i] + pwidth]

        pguess = (np.nanmax(yi), np.nanmedian(flux), float(np.nanargmax(yi)), 2.)
        try:
            popt, pcov = curve_fit(_gaus, np.arange(len(xi), dtype='float'),
                                   yi, p0=pguess)

            # the gaussian center of the line in pixel units
            pcent_pix[i] = (pk[i]-pwidth) + popt[2]
            # and the peak in wavelength units
            wcent_pix[i] = xi[np.nanargmax(yi)]

        except RuntimeError:
            pcent_pix[i] = float('nan')
            wcent_pix[i] = float('nan')

    wcent_pix, ss = np.unique(wcent_pix, return_index=True)
    pcent_pix = pcent_pix[ss]
    okcent = np.where((np.isfinite(pcent_pix)))[0]
    return pcent_pix[okcent], wcent_pix[okcent]


def loadlinelist(file):
    """
    Load a list of arclamp lines from the supplied library of files in the
    directory: pykosmos/resources/linelists.

    Note: this directory was mostly taken from IRAF.
    https://github.com/joequant/iraf/tree/master/noao/lib/linelists

    Parameters
    ----------
    file : str
        name of linelist to load

    Returns
    -------
    numpy array of arclines
    """

    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'resources', 'linelists')

    if not os.path.isfile(os.path.join(dir, file)):
        msg2 = "No valid linelist file found at: " + os.path.join(dir, file)
        raise ValueError(msg2)

    # astropy Tables try too hard here, dont have the control to only read a single column...
    # so we're switching to Pandas for this step!
    # henear_tbl = Table.read(file, names=('wave', 'name'), format='ascii')
    # henear_tbl['wave'].unit = u.AA
    # arc = henear_tbl['wave']

    df = pd.read_table(os.path.join(dir, file), usecols=(0,), names=('wave',),
                       delim_whitespace=True, comment='#')
    arc = df['wave'].values  # * u.angstrom
    return arc


def identify_nearest(arcspec, wapprox=None, linelist=None, linewave=None,
                     autotol=25, silent=False):
    """
    Identify arc lines using a simple greedy "nearest neighbor" approach.
    Requires an approximate wavelength solution (e.g. as provided by
    image header keywords). Peaks are first detected in the 1d spectrum.
    Starting from the center of the spectrum, the closest lines within a
    tolerance are picked. A linear interpolation solution is iteratively
    fit with each successive line added.

    Parameters
    ----------
    arcspec : Spectrum1D
        the 1d spectrum of the arc lamp to be fit.
    wapprox : astropy Quantity, or None
        the approximate wavelenth solution, as e.g. provided by the
        image header. Must have sensible units, like Angstroms.
        NOTE: If set to None, assumes the `arcspec` object has the
        approximate wavelength axis.
    linelist : str, optional
        name of linelist to load, is passed to `loadlinelist()`
    linewave : numpy array or None, optional
        Optionally pass an array of arclines to fit, as returned by e.g.
        `loadlinelist()`
        NOTE: either linelist or linewave must be provided.
    autotol : int, optional (default is 25)
        the tolerance in pixel units to allow nearest matches within.
    silent : bool, optional (default is False)
        suppress a few helpful summary messages

    Returns
    -------
    xpoints, wpoints : the pixel and wavelength values of the
        successfully identified lines.
    """

    if linelist is not None:
        linewave = loadlinelist(linelist)

    if linewave is None:
        msg_fail = '''
        linewave must be an array of known line wavelengths.'''
        raise ValueError(msg_fail)

    # the fluxes within the arc-spectrum
    flux = arcspec.flux.value
    
    if wapprox is not None:
        xpixels = wapprox
    else:
        xpixels = arcspec.spectral_axis
        
    # in this mode, the xpixel input array is actually the approximate
    # wavelength solution (e.g. from the header info)
    pcent_pix, wcent_pix = find_peaks(xpixels.value, flux, pwidth=10,
                                      pthreshold=0.97)

    # A simple, greedy, line-finding solution.
    # Loop thru each detected peak, from center outwards. Find nearest
    # known list line. If no known line within tolerance, skip

    # PLAN: predict solution w/ spline, start in middle, identify nearest match,
    # every time there's a new match, recalc the spline sol'n, work all the way out
    # this both identifies lines, and has byproduct of ending w/ a spline model

    xpoints = np.array([], dtype=float)  # pixel line centers
    wpoints = np.array([], dtype=float)  # wavelength line centers

    # find center-most lines, sort by dist from center pixels
    ss = np.argsort(np.abs(wcent_pix - np.nanmedian(xpixels.value)))

    # 1st guess is the peak locations in the wavelength units as given by user
    wcent_guess = wcent_pix

    for i in range(len(pcent_pix)):
        # if there is a match within the tolerance
        if np.nanmin(np.abs(wcent_guess[ss][i] - linewave)) < autotol:
            # add corresponding pixel and known wavelength to output vectors
            xpoints = np.append(xpoints, pcent_pix[ss[i]])
            wpoints = np.append(wpoints, linewave[np.nanargmin(np.abs(wcent_guess[ss[i]] - linewave))])

            # start guessing new wavelength model after first few lines identified
            if len(wpoints) > 4:
                xps = np.argsort(xpoints)
                # spl = UnivariateSpline(xpoints[xps], wpoints[xps], ext=0, k=3, s=1e3)
                # wcent_guess = spl(pcent_pix)
                spl = interp1d(xpoints[xps], wpoints[xps], kind=1, fill_value='extrapolate')
                wcent_guess = spl(pcent_pix)

    inrng = sum((linewave >= np.nanmin(wcent_guess)) & (linewave <= np.nanmax(wcent_guess)))
    if not silent:
        print(str(len(wpoints)) + ' lines matched from ' + str(inrng) +
              ' within estimated range.')

    # at this point we have (xpoints, wpoints), so next is generic interpolation.
    #       should this be part of another routine, used by all identify modes?

    # sort the points, just in case the method (or prev run) returns in weird order
    srt = np.argsort(xpoints)
    xpoints = xpoints[srt]
    wpoints = wpoints[srt]

    return xpoints, wpoints * xpixels.unit


def identify_widget(arcspec, silent=False):
    """
    Interactive version of the Identify GUI, specifically using ipython widgets.

    Each line is roughly identified by the user, then a Gaussian is fit to
    determine the precise line center. The reference value for the line is then
    entered by the user.

    When finished, the output lines should usually be passed in a new Jupter
    notebook cell to `identify` for determining the wavelength solution:
    >>>> xpl,wav = identify_widget(arcspec) # doctest: +SKIP
    >>>> fit_spec = fit_wavelength(obj_spec, xpl, wav) # doctest: +SKIP

    NOTE: Because of the widgets used, this is not well suited for inclusion in
    pipelines, and instead is ideal for interactive analysis.

    Parameters
    ----------
    arcspec : Spectrum1D
        the 1d spectrum of the arc lamp to be fit.
    silent : bool, optional (default is False)
        Set to True to silence the instruction print out each time.

    Returns
    -------
    The pixel locations and wavelengths of the identified lines:
    pixel, wavelength
    """

    # the fluxes & pixels within the arc-spectrum
    flux = arcspec.flux.value
    xpixels = arcspec.spectral_axis.value

    msg = '''
    Instructions:
    ------------
    0) For proper interactive widgets, ensure you're using the Notebook backend
    in the Jupyter notebook, e.g.:
        %matplotlib notebook
    1) Click on arc-line features (peaks) in the plot. The Pixel Value box should update.
    2) Enter the known wavelength of the feature in the Wavelength box.
    3) Click the Assign button, a red line will be drawn marking the feature.
    4) When you've identified all your lines, stop the interaction for (or close) the figure.'''

    if not silent:
        print(msg)

    xpxl = []
    waves = []

    # Create widgets, two text boxes and a button
    xval = widgets.BoundedFloatText(
        value=5555.0,
        min=np.nanmin(xpixels),
        max=np.nanmax(xpixels),
        step=0.1,
        description='Pixel Value (from click):',
        style={'description_width': 'initial'})

    linename = widgets.Text(  # value='Enter Wavelength',
        placeholder='Enter Wavelength',
        description='Wavelength:',
        style={'description_width': 'initial'})

    button = widgets.Button(description='Assign')

    fig, ax = plt.subplots(figsize=(9, 3))

    # Handle plot clicks
    def onplotclick(event):
        # try to fit a Gaussian in the REGION (rgn) near the click
        rgn = np.where((np.abs(xpixels - event.xdata) <= 5.))[0]
        try:
            sig_guess = 3.
            p0 = [np.nanmax(flux[rgn]), np.nanmedian(flux), event.xdata, sig_guess]
            popt, _ = curve_fit(_gaus, xpixels[rgn], flux[rgn], p0=p0)
            # Record x value of click in text box
            xval.value = popt[2]
        except RuntimeError:
            # fall back to click itself if that doesnt work
            xval.value = event.xdata
        return

    fig.canvas.mpl_connect('button_press_event', onplotclick)

    # Handle button clicks
    def onbuttonclick(_):
        xpxl.append(xval.value)
        waves.append(float(linename.value))
        print(xpxl, waves)

        ax.axvline(xval.value, lw=1, c='r', alpha=0.7)
        return

    button.on_click(onbuttonclick)

    # Do the plot
    ax.plot(xpixels, flux)
    plt.draw()

    # Display widgets
    display(widgets.HBox([xval, linename, button]))

    # return np.array(xpxl), np.array(waves)
    return xpxl, waves


def identify_dtw(arc, ref, display=False, upsample=False, Ufactor=5,
                 step_pattern='symmetric1',
                 open_begin=False, open_end=False,
                 peak_spline=True, pthreshold=0.95):
    """
    Align an arc lamp spectrum in pixel-units to a reference spectrum
    in wavelength units using Dynamic Time Warping (DTW).

    Notes: very simple, fairly robust, but has several key limitations:
    - resulting wavelength axis may not be smooth (mapping to reference)
    - DTW fixes the first/last pixel to the start/stop of reference spectrum.

    If reference is much wider than observed lamp, this is a big problem.
    Suggest using `identify_dtw` for 1st-pass, but examine (pixel, wavelength)
    plot to see if it diverges strongly!

    This function should probably be wrapped with something instrument-
    specific, to handle known limitations and input a sensible reference.

    Parameters
    ----------
    arc : Spectrum1D object
        the observed Arc-lamp spectrum to align, as returned by e.g. BoxcarExtract
        spectral axis typically has units of pixels.
    ref : Spectrum1D object
        reference spectrum to match to
    upsample : bool (default=True)
        do the DTW on an up-sampled version of the observed arc and reference
        spectra using a gaussian smooth. Linearlly down-sample result.
        WARNING: doesn't like backwards wavelength axis for either arc or ref...
    Ufactor : int (default=5)
        the factor to up-sample both the ref and arc spectra by.
        UPGRADE IDEA: up-sample the arc and ref by different factors?
    peak_spline : bool (default=True)
        After DTW match has been run on the whole spectrum, select pixels
        with peaks in arc spectrum and fit a spline. Final wavelength
        solution returned comes from spline fit. This is often useful since
        only the peaks carry "information" in the DTW match, and the
        wavelength solution can be very non-smooth between peaks. This
        mode essentially uses DTW to do peak-wavelength identification.
        If you don't like the spline default, set to False and do your
        own interpolation of the line wavelengths.
        NEED TO UPDATE DESCRIPTION HERE... RETURNS ONLY PEAKS, LIKE OTHER IDENTIFY MODES!
    pthreshold : float (default=0.95)
        Number between 0 and 1, the threshold to use in defining
        "peaks" in the spectrum if `peak_spline=True`.
    display : bool (optional, default=False)
        if set, produce a plot of pixel vs wavelength solution

    Returns
    -------
    The pixel locations and wavelengths of the identified features
    (lines or the whole spectrum):
        pixel, wavelength
    """

    # use Dynamic Time Warping!
    # https://doi.org/10.18637/jss.v031.i07
    import dtw

    # IMPROVEMENT NEEDED: check that both spectra are sorted, FCR returns NaNs if not.
    # if not, sort them, and then return back in the original form

    if upsample is True:
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

    # Normalize by the mean, seems to help if amplitudes are vastly different.
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

    if peak_spline:
        pks = np.where((arc.flux.value >= np.percentile(arc.flux.value, pthreshold*100)))[0]

        spl = UnivariateSpline(pks, wav_guess[pks],
                               ext=0, k=3, s=len(pks)*10)
        wav_guess = spl(arc.spectral_axis.value)

    # plot pixel observed vs wavelength matched
    if display:
        plt.plot(arc.spectral_axis.value, wav_guess)
        if peak_spline:
            plt.scatter(pks, wav_guess[pks], alpha=0.8)
        plt.xlabel(arc.spectral_axis.unit)
        plt.ylabel(ref.spectral_axis.unit)

    # to fit w/ other "identify" methods, return (pixels, wavelength)
    xpoints, wpoints = arc.spectral_axis.value, wav_guess * ref.spectral_axis.unit

    if peak_spline:
        xpoints, wpoints = pks, wav_guess[pks] * ref.spectral_axis.unit

    return xpoints, wpoints


def fit_wavelength(spec, xpoints, wpoints, display=False,
                   mode='poly', deg=7, GPRscale=101,
                   returnpoints=False, returnvar=False):
    """
    Fit the wavelength solution from a series of (pixel, Wavelength)
    datapoints, and apply it a spectrum

    Parameters
    ----------
    spec : Spectrum1D
        the object spectrum to have a new wavelength axis added
    xpoints : array-like object
        the pixel values of identified arcline features
    wpoints : astropy Quantity
        the corresponding wavelengths for the identiifed pixels.
        NOTE: Must have sensible units like angstroms, which will be
        applied to the resulting spectrum.
    display : bool, optional (default is False)
        should we plot the (pixel,wavelength) fit residuals?
    mode : str, ['poly', 'spline', 'interp', 'gp']
        which fitting mode should be used? (Default is 'poly')
        Select between Polynomial, UnivariateSpline, Interpolation, and
        a Gaussian Process (via `george`, using ExpSquaredKernel)
    deg : int, optional (default is 7)
        if mode='poly', set the polynomial degree to use
        if mode='interp', set the interpolation degree (passed as
        `kind=deg` to `interp1d()`).
    GPRscale : int, optional (default is 101)
        If mode='gp', the Rscale parameter to use with ExpSquaredKernel
    returnpoints : bool, optional (default is False)
        If set, return just the fit values corresponding to the input
        (xpoints, wpoints)
    returnvar : bool, optional (default is False)
        If set and mode='gp', additionally return the variance on the
        resulting wavelength axis

    Returns
    -------
    outspec : Sepctrum1D object
        the same input spectrum, but with the newly fit wavelength
        axis added.

    if returnvar=True, then return:
        outspec, wavelength_variance

    """

    # Improvements Needed
    # ------------
    # should the fit and apply steps be separated?

    # sort, just in case
    srt = np.argsort(xpoints)
    xpt = np.array(xpoints)[srt]
    wpt = np.array(wpoints.value)[srt]
    fpt = np.zeros_like(xpt)  # the fit wavelength points

    if mode.lower() == 'poly':
        fit = np.polyfit(xpt, wpt, deg)
        wavesolved = np.polyval(fit, spec.spectral_axis.value)
        fpt = np.polyval(fit, xpt)

    if mode.lower() == 'spline':
        spl = UnivariateSpline(xpt, wpt, ext=0, k=3, s=1e3)
        wavesolved = spl(spec.spectral_axis.value)
        fpt = spl(xpt)

    if mode.lower() == 'interp':
        spl = interp1d(xpt, wpt, kind=deg, fill_value='extrapolate')
        wavesolved = spl(spec.spectral_axis.value)
        fpt = spl(xpt)

    if mode.lower() == 'gp':
        # assume 1/2 pixel precision of centering arc lines (prob better actually)
        yerr = np.ones_like(xpt) * np.mean(np.abs(np.diff(spec.spectral_axis.value))) / 2

        # follow BASIC tutorial from "george"
        # https://george.readthedocs.io/en/latest/tutorials/first/
        import george
        from george import kernels
        from scipy.optimize import minimize

        # Rscale = 100 # the magic scale param... hopefully OK, YMMV
        kernel = np.var(wpt) * kernels.ExpSquaredKernel(GPRscale)
        gp = george.GP(kernel, fit_mean=True)
        gp.compute(xpt, yerr)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(wpt)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(wpt)

        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, method='L-BFGS-B')
        # print(result)
        gp.set_parameter_vector(result.x)

        wavesolved, wavesolved_var = gp.predict(wpt, spec.spectral_axis.value, return_var=True)
        fpt = gp.predict(wpt, xpt, return_var=False)

    if display:
        plt.scatter(xpt, wpt - fpt)
        plt.xlabel('Xpoints')
        plt.ylabel('Residuals')

    outspec = Spectrum1D(spectral_axis=wavesolved * wpoints.unit,
                         flux=spec.flux,
                         uncertainty=spec.uncertainty
                         )
    if returnpoints:
        return fpt

    if returnvar is True and mode.lower() == 'gp':
        # since there's no way to package uncertainty/variance w/ the Spectrum1D object currently
        return outspec, wavesolved_var

    return outspec


def air_to_vac(spec):
    """
    Simple wrapper for the `air_to_vac` calculation within `specutils.utils.wcs_utils`

    Parameters
    ----------
    spec : Spectrum1D object

    Returns
    -------
    Spectrum1D object with spectral_axis converted from air to vaccum units

    """
    new_wave = a2v(spec.wavelength)
    outspec = Spectrum1D(spectral_axis=new_wave,
                         flux=spec.flux,
                         uncertainty=spec.uncertainty
                         )
    return outspec
