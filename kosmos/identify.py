"""
NOT EDITED FOR KOSMOS YET
"""
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from astropy.table import Table

__all__ = ['identify', 'identify_widget', 'find_peaks']


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
        Wavelength
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
    pk = high[1:][ ( (high[1:]-high[:-1]) > minsep ) ]

    # offset from start/end of array by at least same # of pixels
    pk = pk[pk > pwidth]
    pk = pk[pk < (len(flux) - pwidth)]

    pcent_pix = np.zeros_like(pk,dtype='float')
    wcent_pix = np.zeros_like(pk,dtype='float') # wtemp[pk]
    # for each peak, fit a gaussian to find center
    for i in range(len(pk)):
        xi = wave[pk[i] - pwidth:pk[i] + pwidth]
        yi = flux[pk[i] - pwidth:pk[i] + pwidth]

        pguess = (np.nanmax(yi), np.nanmedian(flux), float(np.nanargmax(yi)), 2.)
        try:
            popt,pcov = curve_fit(_gaus, np.arange(len(xi),dtype='float'), yi,
                                  p0=pguess)

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


def identify(xpixels, flux, identify_mode='',
             previous_file='',
             xpoints=[], wpoints=[],
             linewave=[], autotol=25,
             ref_wave=[], ref_flux=[], cbins=50,
             fit_mode='spline', polydeg=7,):
    """
    identify's job is to find peaks/features and identify which wavelength they are

    identify_mode: available choices are
    {'nearest', 'file', 'lines', 'crosscor'}.
    For an interactive mode, use

    identify methods to consider:
    - interactive widget (basically done)
    - nearest guess from a line list (using approximate wavelength sol'n, e.g. from header)
    - cross-corl from previous solution (best auto method)
    - automatic line hash (maybe get to)

    then it can (by default) generate a solution for the lines/features -> all xpixels
    - polynomial (easy - can use BIC to guess the order)
    - interpolation (easy, but not smooth)
    - spline (fairly simple, but fickle esp. at edges)
    - GaussianProcess

    """

    # Check that identify mode is valid

    #######
    # if identify_mode.lower() == 'interact':
    #     xpoints, wpoints = identify_widget(xpixels, flux)

    #######
    if identify_mode.lower() == 'nearest':
        if len(linewave)<1:
            msg_fail = '''
            linewave not provided! 
            For identify_mode='nearest', linewave=[] must be an array of known line wavelengths.'''
            raise ValueError(msg_fail)

        # in this mode, the xpixel input array is actually the approximate
        # wavelength solution (e.g. from the header info)
        pcent_pix, wcent_pix = find_peaks(xpixels, flux, pwidth=10, pthreshold=97)

        # A simple, greedy, line-finding solution.
        # Loop thru each detected peak, from center outwards. Find nearest
        # known list line. If no known line within tolerance, skip

        # PLAN: predict solution w/ spline, start in middle, identify nearest match,
        # every time there's a new match, recalc the spline sol'n, work all the way out
        # this both identifies lines, and has byproduct of ending w/ a spline model

        xpoints = np.array([], dtype=np.float) # pixel line centers
        wpoints = np.array([], dtype=np.float) # wavelength line centers

        # find center-most lines, sort by dist from center pixels
        ss = np.argsort(np.abs(wcent_pix - np.nanmedian(xpixels)))

        # 1st guess is the peak locations in the wavelength units as given by user
        wcent_guess = wcent_pix

        for i in range(len(pcent_pix)):
            # if there is a match within the tolerance
            if (np.nanmin(np.abs(wcent_guess[ss][i] - linewave)) < autotol):
                # add corresponding pixel and known wavelength to output vectors
                xpoints = np.append(xpoints, pcent_pix[ss[i]])
                wpoints = np.append(wpoints, linewave[np.nanargmin(np.abs(wcent_guess[ss[i]] - linewave))])

                # start guessing new wavelength model after first few lines identified
                if (len(wpoints) > 4):
                    xps = np.argsort(xpoints)
                    spl = UnivariateSpline(xpoints[xps], wpoints[xps], ext=0, k=3, s=1e3)
                    wcent_guess = spl(pcent_pix)
        print("Mode='nearest': " + str(len(wpoints)) + ' lines matched.')


    #######
    if identify_mode.lower() == "file":
        # read a previously saved .lines file
        if len(previous_file)==0:
            raise ValueError('For identify_mode="previous", `previous_file` must give the path to the previously saved lines file.')

        tbl = Table.read(previous_file, format='ascii', names=('pix', 'wave'))
        xpoints, wpoints = tbl['pix'], tbl['wave']
        print("Mode='file': " + str(len(wpoints)) + ' lines used from '+previous_file)

    if identify_mode.lower() == 'lines':
        if (len(xpoints) != len(wpoints)) | (len(xpoints) == 0):
            raise ValueError('xpoints and wpoints must match in length, and be greater than 0 length')
        print("Mode='lines': " + str(len(wpoints)) + ' lines used.')

    #######
    # if identify_mode.lower() == 'crosscor':

    # sort the points, just in case the method (or prev run) returns in weird order
    srt = np.argsort(xpoints)
    xpoints = xpoints[srt]
    wpoints = wpoints[srt]

    # now turn the identified (xpixel, wavelength) points -> wavelength(x)
    # Require at least... 4 identified lines to generate a solution?
    if (len(xpoints) > 4):
        if (fit_mode.lower() == 'spline'):
            # assuming there is a flux value for every xpixel of interest
            # and that it starts at pixel = 0
            # apply our final wavelength spline solution to the entire array
            spl = UnivariateSpline(xpoints, wpoints, ext=0, k=3, s=1e3)
            wavesolved = spl(np.arange(np.size(flux)))

        if (fit_mode.lower() == 'poly'):
            fit = np.polyfit(xpoints, wpoints, polydeg)
            wavesolved = np.polyval(fit, np.arange(np.size(flux)))

        if (fit_mode.lower() == 'interp'):
            wavesolved = np.interp(np.arange(np.size(flux)), xpoints, wpoints)
    else:
        raise ValueError('Too few lines identified to derive a solution. len(xpoints)='+str(len(xpoints)))

    return wavesolved


def identify_widget(xpixels, flux, silent=False):
    """
    Interactive version of the Identify GUI, specifically using ipython widgets.

    Each line is roughly identified by the user, then a Gaussian is fit to
    determine the precise line center. The reference value for the line is then
    entered by the user.

    When finished, the output lines should usually be passed in a new Jupter
    notebook cell to `identify` for determining the wavelength solution:
    >>>> xpl,wav = identify_widget(wapprox, flux) # doctest: +SKIP
    >>>> new_wave = identify(wapprox, flux, identify_mode='lines', xpoints=xpl, wpoints=wav) # doctest: +SKIP

    NOTE: Because of the widgets used, this is not well suited for inclusion in
    pipelines, and instead is ideal for interactive analysis.

    Parameters
    ----------
    xpixels : `~numpy.ndarray`
        Array of the pixel values along wavelength dimension of the trace.
    flux : `~numpy.ndarray`
        Array of the flux values along the trace. Must have same size as xpixels.
    silent : bool
        Set to True to silence the instruction print out each time. (Default: False)

    Returns
    -------
    The pixel locations and wavelengths of the identified lines:
    pixel, wavelength
    """


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
        rgn = np.where((np.abs(xpixels - event.xdata ) <= 5.))[0]
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
        return #xpxl, waves

    button.on_click(onbuttonclick)

    # Do the plot
    ax.plot(xpixels, flux)
    plt.draw()

    # Display widgets
    display(widgets.HBox([xval, linename, button]))

    # return np.array(xpxl), np.array(waves)
    return xpxl, waves