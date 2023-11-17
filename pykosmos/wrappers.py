"""
Scripts that wrap around the spectral reduction functions. In the future,
could include instrument or setup sepcific reduction workflows here.
"""

import pykosmos as pk
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
import datetime

__all__ = ['script_reduce']

def script_reduce(script,
                  trim=True,
                  apwidth=10, skysep=5, skywidth=5,
                  trace_nbins=15, trace_guess=None, trace_window=None,
                  stdtrace=False,
                  obs_file='apoextinct.dat',
                  linelist='apohenear.dat',
                  waveapprox=False,
                  Saxis=0, Waxis=1,
                  display=True, display_all=False,
                  write_reduced=True,
                  debug=False, silencewarnings=False):
    """
    Wrapper function that carries out all aspects of simple spectral
    reduction, according to a script. Here the script file is analogous to
    a simple observing log, a CSV that has two columns:
        `file path, type`

    where `type` must be from the list of available filetypes:
        `(bias, flat, arc, std, object)`

    The script is processed top-to-bottom. Sequential files of the same
    type are processed together (e.g. biases are combined), and applied
    on all subsequent files. Wavelength solutions or sensitivity functions
    are defined and used on all subsequent files. If Arc lamps are taken
    after each observation, putting the arc file *before* the observation
    file ensures a new wavelength solution is applied.

    Standards must have the `onedstd` reference included as:
        std:LIBRARY/FILE.dat
    (see example script below)

    This framework allows `script_reduce` to automatically process most
    sets of simple long-slit observations -- even from multiple nights --
    with well-defined results, but does not allow for fully customized
    adjustments that may be needed for precision work.

    EXAMPLE SCRIPT FILE
        file1.fits, bias
        file2.fits, bias
        file3.fits, flat
        file4.fits, flat
        file8.fits, arc
        file6.fits, std:spec50cal/bd284211.dat
        file7.fits, object

    Parameters
    ----------
    script : str
        The .csv file to process (required)
    trim : bool, optional (default is True)
    apwidth : int, optional (default is 10)
        The width along the spatial axis on either side of the trace to
        extract. (passed to `apextract`)
    skysep : int, optional (default is 5)
        The separation in pixels from the aperture to the sky window.
        (passed to `apextract`)
    skywidth : int, optional (default is 5)
        The width in pixels of the sky windows on either side of the
        aperture. (passed to `apextract`)
    trace_nbins : int, optional (default is 15)
        number of bins in wavelength direction to chop image into.
        (passed to `trace` as `nbins`)
    trace_guess : int or None, optional (default is None)
        A guess at where the desired trace is in the spatial direction. If set,
        overrides the normal max peak finder. Good for tracing a fainter source if
        multiple traces are present. (passed to `trace` as `guess`)
    trace_window : int or None, optional (default is None)
        If set, only fit the trace within a given region around the guess position.
        Useful for tracing faint sources if multiple traces are present, but
        potentially bad if the trace is substantially bent or warped.
        (passed to `trace` as `window`)
    stdtrace : bool, optional (default is False)
        if True, use first object or standard to establish the trace, and use
        for extracting every spectrum thereafter.
    obs_file : str, optional (default is 'apoextinct.dat')
        Observatory-specific airmass extinction file
         passed to `obs_extinction`
    linelist : str, optional (default is 'apohenear.dat')
        Passed to `loadlinelist` to load arclines, which are used
        by `identify_nearest`.
    waveapprox : bool, optional (default is False)
        if set, use the approximate wavelength from the header
        'DISPDW' and 'DISPWC' keywords. Usually not great. 
        (probably needs to be genearlized better)
    Saxis : int, optional (default is 0)
        Set which axis is the spatial dimension. For DIS, Saxis=0
        (corresponds to NAXIS2 in header). For KOSMOS, Saxis=1.
    Waxis : int, optional (default is 1)
        Set which axis is the wavelength dimension. For DIS, Waxis=1
        (corresponds to NAXIS1 in the header). For KOSMOS, Waxis=0.
        NOTE: if Saxis is changed, Waxis will be updated, and visa versa.
    display : bool, optional (default is True)
        if set, plot reduced object and standard spectra to the screen
    display_all : bool, optional (default is False)
        if set, passes `display=True` to all other functions
    write_reduced : bool, optional (default is True)
        if set, write reduced object and standard spectra to a
        .fits file, and make a log file for this run.
    silencewarnings : bool, optional (default is False)
        aggresively silence warnings that get spit out, mainly
        from WCS. (caution: you might miss an important failure)

    Returns
    -------
    Output files are created for combined calibration frames
        (e.g. bias.fits, flat.fits), and fully reduced versions of objects

    """

    # Improvements Needed
    # ------------
    # - add other wavelenth solution methods (e.g. DTW based on Kosmos templates)
    # - Output reduced spectra using different file types? (text file?)
    # - enable jpeg save w/o display (faster/automatic)

    if silencewarnings is True:
        # i hate doing this, but wcs/fits gives SO many warnings for our images...
        import warnings
        from astropy.io.fits.verify import VerifyWarning
        warnings.simplefilter('ignore', category=VerifyWarning)
        from astropy.wcs import FITSFixedWarning
        warnings.simplefilter('ignore', category=FITSFixedWarning)

    # read the reduction script - this drives everything!
    tbl = Table.read(script, format='csv', names=('file', 'type'))

    # start calibration frames off w/ None...
    bias = None
    flat = None
    ilum = None
    fallbacktrace = None

    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0

    # read the reference arcline list in
    # NEED TO STUDY HOW THIS WORKS WITH KOSMOS...
    if linelist is not None:
        arclines = pk.loadlinelist(linelist)

    # write the log file to save input settings
    if write_reduced is True:
        lout = open(script+'.log', 'w')
        lout.write('# pykosmos/scriptreduce log for: ' + script +'\n')
        now = datetime.datetime.now()
        lout.write('DATE-REDUCED = ' + str(now) + '\n')
        # save state of parameters
        lout.write('trim = ' + str(trim) + '\n')
        lout.write('trace_nbins = ' + str(trace_nbins) + '\n')
        lout.write('trace_guess = ' + str(trace_guess) + '\n')
        lout.write('trace_window = ' + str(trace_window) + '\n')
        lout.write('stdtrace = ' + str(trace_window) + '\n')
        if obs_file is not None:
            lout.write('obs_file = ' + str(obs_file) + '\n')
        lout.write('linelist = ' + linelist + '\n')
        lout.write('waveapprox = ' + str(waveapprox) + '\n')
        lout.write('Saxis = ' + str(Saxis) + '\n')
        lout.write('Waxis = ' + str(Waxis) + '\n')
        lout.close() # close log file
        # can move CLOSE to the end of script, in case there's things to add later


    # airmass observatory extinction file to use
    if obs_file is not None:
        obs_file = pk.obs_extinction(obs_file)

    j = 0
    while j < len(tbl):
        # print(j, len(tbl) - 1, tbl['file'].data[j], tbl['type'].data[j])

        if (tbl['type'].data[j] == 'bias') | (tbl['type'].data[j] == 'flat'):
            if debug:
                print(tbl['file'].data[j], tbl['type'].data[j], ' bias/flat')

            # bias or flat combine only runs if n>1 files.
            # if this isn't the last row in the script, look for more rows of same type
            if j < (len(tbl) - 1):
                # find the next row that is NOT a bias or flat
                nxt = np.where((tbl['type'][j:] != tbl['type'][j]))[0]
                if len(nxt) > 0:
                    nxtfile = nxt[0]
                # if all remaining rows are the same type, grab all the rest
                if len(nxt) == 0:
                    nxtfile = len(tbl)-j

                # handle biases vs flats
                if (tbl['type'][j] == 'bias'):
                    print('combining ' + str(len(tbl['file'].data[j:(j + nxtfile)])) + ' biases')
                    bias = pk.biascombine(tbl['file'].data[j:(j + nxtfile)])

                if (tbl['type'][j] == 'flat'):
                    print('combining ' + str(len(tbl['file'].data[j:(j + nxtfile)])) + ' flats')
                    flat, ilum = pk.flatcombine(tbl['file'].data[j:(j + nxtfile)], bias=bias,
                                                    trim=trim, Waxis=Waxis, Saxis=Saxis)

                # set the counter to jump the run of biases
                j = j + nxtfile - 1

        if tbl['type'].data[j] == 'arc':
            if debug:
                print(tbl['file'].data[j], ' arc')
            # there's going to be some tedious logic here if things arent
            # defined "in order", or are missing for an object reduction...
            # => start w/ the "right" order first, build fall-backs second

            arcimg = pk.proc(tbl['file'].data[j], bias=bias, ilum=ilum,
                                 trim=trim, Waxis=Waxis, Saxis=Saxis)
            # if trace is defined use that, otherwise use flat cut

        # for standard OR object files
        if (tbl['type'].data[j][0:3] == 'std') | (tbl['type'].data[j] == 'object'):
            # reduce image
            if debug:
                print(tbl['file'].data[j], ' std/object')

            img = pk.proc(tbl['file'].data[j], bias=bias, ilum=ilum, flat=flat,
                              trim=trim, Waxis=Waxis, Saxis=Saxis)

            # trace/extract data
            if (fallbacktrace is not None) and (stdtrace is True) and (tbl['type'].data[j] == 'object'):
                if debug:
                    print('using previous std trace')
                trace = fallbacktrace
            else:
                trace = pk.trace(img, display=display_all, nbins=trace_nbins,
                                     guess=trace_guess, window=trace_window,
                                     Waxis=Waxis, Saxis=Saxis)

            sci_ex, sci_sky = pk.BoxcarExtract(img, trace, apwidth=apwidth, skysep=skysep, skywidth=skywidth, Waxis=Waxis, Saxis=Saxis)
            spectrum = sci_ex.subtract(sci_sky, compare_wcs=None)

            # EVERY frame is: traced, extracted from both the data & arc, and identified

            # wavelength solution
            # extract trace across the Arc lamp image
            sciarc_ex, _ = pk.BoxcarExtract(arcimg, trace, apwidth=apwidth, skysep=skysep, skywidth=skywidth, Waxis=Waxis, Saxis=Saxis)

            wapprox = (np.arange(img.shape[1]) - img.shape[1] / 2)[::-1] * img.header['DISPDW'] + img.header['DISPWC']
            wapprox = wapprox * u.angstrom

            # just use the header-based approximate wavelength? (usually not great)
            if waveapprox:
                sci_fit = pk.fit_wavelength(spectrum, np.arange((len(wapprox))), wapprox,
                                                mode='poly', deg=1)
            else:
                sci_xpts, sci_wpts = pk.identify_nearest(sciarc_ex, wapprox=wapprox,
                                                             linewave=arclines, autotol=5)
                sci_fit = pk.fit_wavelength(spectrum, sci_xpts, sci_wpts, mode='interp', deg=3)

            # apply airmass calibration
            if obs_file is not None:
                ZD = img.header['ZD'] / 180.0 * np.pi
                sci_airmass = 1.0 / np.cos(ZD)  # approximate Zenith Distance -> Airmass conversion
                sci_fit = pk.airmass_cor(sci_fit, sci_airmass, obs_file)

            if tbl['type'].data[j][0:3] == 'std':
                # import standard star
                if debug:
                    print(tbl['file'].data[j], ' std ', tbl['type'].data[j][4:])
                if stdtrace is True:
                    if debug:
                        print('saving std trace')
                    fallbacktrace = trace
                standardstar = pk.onedstd(tbl['type'].data[j][4:])
                # generate new sensitivity function
                sensfunc = pk.standard_sensfunc(sci_fit, standardstar,
                                                    mode='linear', display=display_all)

            # if tbl['type'].data[j] == 'object':
            #     if debug:
            #         print(tbl['file'].data[j], ' object')

            # flux calibration, apply sensfunc (to either an Obj or Std)
            final_spectrum = pk.apply_sensfunc(sci_fit, sensfunc)

            if display is True:
                plt.figure()
                plt.plot(final_spectrum.wavelength, final_spectrum.flux)
                plt.xlabel('Wavelength [' + str(final_spectrum.wavelength.unit) + ']')
                plt.ylabel('Flux [' + str(final_spectrum.flux.unit) + ']')

                # save a jpeg version... b/c its nice for quick reference!
                if write_reduced:
                    plt.savefig(tbl['file'].data[j].split('.fits')[0]+'_reduced.jpeg',
                                dpi=150, bbox_inches='tight', pad_inches=0.25)
                plt.show()

            # save 1D spectrum as some kind of ascii table
            if write_reduced:
                red_file = tbl['file'].data[j].split('.fits')[0]+'_reduced.fits'
                final_spectrum.write(red_file, format='tabular-fits')

        j += 1

    return

