"""
Scripts that wrap around the spectral reduction functions
"""

import kosmos
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u


# __all__ = ['autoreduce', 'CoAddFinal', 'ReduceCoAdd', 'ReduceTwo']

def script_reduce(script,
                  trim=True,
                  apwidth=10, skysep=5, skywidth=5,
                  trace_nbins=15, trace_guess=None, trace_window=None,
                  stdtrace=False,
                  Xfile='apoextinct.dat',
                  linelist='apohenear.dat',
                  waveapprox=False,
                  display=False, debug=False, silencewarnings=False):
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
    -------------------
        file1.fits, bias
        file2.fits, bias
        file3.fits, flat
        file4.fits, flat
        file8.fits, arc
        file6.fits, std:spec50cal/bd284211.dat
        file7.fits, object

    Returns
    -------
    Output files are created for combined calibration frames
    (e.g. bias.fits, flat.fits), and fully reduced versions of objects

    Add logs/intermediate files?

    """


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
    fallbacktrace=None

    # read the reference arcline list in
    if linelist is not None:
        arclines = kosmos.loadlinelist(linelist)

    # airmass extinction file to use
    if Xfile is not None:
        Xfile = kosmos.obs_extinction(Xfile)

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
                    bias = kosmos.biascombine(tbl['file'].data[j:(j + nxtfile)])

                if (tbl['type'][j] == 'flat'):
                    print('combining ' + str(len(tbl['file'].data[j:(j + nxtfile)])) + ' flats')
                    flat, ilum = kosmos.flatcombine(tbl['file'].data[j:(j + nxtfile)], bias=bias, trim=trim)

                # set the counter to jump the run of biases
                j = j + nxtfile - 1

        if tbl['type'].data[j] == 'arc':
            if debug:
                print(tbl['file'].data[j], ' arc')
            # there's going to be some tedious logic here if things arent
            # defined "in order", or are missing for an object reduction...
            # => start w/ the "right" order first, build fall-backs second

            arcimg = kosmos.proc(tbl['file'].data[j], bias=bias, ilum=ilum, trim=trim)
            # if trace is defined use that, otherwise use flat cut

        # for standard OR object files
        if (tbl['type'].data[j][0:3] == 'std') | (tbl['type'].data[j] == 'object'):
            # reduce image
            if debug:
                print(tbl['file'].data[j], ' std/object')

            img = kosmos.proc(tbl['file'].data[j], bias=bias, ilum=ilum, flat=flat, trim=trim)

            # trace/extract data
            if (fallbacktrace is not None) and (stdtrace is True) and (tbl['type'].data[j] == 'object'):
                if debug:
                    print('using previous std trace')
                trace = fallbacktrace
            else:
                trace = kosmos.trace(img, display=display, nbins=trace_nbins,
                                     guess=trace_guess, window=trace_window)

            sci_ex, sci_sky = kosmos.BoxcarExtract(img, trace, apwidth=apwidth,
                                                   skysep=skysep, skywidth=skywidth)
            spectrum = sci_ex - sci_sky

            # EVERY frame is: traced, extracted from both the data & arc, and identified

            # wavelength solution
            # extract trace across the Arc lamp image
            sciarc_ex, _ = kosmos.BoxcarExtract(arcimg, trace, apwidth=apwidth,
                                                skysep=skysep, skywidth=skywidth)

            wapprox = (np.arange(img.shape[1]) - img.shape[1] / 2)[::-1] * img.header['DISPDW'] + img.header['DISPWC']
            wapprox = wapprox * u.angstrom

            # just use the header-based approximate wavelength? (usually not great)
            if waveapprox:
                sci_fit = kosmos.fit_wavelength(spectrum, np.arange((len(wapprox))), wapprox,
                                                mode='poly', deg=1)
            else:
                sci_xpts, sci_wpts = kosmos.identify_nearest(sciarc_ex, wapprox=wapprox,
                                                             linewave=arclines, autotol=5)
                sci_fit = kosmos.fit_wavelength(spectrum, sci_xpts, sci_wpts, mode='interp', deg=3)

            # apply airmass calibration
            if Xfile is not None:
                ZD = img.header['ZD'] / 180.0 * np.pi
                sci_airmass = 1.0 / np.cos(ZD)  # approximate Zenith Distance -> Airmass conversion
                sci_fit = kosmos.airmass_cor(sci_fit, sci_airmass, Xfile)

            if tbl['type'].data[j][0:3] == 'std':
                # import standard star
                if debug:
                    print(tbl['file'].data[j], ' std ', tbl['type'].data[j][4:])
                if stdtrace is True:
                    if debug:
                        print('saving std trace')
                    fallbacktrace = trace
                standardstar = kosmos.onedstd(tbl['type'].data[j][4:])
                # generate new sensitivity function
                sensfunc = kosmos.standard_sensfunc(sci_fit, standardstar,
                                                    mode='linear', display=display)

            if tbl['type'].data[j] == 'object':
                if debug:
                    print(tbl['file'].data[j], ' object')

                # flux calibration, apply sensfunc
                final_spectrum = kosmos.apply_sensfunc(sci_fit, sensfunc)

                plt.figure(figsize=(9, 4))
                plt.plot(final_spectrum.wavelength, final_spectrum.flux, c='k', label='New reduction')
                plt.xlabel('Wavelength [' + str(final_spectrum.wavelength.unit) + ']')
                plt.ylabel('Flux [' + str(final_spectrum.flux.unit) + ']')

                # write log file
                # save 1D spectrum as some kind of ascii table

        j += 1

    return
