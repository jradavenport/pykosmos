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
                  trim=True, Xfile = 'apoextinct.dat',
                  display=False):
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

    tbl = Table.read(script, format='csv', names=('file', 'type'))

    # start things off w/ None, before stuff may be read...
    bias = None
    flat = None
    ilum = None

    # The old IRAF-style linelists just have 2 col: (wavelength, line name)
    henear_tbl = Table.read('../kosmos/resources/linelists/apohenear.dat',
                            names=('wave', 'name'), format='ascii')
    # IMPROVEMENT NEEDED: need to make `identify_nearest` point to these linelists itself?
    henear_tbl['wave'].unit = u.angstrom
    apo_henear = henear_tbl['wave']

    # airmass extinction file to use
    Xfile = kosmos.obs_extinction(Xfile)

    j = 0
    while j < len(tbl):
        # print(j, len(tbl) - 1, tbl['file'].data[j], tbl['type'].data[j])

        if (tbl['type'].data[j] == 'bias') | (tbl['type'].data[j] == 'flat'):
            # bias or flat combine only runs if n>1 files.
            # if this isn't the last row in the script, look for more rows of same type
            if j < (len(tbl) - 1):
                # find the next row that is NOT a bias or flat
                nxt = np.where((tbl['type'][j:] != tbl['type'][j]))[0][0]

                # how many consecutive bias files are there?
                print(len(tbl['file'].data[j:(j + nxt)]))

                # handle biases vs flats
                if (tbl['type'][j] == 'bias'):
                    bias = kosmos.biascombine(tbl['file'].data[j:(j + nxt)])

                if (tbl['type'][j] == 'flat'):
                    flat, ilum = kosmos.flatcombine(tbl['file'].data[j:(j + nxt)], bias=bias)

                # set the counter to jump the run of biases
                j = j + nxt - 1

        if tbl['type'].data[j] == 'arc':
            print('loading arc image')
            # there's going to be some tedious logic here if things arent
            # defined "in order", or are missing for an object reduction...
            # => start w/ the "right" order first, build fall-backs second

            arcimg = kosmos.proc(tbl['file'].data[j], bias=bias, ilum=ilum, trim=True)
            # if trace is defined use that, otherwise use flat cut

        # for standard OR object files
        if (tbl['type'].data[j][0:3] == 'std') | (tbl['type'].data[j] == 'object'):
            # reduce image
            print('loading file')

            img = kosmos.proc(tbl['file'].data[j], bias=bias, ilum=ilum, flat=flat, trim=trim)
            # trace/extract data
            trace = kosmos.trace(img, display=False, nbins=55)
            sci_ex, sci_sky = kosmos.BoxcarExtract(img, trace, display=False,
                                                   apwidth=10, skysep=5, skywidth=5)
            spectrum = sci_ex - sci_sky

            # EVERY frame is: traced, extracted from both the data & arc, and identified

            # wavelength solution
            # extract trace across the Arc lamp image
            sciarc_ex, _ = kosmos.BoxcarExtract(arcimg, trace, apwidth=3, skysep=5, skywidth=5)

            wapprox = (np.arange(img.shape[1]) - img.shape[1] / 2)[::-1] * img.header['DISPDW'] + img.header['DISPWC']
            wapprox = wapprox * u.angstrom

            sci_xpts, sci_wpts = kosmos.identify_nearest(sciarc_ex, wapprox=wapprox, linewave=apo_henear, autotol=5)
            sci_fit = kosmos.fit_wavelength(spectrum, sci_xpts, sci_wpts, mode='interp', deg=3)

            # apply airmass calibration
            # Select the observatory-specific airmass extinction profile from the provided "extinction" library
            ZD = img.header['ZD'] / 180.0 * np.pi
            sci_airmass = 1.0 / np.cos(ZD)  # approximate Zenith Distance -> Airmass conversion

            sci_fitX = kosmos.airmass_cor(sci_fit, sci_airmass, Xfile)

        if tbl['type'].data[j][0:3] == 'std':
            # import standard star
            print('std')
            standardstar = kosmos.onedstd(tbl['type'].data[j][4:])
            # generate new sensitivity function
            sensfunc = kosmos.standard_sensfunc(sci_fitX, standardstar, mode='linear', display=False)

        if tbl['type'].data[j] == 'object':
            print('object')
            # flux calibration, apply sensfunc
            final_spectrum = kosmos.apply_sensfunc(sci_fitX, sensfunc)

            plt.figure(figsize=(9, 4))
            plt.plot(final_spectrum.wavelength, final_spectrum.flux, c='k', label='New reduction')
            plt.xlabel('Wavelength [' + str(final_spectrum.wavelength.unit) + ']')
            plt.ylabel('Flux [' + str(final_spectrum.flux.unit) + ']')

        j += 1

    return

