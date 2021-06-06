"""
NOT EDITED FOR KOSMOS YET
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import UnivariateSpline
from astropy.constants import c as cc
import astropy.units as u
import os


__all__ = ['airmass_cor', 'obs_extinction', 'standard_sensfunc', 'apply_sensfunc', 'mag2flux']


def mag2flux(wave, mag, zeropt=48.60):
    '''
    Convert magnitudes to flux units. This is important for dealing with standards
    and files from IRAF, which are stored in AB mag units. To be clear, this converts
    to "PHOTFLAM" units in IRAF-speak. Assumes the common flux zeropoint used in IRAF

    Parameters
    ----------
    wave : 1d numpy array
        The wavelength of the data points in Angstroms
    mag : 1d numpy array
        The magnitudes of the data
    zeropt : float, optional
        Conversion factor for mag->flux. (Default is 48.60)

    Returns
    -------
    Flux values

    Improvements Needed
    -------------------
    1) make input units awareness work (angstroms)
        in fact, can this be totally done within astropy units?
        https://docs.astropy.org/en/stable/units/logarithmic_units.html
    2) use Spectrum1D object?
    3) is this a function that should be moved to a different package? (specutils)
    '''

    # c = 2.99792458e18 # speed of light, in A/s
    flux = (10.0**( (mag + zeropt) / (-2.5))) * (cc.to('AA/s').value / wave ** 2.0)
    return flux * u.erg / u.s / u.angstrom / (u.cm * u.cm)


def obs_extinction(obs_file):
    '''
    Load the observatory-specific airmass extinction file from the supplied library
    in the directory kosmos/resources/extinction

    Parameters
    ----------
    obs_file : str, {'apoextinct.dat', 'ctioextinct.dat', 'kpnoextinct.dat', 'ormextinct.dat'}
        The observatory-specific airmass extinction file. If not known for your
        observatory, use one of the provided files (e.g. `kpnoextinct.dat`).

        Following IRAF standard, extinction files have 2-column format
        wavelength (Angstroms), Extinction (Mag per Airmass)

    Returns
    -------
    Astropy Table with the observatory extinction data
    '''
    if len(obs_file) == 0:
        raise ValueError('Must select an observatory extinction file.')

    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'resources', 'extinction')

    if not os.path.isfile(os.path.join(dir, obs_file)):
        msg2 = "No valid standard star found at: " + os.path.join(dir, obs_file)
        raise ValueError(msg2)
    # read in the airmass extinction curve
    Xfile = Table.read(os.path.join(dir, obs_file), format='ascii', names=('wave', 'X'))
    Xfile['wave'].unit = 'AA'

    return Xfile


def airmass_cor(object_spectrum, airmass, Xfile):
    """
    Correct the spectrum based on the airmass. Requires observatory extinction file

    Parameters
    ----------
    object_spectrum : Spectrum1D object
    airmass : float
        The value of the airmass. Note: NOT the header keyword.
    Xfile : astropy table
        The extinction table from `obs_extinction`, with columns ('wave', 'X')
        that have standard units of: (angstroms, mag/airmass)

    Returns
    -------
    The airmass corrected Spectrum1D object

    """

    obj_wave, obj_flux = object_spectrum.wavelength, object_spectrum.flux

    # linear interpol airmass extinction onto observed wavelengths
    new_X = np.interp(obj_wave.value, Xfile['wave'], Xfile['X'])

    # air_cor in units of mag/airmass, convert to flux/airmass
    airmass_ext = 10.0**(0.4 * airmass * new_X)

    # output_spec = object_spectrum.multiply(object_spectrum.flux, airmass_ext)
    # output_spec.flux = obj_flux * airmass_ext
    # return output_spec
    return object_spectrum.multiply(airmass_ext * u.dimensionless_unscaled)


def onedstd(stdstar):
    '''
    Load the onedstd from the supplied library

    Parameters
    ----------
    stdstar : str
        Name of the standard star file in the kosmos/resources/onedstds
        directory to be used for the flux calibration. The user must provide the
        subdirectory and file name. For example:

        >>> standard_sensfunc(obj_wave, obj_flux, stdstar='spec50cal/bd284211.dat', mode='spline')  \
        # doctest: +SKIP

        If no std is supplied, or an improper path is given, will raise a ValueError.

    Returns
    -------
        astropy Table with onedstd data
    '''
    std_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'resources', 'onedstds')

    if not os.path.isfile(os.path.join(std_dir, stdstar)):
        msg2 = "No valid standard star found at: " + os.path.join(std_dir, stdstar)
        raise ValueError(msg2)

    std = Table.read(os.path.join(std_dir, stdstar),
                     format='ascii', names=('wave', 'mag', 'width'))
    std['wave'].unit = u.angstrom
    std['width'].unit = u.angstrom
    # standard star spectrum is stored in magnitude units (IRAF conventions)
    std_flux = mag2flux(std['wave'], std['mag'])

    std['mag'].unit = u.mag
    std.add_column(std_flux, name='flux')

    return std


def standard_sensfunc(object_spectrum, std, mode='linear', polydeg=9,
                      badlines=[6563, 4861, 4341], display=False):
    """
    Compute the standard star sensitivity function.

    Parameters
    ----------
    object_spectrum : Spectrum1D object
        The observed standard star spectrum
    std : astropy table
        output from `onedstd`, has columns ('wave', 'width', 'mag', 'flux')
    mode : str, optional
        either "linear", "spline", or "poly" (Default is linear)
    polydeg : float, optional
        if mode='poly', this is the order of the polynomial to fit through
        (Default is 9)
    display : bool, optional
        If True, plot the sensfunc (Default is False)
    badlines : array-like
        A list of values (lines) to mask out of when generating sensfunc

    Returns
    -------
    sensfunc : astropy table
        The wavelength and sensitivity function for the given standard star

    Improvements Needed
    -------------------
    1) change "badlines" to use SpectralRegion
        https://specutils.readthedocs.io/en/stable/spectral_regions.html
    """

    obj_wave, obj_flux = object_spectrum.wavelength, object_spectrum.flux

    # Automatically exclude some lines b/c resolution dependent response
    badlines = np.array(badlines, dtype='float') # Balmer lines

    # down-sample (ds) the observed flux to the standard's bins
    obj_flux_ds = np.array([], dtype=np.float)
    obj_wave_ds = np.array([], dtype=np.float)
    std_flux_ds = np.array([], dtype=np.float)
    for i in range(len(std['flux'])):
        rng = np.where((obj_wave.value >= std['wave'][i] - std['width'][i] / 2.0) &
                       (obj_wave.value < std['wave'][i] + std['width'][i] / 2.0))[0]

        IsH = np.where((badlines >= std['wave'][i] - std['width'][i] / 2.0) &
                       (badlines < std['wave'][i] + std['width'][i] / 2.0))[0]

        # does this bin contain observed spectra, and no Balmer lines?
        if (len(rng) > 1) and (len(IsH) == 0):
            obj_flux_ds = np.append(obj_flux_ds, np.nanmean(obj_flux.value[rng]))
            obj_wave_ds = np.append(obj_wave_ds, std['wave'][i])
            std_flux_ds = np.append(std_flux_ds, std['flux'][i])

    # the ratio between the standard star catalog flux and observed flux
    ratio = np.abs(std_flux_ds / obj_flux_ds)

    # actually fit the log of this sensfunc ratio
    # since IRAF does the 2.5*log(ratio), everything in mag units!
    LogSensfunc = np.log10(ratio)

    # if invalid interpolation mode selected, make it spline
    if mode.lower() not in ('linear', 'spline', 'poly'):
        mode = 'spline'
        import warnings
        warnings.warn("WARNING: invalid mode set. Changing to default mode 'spline'")

    # interpolate the calibration (sensfunc) on to observed wavelength grid
    if mode.lower()=='linear':
        sensfunc2 = np.interp(obj_wave.value, obj_wave_ds, LogSensfunc)
    elif mode.lower()=='spline':
        spl = UnivariateSpline(obj_wave_ds, LogSensfunc, ext=0, k=2 ,s=0.0025)
        sensfunc2 = spl(obj_wave.value)
    elif mode.lower()=='poly':
        fit = np.polyfit(obj_wave_ds, LogSensfunc, polydeg)
        sensfunc2 = np.polyval(fit, obj_wave.value)

    sensfunc_out = (10 ** sensfunc2) * std['flux'].unit / obj_flux.unit

    if display is True:
        plt.figure()
        plt.plot(obj_wave, obj_flux * sensfunc_out, c='C0',
                    label='Observed x sensfunc', alpha=0.5)
        # plt.scatter(std['wave'], std_flux, color='C1', alpha=0.75, label=stdstar)
        plt.scatter(obj_wave_ds, std_flux_ds, color='C1', alpha=0.75)#, label=stdstar)

        plt.xlabel('Wavelength')
        plt.ylabel('Flux')

        plt.xlim(np.nanmin(obj_wave.value), np.nanmax(obj_wave.value))
        plt.ylim(np.nanmin(obj_flux.value * sensfunc_out.value)*0.98,
                 np.nanmax(obj_flux.value * sensfunc_out.value) * 1.02)
        # plt.legend()
        plt.show()

    tbl_out = Table()
    tbl_out.add_columns([obj_wave, sensfunc_out], names=['wave', 'S'])
    return tbl_out


def apply_sensfunc(object_spectrum, sensfunc):
    '''
    Apply the derived sensitivity function, converts observed units (e.g. ADU/s)
    to physical units (e.g. erg/s/cm2/A).

    Sensitivity function is first linearly interpolated onto the wavelength scale
    of the observed data, and then directly multiplied.

    Parameters
    ----------
    object_spectrum : Spectrum1D object
        the observed object spectrum to apply the sensfunc to
    sensfunc : astropy table
        the output of `standard_sensfunc`, table has columns ('wave', 'S')
    Returns
    -------
    The sensfunc corrected Spectrum1D object
    '''

    obj_wave, obj_flux = object_spectrum.wavelength, object_spectrum.flux

    # sort, in case the sensfunc wavelength axis is backwards
    ss = np.argsort(obj_wave.value)
    # interpolate the sensfunc onto the observed wavelength axis
    sensfunc2 = np.interp(obj_wave.value, sensfunc['wave'][ss], sensfunc['S'][ss])

    return object_spectrum * (sensfunc2 * sensfunc['S'].unit)
