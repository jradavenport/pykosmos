"""
This file contains tools to help with image manipulation, such as wrappers around
astropy tools for bias combining.
"""

from astropy.nddata import CCDData
from ccdproc import Combiner, trim_image
from astropy import units as u
from ccdproc import cosmicray_lacosmic

__all__ = ['biascombine', 'proc']


def biascombine(bfiles):
    """
    A simple wrapper to go through bias frames, read them in, and combine them.

    Currently median combine is hard-coded, but ccdproc.Combiner does have functions
    for other methods. See:
    https://ccdproc.readthedocs.io/en/latest/api/ccdproc.Combiner.html

    Parameters
    ----------
    bfiles : list of paths to bias frame .fits files

    Returns
    -------
    bias : CCDData object
    """

    blist = []
    # loop over all bias frames
    for k in range(len(bfiles)):
        # read them in as CCD Data objects
        img = CCDData.read(bfiles[k], unit=u.adu)
        # append them to a big image list
        blist.append(img)

    bias = Combiner(blist).median_combine()
    return bias


def proc(file, bias=None, flat=None, dark=None,
         trim=True, ilum=None, Saxis=0, Waxis=1,
         EXPTIME='EXPTIME', DATASEC='DATASEC',
         CR=False, GAIN='GAIN', READNOISE='RDNOISE', CRsigclip=4.5):
    """
    Semi-generalized function to read a FITS file in, divide by exposure
    time (returns units of ADU/s), and optionally perform basic CCD
    processing to it (bias, dark, flat corrections, biassec and
    illumination region trimming).

    Parameters
    ----------
    file : string
        path to FITS file
    bias : CCDData object, optional (default=None)
        median bias frame generated using e.g. `biascombine` to subtract
        from each flat image
    dark : CCDData object, optional
        dark frame to subtract
    flat : CCDData object, optional
        combined flat frame to divide
    trim : bool (default=True)
        Trim the "bias section" out of each flat frame. Uses fits header
        field defined by `DATASEC` keyword
    ilum : array, optional
        if provided, trim image to the illuminated portion of the CCD.
    EXPTIME : string (optional, default='EXPTIME')
        FITS header field containing the exposure time in seconds.
    DATASEC : string (optional, default='DATASEC')
        FITS header field containing the data section of the CCD, i.e. to
        remove the bias section. Used if `trim=True`
    Saxis : int, optional
        Set which axis is the spatial dimension. For DIS, Saxis=0
        (corresponds to NAXIS2 in header). For KOSMOS, Saxis=1.
        (Default is 0)
    Waxis : int, optional
        Set which axis is the wavelength dimension. For DIS, Waxis=1
        (corresponds to NAXIS1 in the header). For KOSMOS, Waxis=0.
        (Default is 1)
        NOTE: if Saxis is changed, Waxis will be updated, and visa versa.
    CR : bool (default=False)
        If True, use the L.A. Cosmic routine to remove cosmic rays from
        image before reducing.
    GAIN : string (optional, default='GAIN')
        FITS header field containing the Gain parameter, used by
        L.A. Cosmic
    READNOISE : string (optional, default='RDNOISE')
        FITS header field containing the Read Noise parameter, used by
        L.A. Cosmic
    CRsigclip : int (optional, default=4.5)
        sigma-clipping parameter passed to L.A. Cosmic
    gain_apply : bool (optional, default=False)
        apply gain to image. If this is set to true and the bias or flat are not 
        in units of electrons per adu this may result in an error

    Returns
    -------
    img : CCDData object
    """

    img = CCDData.read(file, unit=u.adu)

    # should we remove cosmic rays?
    if CR:
        # IMPROVEMENT IDEA: could we pass more parameters as kwargs?
        # also, specify either header fields OR actual values?
        img = cosmicray_lacosmic(img, gain=img.header[GAIN]*u.electron/u.adu,
                                 readnoise=img.header[READNOISE] * u.electron,
                                 sigclip=CRsigclip, gain_apply=False)

    # subtract the bias, divide by exposure time, update units to ADU/s
    if bias is None:
        pass
    else:
        img.data = img.data - bias

    if dark is None:
        pass
    else:
        img.data = img.data - dark

    # trim off bias section
    if trim:
        img = trim_image(img, fits_section=img.header[DATASEC])

    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0

    # trim to illuminated region of CCD
    if ilum is None:
        pass
    else:
        # img.data = img.data[ilum, :]
        # img = trim_image(img[ilum, :])
        # use continuous region, not array, to play nice w/ WCS slice
        # img = trim_image(img[ilum[0]:(ilum[-1]+1), :])
        if Waxis == 1:
            img = trim_image(img[ilum[0]:(ilum[-1]+1), :])
        if Waxis == 0:
            img = trim_image(img[:, ilum[0]:(ilum[-1] + 1)])

    # divide out the flat
    if flat is None:
        pass
    else:
        img.data = img.data / flat
        # img = img.divide(flat) # don't do this b/c it breaks header

    # ISSUE: what if this keyword doesn't exist... need try/except?
    img.data = img.data / img.header[EXPTIME]
    img.unit = img.unit / u.s

    return img
