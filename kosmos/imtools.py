"""
This file contains tools to help with image manipulation, such as wrappers around
astropy tools for bias combining.
"""

from astropy.nddata import CCDData
from ccdproc import Combiner, trim_image
from astropy import units as u

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
         trim=True, ilum=None,
         EXPTIME='EXPTIME', DATASEC='DATASEC'):
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

    Returns
    -------
    img : CCDData object
    """

    img = CCDData.read(file, unit=u.adu)

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

    # trim to illuminated region of CCD
    if ilum is None:
        pass
    else:
        # img.data = img.data[ilum, :]
        img = trim_image(img[ilum, :])

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
