import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel
from ccdproc import Combiner, trim_image
from .imtools import proc
from astropy.nddata import CCDData

__all__ = ['find_illum', 'flat_response', 'flatcombine']


def find_illum(flat, threshold=0.9, Saxis=0, Waxis=1):
    """
    Use threshold to define the illuminated portion of the image.

    Parameters
    ----------
    flat : CCDData object
        An image, typically the median-combined flat
    threshold : float
        the fraction to clip to determine the illuminated portion (between 0 and 1)
    Saxis : int, optional
        Set which axis is the spatial dimension. For DIS, Saxis=0
        (corresponds to NAXIS2 in header). For KOSMOS, Saxis=1.
        (Default is 0)
    Waxis : int, optional
        Set which axis is the wavelength dimension. For DIS, Waxis=1
        (corresponds to NAXIS1 in the header). For KOSMOS, Waxis=0.
        (Default is 1)
        NOTE: if Saxis is changed, Waxis will be updated, and visa versa.

    Returns
    -------
    ilum : numpy array
        the indicies along the spatial dimension that are illuminated
    """
    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0

    # compress all wavelength for max S/N
    ycompress = np.nansum(flat, axis=Waxis)

    # find rows (along spatial axis) with illumination above threshold
    ilum = np.where(((ycompress / np.nanmedian(ycompress)) >= threshold))[0]
    return ilum


def flat_response(medflat, smooth=False, npix=11, display=False,
                  Saxis=0, Waxis=1, ax=None):
    """
    Divide out the spatially-averaged spectrum response from the flat image.
    This is to remove the spectral response of the flatfield (e.g. Quartz) lamp.

    Input flat is first averaged along the spatial dimension to make a 1-D flat.
    This is optionally smoothed, and then the 1-D flat is divided out of each row
    of the image.

    Note: implicitly assumes spatial and spectral axes are orthogonal, i.e. does not
    trace lines of constant wavelength for normalization.

    Parameters
    ----------
    medflat : CCDData object
        An image, typically the median-combined flat
    smooth : bool (default=False)
        Should the 1-D, mean-combined flat be smoothed before dividing out?
    npix : int (default=11)
        if `smooth=True`, how big of a boxcar smooth kernel should be used (in pixels)?
    display : bool (default=False)
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
    Returns
    -------
    flat : CCDData object

    """
    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0

    # average the data together along the "spatial" axis
    flat_1d = np.nanmean(medflat, axis=Saxis)

    # optionally: add boxcar smoothing to the 1-D average
    if smooth:
        flat_1d = convolve(flat_1d, Box1DKernel(npix), boundary='extend')

    # ADD? this averaged curve could be modeled w/ spline, polynomial, etc

    # divide the spectral response from the flat lamp (e.g. quartz lamp)
    ## the old way w/ numpy
    # flat = np.zeros_like(medflat)
    # for i in range(medflat.shape[Saxis]):
    #     flat[i, :] = medflat[i, :] / flat_1d

    ## the new way w/ CCDdata objects... i hope!
    # flat = medflat.divide(flat_1d)
    # NOPE, b/c CCDData divide doesn't like dividing arrays across the X-axis (works for Y-axis)

    flat = np.zeros_like(medflat)
    for i in range(flat.shape[Saxis]):
        if Saxis == 0:
            flat[i, :] = medflat[i, :].divide(flat_1d).data
        if Saxis == 1:
            flat[:, i] = medflat[:, i].divide(flat_1d).data
    flat = CCDData(flat, unit=medflat.unit)

    # once again normalize, since (e.g. if haven't trimmed illumination region)
    # averaging could be skewed by including some non-illuminated portion.
    flat = flat.divide(np.nanmedian(flat.data))

    # the resulting flat should just show the pixel-to-pixel variations we're after
    if display:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        im = ax.imshow(flat, origin='lower', aspect='auto', cmap=plt.cm.inferno)
        plt.colorbar(mappable=im)
        ax.set_title('flat')
        plt.show()

    return flat


def flatcombine(ffiles, bias=None, trim=True, normframe=True,
                illumcor=True, threshold=0.9,
                responsecor=True, smooth=False, npix=11,
                Saxis=0, Waxis=1,
                EXPTIME='EXPTIME', DATASEC='DATASEC'  # header keywords
                ):
    """
    A general-purpose wrapper function to create a science-ready
    flatfield image.


    Parameters
    ----------
    ffiles : list of paths to the flat frame .fits files
    bias : CCDData object, optional (default=None)
        median bias frame generated using e.g. `biascombine` to subtract
        from each flat image
    trim : bool (default=True)
        Trim the "bias section" out of each flat frame. Uses fits header
        field defined by `DATASEC` keyword
    normframe : bool (default=True)
        if set normalize each bias frame by its median value before combining
    illumcor : bool (default=True)
        use the median-combined flat to determine the illuminated portion
        of the CCD. Runs `find_illum`.
    threshold : float (optional, default=0.9)
        Passed to `find_illum`.
        the fraction to clip to determine the illuminated portion (between 0 and 1)
    responsecor : bool (default=True)
        Divide out the spatially-averaged spectrum response from the flat image.
        Runs `flat_response`
    smooth : bool (default=False)
        Passed to `flat_response`.
        Should the 1-D, mean-combined flat be smoothed before dividing out?
    npix : int (default=11)
        Passed to `flat_response`.
        if `smooth=True`, how big of a boxcar smooth kernel should be used (in pixels)?
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

    Returns
    -------
    flat : CCDData object
        Always returned, the final flat image object
    ilum : array
        Returned if `illumcor=True`, the 1-D array to use for trimming
        science images to the illuminated portion of the CCD.

    """

    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0

    flist = []
    # loop over all flat frames
    for k in range(len(ffiles)):
        img = proc(ffiles[k], bias=bias, EXPTIME=EXPTIME, DATASEC=DATASEC, trim=trim)

        # normalize each flat frame by its median
        # ISSUE: this might not be needed, since each frame is in flux units?
        if normframe:
            img.data = img.data / np.nanmedian(img.data)

        flist.append(img)

    # combine the flat frames w/ a median
    medflat = Combiner(flist).median_combine()

    # should we use the median flat to detect the illuminated portion of the CCD?
    if illumcor:
        ilum = find_illum(medflat, threshold=threshold, Waxis=Waxis)

        # if so, trim the median flat to only the illuminated portion
        # ISSUE: this hard-codes the wavelength vs spatial axes!
        # medflat = medflat[ilum, :]
        # use trim_image to make a proper copy
        # use continuous region, not array, to play nice w/ WCS slice
        if Waxis == 1:
            medflat = trim_image(medflat[ilum[0]:(ilum[-1] + 1), :])
        if Waxis == 0:
            medflat = trim_image(medflat[:, ilum[0]:(ilum[-1] + 1)])

    if responsecor:
        medflat = flat_response(medflat, smooth=smooth, npix=npix, Saxis=Saxis)

    if illumcor:
        return medflat, ilum
    else:
        return medflat
