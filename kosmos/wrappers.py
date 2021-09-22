"""
Scripts that wrap around the spectral reduction functions
"""

import kosmos
from astropy.table import Table

# __all__ = ['autoreduce', 'CoAddFinal', 'ReduceCoAdd', 'ReduceTwo']

def script_reduce(script,
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
        file5.fits, arc
        file6.fits, std:spec50cal/bd284211.dat
        file7.fits, object

    Returns
    -------
    Output files are created for combined calibration frames
    (e.g. bias.fits, flat.fits), and fully reduced versions of objects

    Add logs/intermediate files?

    """
    tbl = Table.read(script, format='csv', names=('file', 'type'))



    return

