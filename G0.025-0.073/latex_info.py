import numpy as np
from astropy import units as u
from astropy.io import ascii

def exp_to_tex(st):
    if st == 'nan':
        return '-'
    elif 'e' in st:
        pt1,pt2 = st.split('e')
        return "{0}\\ee{{{1:d}}}".format(pt1,int(pt2))
    return st

def format_float(st):
    return exp_to_tex("{0:0.2g}".format(st))


latexdict = ascii.latex.latexdicts['AA']
latexdict['tabletype'] = 'table*'
latexdict['tablealign'] = 'htp'

def ndigits(error, extra=1):
    return int(np.ceil(-np.log10(error))) + extra

def rounded(value, error, extra=1):
    """
    Return the value and error both rounded to the error's first digit
    """

    if error == 0:
        return (0,0)

    if hasattr(value, 'unit'):
        value = value.value

    digit = ndigits(error, extra=extra)
    assert np.round(error, digit) != 0
    return np.round(value, digit), np.round(error, digit)#, digit

def round_to_n(x, n):
    if np.isnan(x):
        return np.nan
    elif x == 0:
        return 0
    else:
        return round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))

def strip_trailing_zeros(x):
    if '.' in x:
        y = x.rstrip("0")
        return y.rstrip(".")
    else:
        return x
