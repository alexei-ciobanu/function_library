import sympy as sp

def lorentzian2(fwhm,x,x0=0):
    '''maximum of lorentzian is 1.0'''
    return (fwhm/2)**2/((x-x0)**2+(fwhm/2)**2)
