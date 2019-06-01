# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:35:33 2019

Unit: dimensionless unit and pc - M_sun - yr

@author: Yanlong
"""

import numpy as np
from scipy import special
import scipy.integrate as integrate
from scipy.misc import derivative


mmax=1
mmin=0.5/100
mmean = 0.376176/100
ind_imf = 2.3

def pdf(x, a, b):
    return x**(-a) *(1+x)**(a-b)

def cdf(x, a, b):
    aa = 3-a
    bb = 1+a-b
    #print(aa, bb)
    beta = x**aa / aa * special.hyp2f1(aa, 1-bb, aa+1, -x)
    #print(beta)
    return beta

def cdf_phy(x, rho, a, b, rc):
    x = x/rc
    return 4*np.pi*rho*rc**3 * cdf(x, a, b)

def cdf_inv(m, a, b):
    beta = m
    aa = 3-a
    bb = 1+a-b
    xmin = 1e-4
    xmax = 1e6
    alpha = 0.5
    while xmax-xmin > 1e-6:
        x = xmax*alpha + xmin*(1-alpha)
        if x**aa / aa * special.hyp2f1(aa, 1-bb, aa+1, -x) > beta:
            xmax = x
        else:
            xmin = x
    return x

def rh(a, b):
    mtot = cdf(1e8, a, b)
    return cdf_inv(mtot/2, a, b)

def cdf_inv_phy(m, rho, a, b, rc):
    beta = m/(4*np.pi*rho*rc**3)
    aa = 3-a
    bb = 1+a-b
    xmin = 1e-4
    xmax = 1e6
    alpha = 0.5
    while xmax-xmin > 1e-6:
        x = xmax*alpha + xmin*(1-alpha)
        if x**aa / aa * special.hyp2f1(aa, 1-bb, aa+1, -x) > beta:
            xmax = x
        else:
            xmin = x
    return x*rc

def mcrit(x, a, b, t):
    mrc = cdf(1, a, b)
    m = cdf(x, a, b)/mrc
    return (m*x**3)**0.5 /t
    
def tdfc_phy(rho, a, b, rc):
    G = 4*np.pi**2 / 206265**3
    m_max = 100
    m_mean = 0.376176
    mtot = cdf_phy(rc*1e8, rho, a, b, rc)
    mrc = cdf_phy(rc, rho, a, b, rc)
    lnlamb = np.log(0.1 * mtot/2/m_mean)
    tdfc = 3.3/8.0/lnlamb *np.sqrt(rc**3 /G/mrc) *mrc/m_max
    return tdfc

def rcrit(m_s, a, b, t):
    xmin = 1e-4
    xmax = 1e6
    alpha = 0.5
    while xmax-xmin > 1e-6:
        x = xmax*alpha + xmin*(1-alpha)
        if mcrit(x, a, b, t) > m_s:
            xmax = x
        else:
            xmin = x
    return x
    
def mct(a, b, t):
    r0 = rcrit(mmin, a, b, t)
    r1 = rcrit(mmax, a, b, t)
    int1 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-mcrit(x,a,b,t)**(2-ind_imf))*pdf(x,a,b)*x**2, r0, r1)[0]
    int2 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-mmin**(2-ind_imf))*pdf(x,a,b)*x**2, 0., r0)[0]
    return (int1+int2)*0.079486*100**(2-2.3)/0.376176

def mct_single_phy(rho, a, b, rc, t):
    G = 4*np.pi**2 / 206265**3
    m_max = 100
    m_mean = 0.376176
    mtot = cdf_phy(rc*1e8, rho, a, b, rc)
    mrc = 4* np.pi * rho*rc**3/(3-a)
    lnlamb = np.log(0.1 * mtot/2/m_mean)
    tdfc = 3.3/8.0/lnlamb *np.sqrt(rc**3 /G/mrc) *mrc/m_max
    res = 4* np.pi * rho*rc**3
    res *= 0.079486*100**(2-2.3)/0.376176
    res *= (t/tdfc)**(2*(3-a)/(6-a))
    res *= 1/(2-2.3)*(1/(3-a)-1/((3-a) + (6-a)*(2-2.3)/2))*(1-mmean**(2-2.3+2*(3-a)/(6-a)))
    return res


def mct_dt(a, b, t):
    return derivative(lambda x: mct(a, b, x), t, dx=1e-6)

def mct_dt_single_phy(rho, a, b, rc, t):
    return mct_single_phy(rho, a, b, rc, t)/t*2*(3-a)/(6-a)

def mte(a, b, tdfc):
    tmin = 1.0e-5
    tmax = 1.0e5
    alpha = 0.5
    while tmax-tmin > 1e-6:
        t = tmax*alpha + tmin*(1-alpha)
        if mct(a,b,t)/3.0e6 > derivative(lambda x: mct(a, b, x), t, dx=1e-6)/tdfc:
            tmax = t
        else:
            tmin = t
    return mct(a,b,t), t