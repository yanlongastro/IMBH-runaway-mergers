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

def pdf_pw(x, a, b):
    return 1/(x**a + x**b)

def cdf(x, a, b):
    aa = 3-a
    bb = 1+a-b
    #print(aa, bb)
    beta = x**aa / aa * special.hyp2f1(aa, 1-bb, aa+1, -x)
    #print(beta)
    return beta

def cdf_pw(x, a, b):
    ab = (a-3.) /(a-b)
    #print(aa, bb)
    hgf = x**(3.-a) / (3.-a) * special.hyp2f1(1, ab, ab+1, -x**(b-a))
    #print(beta)
    return hgf

def cdf_phy(x, rho, a, b, rc):
    x = x/rc
    return 4*np.pi*rho*rc**3 * cdf(x, a, b)

def cdf_pw_phy(x, rho, a, b, rc):
    x = x/rc
    return 4*np.pi*rho*rc**3 * cdf_pw(x, a, b)

def cdf_inv(m, a, b):
    beta = m
    aa = 3-a
    bb = 1+a-b
    xmin_lg = -10
    xmax_lg = 10
    alpha = 0.5
    while xmax_lg-xmin_lg > 1e-10:
        x = xmax_lg*alpha + xmin_lg*(1-alpha)
        if (10**x)**aa / aa * special.hyp2f1(aa, 1-bb, aa+1, -10**x) > beta:
            xmax_lg = x
        else:
            xmin_lg = x
    return 10**x

def rh(a, b):
    mtot = cdf(1e10, a, b)
    return cdf_inv(mtot/2, a, b)

def cdf_inv_phy(m, rho, a, b, rc):
    beta = m/(4*np.pi*rho*rc**3)
    aa = 3-a
    bb = 1+a-b
    xmin = 1e-4
    xmax = 1e10
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
    mtot = cdf_phy(rc*1e10, rho, a, b, rc)
    mrc = cdf_phy(rc, rho, a, b, rc)
    lnlamb = np.log(0.1 * mtot/2/m_mean)
    tdfc = 3.3/8.0/lnlamb *np.sqrt(rc**3 /G/mrc) *mrc/m_max
    return tdfc

def tdfc_pw_phy(rho, a, b, rc):
    G = 4*np.pi**2 / 206265**3
    m_max = 100
    m_mean = 0.376176
    mtot = cdf_pw_phy(rc*1e10, rho, a, b, rc)
    mrc = cdf_pw_phy(rc, rho, a, b, rc)
    lnlamb = np.log(0.1 * mtot/2/m_mean)
    tdfc = 3.3/8.0/lnlamb *np.sqrt(rc**3 /G/mrc) *mrc/m_max
    return tdfc

def t_friction_phy(rho, a, b, rc, m, s):
    G = 4*np.pi**2 / 206265**3
    mm = lambda x: cdf_phy(x, rho, a, b, rc)
    pdf_phy = lambda x: rho*pdf(x/rc, a, b)
    v_c = lambda x: np.sqrt(G*mm(x)/x)
    m_mean = 0.376176
    mtot = cdf_phy(rc*1e10, rho, a, b, rc)
    lnlamb = np.log(0.1 * mtot/2/m_mean)
    tdfc = integrate.quad(lambda x: (4*np.pi*x**3*pdf_phy(x) -mm(x))/(2*x**2)*v_c(x)/(0.5*pdf_phy(x)), 0, s)[0]
    tdfc /= (4*np.pi*G*m*lnlamb)
    return tdfc

def mcrit_friction_phy(rho, a, b, rc, t, s):
    G = 4*np.pi**2 / 206265**3
    mm = lambda x: cdf_phy(x, rho, a, b, rc)
    pdf_phy = lambda x: rho*pdf(x/rc, a, b)
    v_c = lambda x: np.sqrt(G*mm(x)/x)
    m_mean = 0.376176
    mtot = cdf_phy(rc*1e10, rho, a, b, rc)
    lnlamb = np.log(0.1 * mtot/2/m_mean)
    mc = integrate.quad(lambda x: (4*np.pi*x**3*pdf_phy(x) -mm(x))/(2*x**2)*v_c(x)/(0.5*pdf_phy(x)), 0, s)[0]
    mc /= (4*np.pi*G*t*lnlamb)
    return mc

def rcrit(m_s, a, b, t):
    xmin_lg = -10
    xmax_lg = 10
    alpha = 0.5
    while xmax_lg-xmin_lg > 1e-10:
        x = xmax_lg*alpha + xmin_lg*(1-alpha)
        if mcrit(10**x, a, b, t) > m_s:
            xmax_lg = x
        else:
            xmin_lg = x
    return 10**x

def rcrit_friction_phy(rho, a, b, rc, t, m_s):
    xmin_lg = -10
    xmax_lg = np.log10(3.125822832*rc)
    if mcrit_friction_phy(rho, a, b, rc, t, 10**xmax_lg) < 100 and m_s ==100:
        return 10**xmax_lg
    alpha = 0.5
    while xmax_lg-xmin_lg > 1e-10:
        x = xmax_lg*alpha + xmin_lg*(1-alpha)
        if mcrit_friction_phy(rho, a, b, rc, t, 10**x) > m_s:
            xmax_lg = x
        else:
            xmin_lg = x
    return 10**x
    
def mct(a, b, t):
    r0 = rcrit(mmin, a, b, t)
    r1 = rcrit(mmax, a, b, t)
    int1 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-mcrit(x,a,b,t)**(2-ind_imf))*pdf(x,a,b)*x**2, r0, r1)[0]
    int2 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-mmin**(2-ind_imf))*pdf(x,a,b)*x**2, 0., r0)[0]
    return (int1+int2)*0.079486*100**(2-2.3)/0.376176

def mct_phy(rho, a, b, rc, t):
    return mct(a, b, t/tdfc_phy(rho, a, b, rc))*4*np.pi*rho*rc**3

def mqt_phy(rho, a, b, rc, t):
    #mct_phy = lambda x: mct(a, b, x/tdfc_phy(rho, a, b, rc))*4*np.pi*rho*rc**3
    res = integrate.quad(lambda x: mct_phy(rho, a, b, rc, x)*np.exp(x/3e6)/3e6, 0, t)[0]
    res = mct_phy(rho, a, b, rc, t) - np.exp(-t/3e6)*res
    return res

def mct_friction_phy(rho, a, b, rc, t):
    r0 = rcrit_friction_phy(rho, a, b, rc, t, mmin*100)
    r1 = rcrit_friction_phy(rho, a, b, rc, t, 100)
    #print(r0, r1)
    int1 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-(mcrit_friction_phy(rho, a, b, rc, t, x)/100)**(2-ind_imf))*rho*pdf(x/rc,a,b)*x**2, r0, r1)[0]
    int2 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-mmin**(2-ind_imf))*rho*pdf(x/rc,a,b)*x**2, 0., r0)[0]
    return (int1+int2)*0.079486*100**(2-2.3)/0.376176 *4*np.pi

def mct_friction_iter(rho, a, b, rc, t, i):
    r0 = rcrit_friction_phy(rho, a, b, rc, t, mmin*100)
    r1 = rcrit_friction_phy(rho, a, b, rc, t, 100)
    if i >0:
        m = mct_friction_iter(rho, a, b, rc, t, i-1)
        rq = (m/100)**0.5 *2600*695700/1.496e8/206265
        #print(i, rq*206265, m)
    else:
        rq = 0.
    int1 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-(mcrit_friction_phy(rho, a, b, rc, t, x)/100)**(2-ind_imf))*rho*pdf(x/rc,a,b)*x**2, r0, r1)[0]
    int2 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-mmin**(2-ind_imf))*rho*pdf(x/rc,a,b)*x**2, rq, r0)[0]
    return (int1+int2)*0.079486*100**(2-2.3)/0.376176 *4*np.pi

def mct_lg(a, b, t):
    r0 = rcrit(mmin, a, b, t)
    r1 = rcrit(mmax, a, b, t)
    r0, r1 = np.log(r0), np.log(r1)
    print(mcrit(np.exp(r0),a,b,t), mcrit(np.exp(r1),a,b,t))
    int1 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-mcrit(np.exp(x),a,b,t)**(2-ind_imf))*pdf(np.exp(x),a,b)*np.exp(x)**3, r0, r1)[0]
    int2 = 1.0/(2-ind_imf)*integrate.quad(lambda x: (1-mmin**(2-ind_imf))*pdf(np.exp(x),a,b)*np.exp(x)**3, 0., r0)[0]
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

def mct_dt_phy(rho, a, b, rc, t):
    return derivative(lambda x: mct_phy(rho, a, b, rc, x), t, dx=1e-6)

def mct_dt_friction_phy(rho, a, b, rc, t):
    return derivative(lambda x: mct_friction_phy(rho, a, b, rc, x), t, dx=1e-6)

def mct_dt_friction_iter(rho, a, b, rc, t, i):
    return derivative(lambda x: mct_friction_iter(rho, a, b, rc, x, i), t, dx=1e-6)

def mct_dt_single_phy(rho, a, b, rc, t):
    return mct_single_phy(rho, a, b, rc, t)/t*2*(3-a)/(6-a)

def mte(a, b, tdfc):
    tmin = 1.0e-10
    tmax = 1.0e10
    alpha = 0.5
    while tmax-tmin > 1e-6:
        t = tmax*alpha + tmin*(1-alpha)
        if mct(a,b,t)/3.0e6 > derivative(lambda x: mct(a, b, x), t, dx=1e-6)/tdfc:
            tmax = t
        else:
            tmin = t
    return mct(a,b,t), t

def mte_lg(a, b, tdfc):
    tmin_lg = -10
    tmax_lg = 10
    alpha = 0.5
    while tmax_lg-tmin_lg > 1e-10:
        t = tmax_lg*alpha + tmin_lg*(1-alpha)
        if mct(a,b,10**t)/3.0e6 > derivative(lambda x: mct(a, b, 10**x), t, dx=1e-6)/10**t/np.log(10)/tdfc:
            tmax_lg = t
        else:
            tmin_lg = t
    return mct(a,b,10**t), 10**t

def mte_lg_friction_phy(rho, a, b, rc):
    tmin_lg = -10
    tmax_lg = 10
    alpha = 0.5
    while tmax_lg-tmin_lg > 1e-10:
        t = tmax_lg*alpha + tmin_lg*(1-alpha)
        if mct_friction_phy(rho, a, b, rc, 10**t)/3.0e6 > derivative(lambda x: mct_friction_phy(rho, a, b, rc, 10**x), t, dx=1e-6)/10**t/np.log(10):
            tmax_lg = t
        else:
            tmin_lg = t
    return mct_friction_phy(rho, a, b, rc, 10**t), 10**t

def mte_lg_friction_iter(rho, a, b, rc, i):
    tmin_lg = -10
    tmax_lg = 10
    alpha = 0.5
    while tmax_lg-tmin_lg > 1e-10:
        t = tmax_lg*alpha + tmin_lg*(1-alpha)
        if mct_friction_iter(rho, a, b, rc, 10**t, i)/3.0e6 > derivative(lambda x: mct_friction_iter(rho, a, b, rc, 10**x, i), t, dx=1e-6)/10**t/np.log(10):
            tmax_lg = t
        else:
            tmin_lg = t
    return mct_friction_iter(rho, a, b, rc, 10**t, i), 10**t

def mte_single_phy(rho, a, b, rc):
    return mct_single_phy(rho, a, b, rc, 6*(3-a)/(6-a)*1e6)