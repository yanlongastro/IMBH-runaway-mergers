# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:46:58 2020

@author: yanlo
"""

import numpy as np
from scipy import special
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from numba import jit

G = 4*np.pi**2/206265**3
m_s = 0.376176
f_bg = 0.799477 # cut off at 8 Msun

def m_enc(x, rho, a, b, rc):
    x = x/rc
    aa = 3-a
    bb = 1+a-b
    beta = x**aa / aa * special.hyp2f1(aa, 1-bb, aa+1, -x)
    res = 4*np.pi*rho*rc**3 * beta
    return res

def menc_inv(m, rho, a, b, rc):
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

def density(x, rho, a, b, rc):
    x = x/ rc
    return rho*x**(-a)*(1+x)**(a-b)

def ode_driver(r, t, m, m_c, r_c, rho, a, b, rc):
    """
    Here I assume that lnlamb>=1, or only consider the process when DF is 'effective'
    """
    if r>=r_c:
        temp = -4*np.pi*G**0.5*m
        menc = m_enc(r, rho, a, b, rc)
        lnlamb = max(np.log(0.1*menc/m_s), 1.0)
        temp *= lnlamb
        temp *= f_bg* density(r, rho, a, b, rc)*r**2.5
        temp /= (menc+m_c)**1.5
        return temp
    else:
        return 0.0

def t_step(r, t, m, m_c, r_c, rho, a, b, rc):
    if r<r_c:
        return np.inf
    integrand = lambda x : -1.0/ode_driver(x, t, m, m_c, r_c, rho, a, b, rc)
    res= integrate.quad(integrand, r_c, r)
    return res[0]

def t_df(r, m, rho, a, b, rc):
    menc = m_enc(r, rho, a, b, rc)
    lnlamb = np.log(0.1*menc/m_s)
    temp = 3.3/8/lnlamb
    temp *= np.sqrt(r**3/G/(menc))
    temp *= menc/m
    return temp
   
def rstar(m):
    return 260*m**0.5 * 695700/1.496e8/206265

#@jit(nopython=False, parallel=True)
def evolve(star_catalog, t_max, rho, a, b, rc, dt=1e5, ep=1e-5, test=False):
    m = np.copy(star_catalog[:,0])
    r = np.copy(star_catalog[:,1])*1e3
    tms = np.copy(star_catalog[:,3])*1e6
    t_max *= 1e6
    nstars = len(m)
    to_sink = list(range(nstars))
    tcol = np.repeat(np.nan, nstars)
    t_sample = np.linspace(0, 1.0, num=4)
    if test == True:
        t_sample = np.linspace(0, 1, num=100)
    t = 0.0
    m_c = 0.0
    r_c= max(rstar(8), menc_inv(np.e*m_s/0.1, rho, a, b, rc))

    times =[]
    masses = []
    times.append(t)
    masses.append(m_c)
    tc_est = np.zeros(nstars)
    while t< t_max:
        if (r<r_c).all() == True:
            break

        while np.sum(m[(r<r_c) & np.isnan(tcol) ]) !=0:
            print(sum((r<r_c) & np.isnan(tcol) ))
            tcol[(r<r_c) & np.isnan(tcol) ] =t
            m_c = np.sum(m[(r<r_c) & (tcol<tms)])
            r_c = max(rstar(8), menc_inv(np.e*m_s/0.1, rho, a, b, rc))
        for i in to_sink:
            tc_est[i] = t_step(r[i], t, m[i], m_c, r_c, rho, a, b, rc)
        ind = tc_est.argmin()
        if (tc_est==np.inf).all():
            break
        
        dt = t_step(r[ind], t, m[ind], m_c, r_c, rho, a, b, rc)*(1+ep)

        if r[ind]-r_c < ep:
            r[ind] -=ep
        if t+dt>t_max:
            print('\t Max time reached.')
            break
        for i in to_sink:
            sol= integrate.odeint(ode_driver, r[i], dt*t_sample, args=(m[i], m_c, r_c, rho, a, b, rc))
            r[i] = sol[-1]
            if test ==True:
                plt.semilogx(t+dt*t_sample, sol)
                plt.xlabel(r'$t[Myr]$')
                plt.ylabel(r'$r[pc]$')
                plt.show()
        t += dt
        tcol[(r<r_c) & np.isnan(tcol) ] =t
        m_c = np.sum(m[(r<r_c) & (tcol<tms)])
        r_c = max(rstar(8), menc_inv(np.e*m_s/0.1, rho, a, b, rc))
        print(t, m_c, r_c)
        times.append(t)
        masses.append(m_c)
    times = np.array(times)
    masses = np.array(masses)
    return np.transpose([times, masses])

if __name__ == '__main__':
    star_catalog=np.array([
            [8, .01/1e3, 1, 1e8],
            [9, .02/1e3, 1, 1e8],
            [10,.03/1e3, 1e1, 1e8],
            [11,.02/1e3, 1e2, 1e10],
            [50, 0.1/1e3, 10, 1e10],
            [100, 0.15/1e3, 10, 1e10]
            ])
    tm = evolve(star_catalog, 1e4, 1e11, 1, 4, 0.3, test=True)     
    plt.figure()
    plt.loglog(tm[:,0], tm[:,1])
    plt.xlabel(r'$t[Myr]$')
    plt.ylabel(r'$M_{\rm c}[M_{\odot}]$')
        
        