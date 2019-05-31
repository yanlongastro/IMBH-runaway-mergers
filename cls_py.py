# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:23:20 2019

Functions for smoothing generating the pdf, cdf.

@author: Yanlong
"""

from __future__ import unicode_literals
import numpy as np

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
from scipy.optimize import curve_fit
import sys
import glob
import lmfit
from scipy import signal
from scipy import interpolate
from scipy import special
from scipy import optimize
from scipy import stats


def release_list(a):
   del a[:]
   del a

def func(x, rho, a, b, rc):
    return np.log(np.exp(rho)/(np.exp(x)/np.exp(rc))**a / (1.+np.exp(x)/np.exp(rc))**(b-a))

def func_pw(x, rho, a, b, rc):
    return np.log(np.exp(rho)/((np.exp(x)/np.exp(rc))**a + (np.exp(x)/np.exp(rc))**b ) )

def func_cdf(x, rho, a, b, rc):
    rc = np.exp(rc)
    rho = np.exp(rho)
    x = np.exp(x)
    x = x/rc
    aa = 3-a
    bb = 1+a-b
    beta = x**aa / aa * special.hyp2f1(aa, 1-bb, aa+1, -x)
    try:
        res = np.log(4*np.pi*rho*rc**3 * beta)
    except:
        pass
    return res

def func_cdf_inv(frac, rho, a, b, c):
    m_h = func_cdf(np.log(1e8), rho, a, b, c) + np.log(frac)
    rmin = np.log(1e-10)
    rmax = np.log(1e10)
    alpha = 0.5
    while rmax - rmin > 1e-6:
        rmid = alpha*rmax + (1.-alpha)*rmin
        if func_cdf(rmid, rho, a, b, c)>m_h:
            rmax = rmid
        else:
            rmin = rmid
    rmid = alpha*rmax + (1.-alpha)*rmin
    return np.exp(rmin)
    

def func_cdf_pw(x, rho, a, b, rc):
    rc = np.exp(rc)
    rho = np.exp(rho)
    x = np.exp(x)
    x = x/rc
    #print(rho, x, rc)
    ab = (a-3.) /(a-b)
    #print(aa, bb)
    hgf = x**(3.-a) / (3.-a) * special.hyp2f1(1, ab, ab+1, -x**(b-a))
    return np.log(4*np.pi*rho*rc**3 * hgf)

def cdf_sample(r_m):
    rmin = r_m[3,0] *1.01
    rmax = r_m[-1,0] *1.01
    n_points = 500
    r = np.logspace(np.log10(rmin), np.log10(rmax), num=n_points)
    mcum = np.zeros(n_points)
    mt = 0.
    i = j = 0
    for i in range(n_points):
        if j>=len(r_m):
            mcum[i] = mt
            continue
        while r_m[j, 0]<r[i]:
            mt += r_m[j, 1]
            j += 1
            if j >=len(r_m):
                break
        mcum[i] = mt
        #print(r[i], r_m[j-1,0], mcum[i])
    return np.array(list(zip(r, mcum)))

def pdf_sample(r_m):
    rmin = r_m[3,0] *1.01
    rmax = r_m[-1,0] *1.01
    n_points = 20
    r = np.logspace(np.log10(rmin), np.log10(rmax), num=n_points-1)
    eta = r[2]/r[1]
    dmcum = np.zeros(n_points-1)
    i = j = 0
    for i in range(n_points-1):
        while j < len(r_m):
            if r_m[j, 0]<r[i+1]:
                dmcum[i]+=r_m[j, 1]
                j+=1
                continue
            else:
                break
        dmcum[i] /= ((r[i]*eta)**3 - r[i]**3)*4.*np.pi/3.
        #print(r[i], r_m[j-1,0], mcum[i])
    result = np.array(list(zip(r*np.sqrt(eta), dmcum)))
    return result[np.all(result > 1e-3, axis=1)]
    

def cdf_smooth(raw_cdf):
    n_points = 500
    x = np.log(raw_cdf[:,0])
    y = np.log(raw_cdf[:,1])
    #tck = interpolate.splrep(x, y, s=0)
    #tck, u = interpolate.splprep([x, y], s=0)
    f = interpolate.interp1d(x, y, kind='linear')
    xnew = np.linspace(x[0], x[-1], n_points)
    #ynew = interpolate.splev(xnew, tck, der=0)
    ynew = f(xnew)
    #spl = interpolate.UnivariateSpline(xnew, ynew)
    #spl.set_smoothing_factor(0.9)
    #ynew = spl(xnew)
    ynew = signal.savgol_filter(ynew, 349, 2)
    return np.array(list(zip(np.exp(xnew), np.exp(ynew))))

def cdf_smooth_cheby(raw_cdf):
    n_points = 500
    x = np.log(raw_cdf[:,0])
    y = np.log(raw_cdf[:,1])
    #tck = interpolate.splrep(x, y, s=0)
    #tck, u = interpolate.splprep([x, y], s=0)
    f = interpolate.interp1d(x, y, kind='linear')
    xnew = np.linspace(x[0], x[-1], n_points)
    #ynew = interpolate.splev(xnew, tck, der=0)
    ynew = f(xnew)
    #spl = interpolate.UnivariateSpline(xnew, ynew)
    #spl.set_smoothing_factor(0.9)
    #ynew = spl(xnew)
    #ynew = signal.savgol_filter(ynew, 349, 2)
    cheby = np.polynomial.Chebyshev.fit(xnew, ynew, 4)
        #y = signal.savgol_filter(y, len(x)//8*2+1, 3)
    ynew = cheby(xnew)
    return np.array(list(zip(np.exp(xnew), np.exp(ynew))))

def cdf_smooth_mono(raw_cdf):
    return

def pdf_cal(cdf):
    x = np.log(cdf[:,0])
    y = np.log(cdf[:,1])
    dydx_log = np.diff(y)/np.diff(x)
    z = dydx_log * np.exp(y[:-1])/4./np.pi/(np.exp(x[:-1]))**3
    return np.array(list(zip(np.exp(x[:-1]), z)))

def fit_pdf(pdf):
    fmodel = lmfit.Model(func)
    #fmodel = lmfit.Model(func_pw)
    fmodel.set_param_hint('a', min=0)
    x = np.log(pdf[:, 0])
    y = np.log(pdf[:, 1])
    result = fmodel.fit(y, x=x, rho = 12., a = 0., b=3., rc =-2.)
    #print(result.fit_report())
    params = list(result.best_values.values())
    params[0] = np.exp(params[0])
    params[-1] = np.exp(params[-1])
    print(params)
    return np.array(list(zip(np.exp(x), np.exp(result.best_fit)))), params

def fit_cdf(raw_cdf, r_h, m_tot):
    for j in range(len(raw_cdf)):
        if raw_cdf[j][1] > raw_cdf[-1, 1]/2.:
            break
    weights = np.ones(len(raw_cdf))
    #weights[[j, -1]] = 50.
    #print(raw_cdf[j, 0])
    
    fmodel = lmfit.Model(func_cdf)
    #print(m_tot, r_h)
    #print((m_tot/2.0/(4*np.pi*np.exp(-2)**3)/((r_h/np.exp(-2))**(3-1)/(3-1) * special.hyp2f1(3-1, 4-1, 4-1, -r_h/np.exp(-2)))))
    #fmodel.set_param_hint('rho', expr='log(%f/2.0/(4*pi*exp(rc)**3)/((%f/exp(rc))**(3-a)/(3-a) * special.hyp2f1(3-a, b-a, 4-a, -%f/exp(rc))))'%(m_tot, r_h, r_h), vary=True)
    #fmodel.set_param_hint('rc', expr='a+b')
    fmodel.set_param_hint('a', value=1, min=0)
    #fmodel.set_param_hint('m_tot', expr='2* (4*pi*exp(rc)**3)*exp(rho)*((r_h/exp(rc))**(3-a)/(3-a) * special.hyp2f1(3-a, b-a, 4-a, -r_h/exp(rc)))')
    #fmodel.set_param_hint('b',)
    x = np.log(raw_cdf[:, 0])
    y = np.log(raw_cdf[:, 1])
    result = fmodel.fit(y, x=x, rho = 12., a = 1, b=4, rc =-2., method='least_square', weights=weights)
    #print(result.fit_report())
    params = list(result.best_values.values())    
    
    params[0] = np.exp(params[0])
    params[-1] = np.exp(params[-1])
    print(params)
    return np.array(list(zip(np.exp(x), np.exp(result.best_fit)))), params

def fit_cdf_pw(raw_cdf):
    for j in range(len(raw_cdf)):
        if raw_cdf[j][1] > raw_cdf[-1, 1]/2.:
            break
    weights = np.ones(len(raw_cdf))
    weights[[j, -1]] = 50.
    #print(raw_cdf[j, 0])
    
    fmodel = lmfit.Model(func_cdf_pw)
    fmodel.set_param_hint('a', value=1, min=0)
    #fmodel.set_param_hint('b')
    x = np.log(raw_cdf[:, 0])
    y = np.log(raw_cdf[:, 1])
    result = fmodel.fit(y, x=x, rho = 12., a = 1, b=4, rc =-2., method='least_squares', weights=weights)
    #print(result.fit_report())
    params = list(result.best_values.values())    
    
    params[0] = np.exp(params[0])
    params[-1] = np.exp(params[-1])
    print(params)
    return np.array(list(zip(np.exp(x), np.exp(result.best_fit)))), params


def fit_cdf_chi2(x, r, m):
    model = func_cdf(r, *x)
    chi_sq = sum((model - m)**2)
    return chi_sq

def fit_cdf_scipy(raw_cdf, r_h, m_h, r_max, m_tot):
    r = np.log(raw_cdf[:,0])
    m = np.log(raw_cdf[:,1])
    fun_con = lambda x: func_cdf(np.log(r_h), *x) - np.log(m_h)
    fun_con_tot = lambda x: func_cdf(np.log(r_max), *x) - np.log(m_tot)
    delta = 0
    cons = ({'type': 'eq', 'fun': fun_con},
            {'type': 'eq', 'fun': fun_con_tot},
        {'type': 'ineq', 'fun': lambda x: x[1]-delta},
        {'type': 'ineq', 'fun': lambda x: 3.0-x[1]-delta},
        {'type': 'ineq', 'fun': lambda x: x[2]-3.0-delta})
    res = optimize.minimize(fit_cdf_chi2, (12, 1, 4, -1), args=(r, m), method='SLSQP', constraints=cons)
    params = res.x
    fits = np.array(func_cdf(r, *params))
    chi_sq_test = stats.chisquare(m, f_exp=fits)
    fits = np.exp(fits)
    print(fun_con(params), fun_con_tot(params), chi_sq_test)
    
    if res.success == False or chi_sq_test[1]<0.05 or fun_con(params)>1e-5 or fun_con_tot(params)>1e-5:
        params[2] = -1.0
    params[0] = np.exp(params[0])
    params[1] = np.abs(params[1])
    params[-1] = np.exp(params[-1])
    r_h_fit = func_cdf_inv(0.5, np.log(params[0]), params[1], params[2], np.log(params[3]))
    if params[-1] > r_max or r_h_fit > r_max:
        params[2] = -1.0
    return np.array(list(zip(np.exp(r), fits))), params