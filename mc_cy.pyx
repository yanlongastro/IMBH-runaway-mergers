# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:45:57 2019

Cython library for sampling the cluster star catalog

@author: Yanlong
"""

#!python
#cython: language_level=3

import numpy as np
import random
import operator
cimport cython

DTYPE = np.float64

# Samples the star mass using Kroupa 2001 IMF.
cdef double imf(double xi):
    #s = powerint(0.7, 0.08, 0.01) + powerint(-0.3, 0.01, 0.5) + poweint(-1.3, 0.5, 100)
    if xi < 0.371431:
        return ((xi + 0.112997) / 2.83835)**(1. / 0.7)
    elif xi < 0.849471:
        return ((1.50176 - xi) / 0.529826)**(-1. / 0.3)
    return ((1.000153584561505 - xi) / 0.06114311522145322)**(-1. / 1.3)

cdef double imf_partial(double xi):
    #s = powerint(0.7, 0.08, 0.01) + powerint(-0.3, 0.01, 0.5) + poweint(-1.3, 0.5, 100)
    return (2.462288826689833 - 2.459776940258323*xi)**(-1. / 1.3)

# Returns the position of the star with density profile (with CDF).
cdef double radius(double [:, ::1] mass_cdf, double xi):
    cdef Py_ssize_t n_tot = mass_cdf.shape[0]
    cdef Py_ssize_t mn = 0
    cdef Py_ssize_t mx = n_tot
    cdef Py_ssize_t mi
    if xi <= mass_cdf[0, 1]:
        return mass_cdf[0, 0]
    while mx - mn > 1:
        mi = int(mn * 0.5 + mx * 0.5)
        if mass_cdf[mi, 1] < xi:
            mn = mi
        else:
            mx = mi
    return (xi - mass_cdf[mn,1])/(mass_cdf[mx,1]-mass_cdf[mn,1])*(mass_cdf[mx,0]-mass_cdf[mn,0])+mass_cdf[mn,0]


# Calculates the relaxation time scale.
from libc.math cimport log
from libc.math cimport sqrt
from libc.math cimport log10
cdef double G = 6.67e-11
cdef double t_rlx(double r_c, double m_c, double m_s):
    r_c *= 3.08578e19
    m_c *= 1.989e30
    m_s *= 1.989e30
    cdef double lamb = 0.1 * m_c / m_s
    cdef double trlx = sqrt(r_c**3 / G / m_c) * m_c / m_s /(8.0 * log(lamb))
    trlx /= 86400. * 365.2422 * 1e6
    return trlx

def t_rlx_py(double r_c, double m_c, double m_s):
    r_c *= 3.08578e19
    m_c *= 1.989e30
    m_s *= 1.989e30
    cdef double lamb = 0.1 * m_c / m_s
    cdef double trlx = sqrt(r_c**3 / G / m_c) * m_c / m_s /(8.0 * log(lamb))
    trlx /= 86400. * 365.2422 * 1e6
    return trlx

# Returns the main sequence time scale of the star.
cdef double t_ms(double m):
    #return 1e20 # test only
    if m < 4.95249:
        return 1.0e10 / m**2.5 /1e6
    return 10.**(-0.3533*log10(m)**3. + 2.6422*log10(m)**2. - 6.4453*log10(m) + 5.5842)


# Gives the mass evolution of the star given its initial mass.
# This should be considered more detailly
cdef double m_t(double m, double t):
    return m

@cython.boundscheck(False)
@cython.wraparound(False)

# Samples the star catalog with the IMF and CDF of mass distribution
def sample(double mass_cls, double [:, ::1] cdf):
    rand = random.random
    cdef double m_s = 0.376176
    cdef Py_ssize_t length_catalog = int(mass_cls/m_s)
    star_catalog = np.zeros((length_catalog, 4), dtype=DTYPE)
    cdef double [:, :] star_catalog_view = star_catalog
    cdef Py_ssize_t i
    cdef double m_temp
    cdef double xi
    cdef double r_temp
    cdef double mass_in
    for i in range(length_catalog):
        m_temp = star_catalog_view[i, 0] = imf(rand())
        xi = rand()
        r_temp = star_catalog_view[i, 1] = radius(cdf, xi)
        mass_in = mass_cls * xi
        # dynamical friction time:
        star_catalog_view[i, 2] = 3.3 * m_s / m_temp * t_rlx(r_temp, mass_in, m_s)
        star_catalog_view[i, 2] = 3.3 * m_s / m_temp * t_rlx(r_temp, mass_in, m_s)* log(0.1* mass_in/m_s) / log(0.1* mass_cls/2./m_s)
        # main-sequence lifetime:
        star_catalog_view[i, 3] = t_ms(m_temp)
    return star_catalog

# As we only care about m>0.5, we may simply sample this range
def sample_partial(double mass_cls, double [:, ::1] cdf):
    rand = random.random
    cdef double m_s = 0.376176
    cdef Py_ssize_t length_catalog = int(mass_cls/m_s* 0.150398)
    star_catalog = np.zeros((length_catalog, 4), dtype=DTYPE)
    cdef double [:, :] star_catalog_view = star_catalog
    cdef Py_ssize_t i
    cdef double m_temp
    cdef double xi
    cdef double r_temp
    cdef double mass_in
    for i in range(length_catalog):
        m_temp = star_catalog_view[i, 0] = imf_partial(rand())
        xi = rand()
        r_temp = star_catalog_view[i, 1] = radius(cdf, xi)
        mass_in = mass_cls * xi
        # dynamical friction time:
        star_catalog_view[i, 2] = 3.3 * m_s / m_temp * t_rlx(r_temp, mass_in, m_s)
        #star_catalog_view[i, 2] = 3.3 * m_s / m_temp * t_rlx(r_temp, mass_in, m_s)* log(0.1* mass_in/m_s) / log(0.1* mass_cls/2./m_s)
        # main-sequence lifetime:
        star_catalog_view[i, 3] = t_ms(m_temp)
        # After this, t_df>t_ms stars will not be accessable:
        if star_catalog_view[i, 3] < star_catalog_view[i, 2]:
            star_catalog_view[i, 2] = 1.0e20
    return star_catalog

#def select_sort(double [:, ::1] star_catalog, double m_cut_l):
#    star_catalog_temp = [row for row in star_catalog if row[0] > m_cut_l and row[3] > row[2]]
#    return sorted(star_catalog, key = operator.itemgetter(2))

# Calculate the collapsed mass for a given time series:
def evolve(double [:, ::1] star_catalog_sorted, double [:] times):
    cdef Py_ssize_t n_step = times.shape[0]
    cdef Py_ssize_t n_star = star_catalog_sorted.shape[0]
    mass_growth = np.zeros((n_step), dtype=DTYPE)
    cdef double [:] mass_growth_view = mass_growth
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef double m_temp = 0.
    for i in range(n_step):
        while j < n_star:
            if star_catalog_sorted[j, 2] < times[i]:
                m_temp += star_catalog_sorted[j, 0]
                j += 1
            else:
                break
        mass_growth_view[i] = m_temp
    return mass_growth
