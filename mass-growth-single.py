# -*- coding: utf-8 -*-
"""
Trace the mass growth of the cluster center.

Units:
    Mass: M_sun
    Length: kpc
    Time: Myr
    Velocity: km/s
"""

from __future__ import unicode_literals
import numpy as np
from numpy import linalg as LA
import operator

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True

import glob
import os
import random
import time
import h5py
from scipy.interpolate import interp1d

def release_list(a):
   del a[:]
   del a


# Samples the star mass using Kroupa 2001 IMF.   
def imf(xi):
    if xi < 0.371431:
        return ((xi + 0.112997) / 2.83835)**(1. / 0.7)
    elif xi < 0.849471:
        return ((1.50176 - xi) / 0.529826)**(-1. / 0.3)
    return ((1. - xi) / 0.0611337)**(-1. / 1.3)


# Returns the position of the star with density profile (with CDF).
def radius(mass_cdf, xi):
    n_tot = len(mass_cdf)
    mn = 0
    mx = n_tot
    while mx - mn > 1:
        mi = int(mn * 0.5 + mx * 0.5)
        if mass_cdf[mi][1] < xi:
            mn = mi
        else:
            mx = mi
    return mass_cdf[mi][0]

   
# Calculates the dynamical friction time scale for the star to collapse.
def t_df(r, m, mass_in, sigma):
    G = 6.67e-11
    r *= 3.08578e19
    m *= 1.989e30
    mass_in *= 1.989e30
    lamb = r * sigma**2 / G / m
    tdf = 1.17 / np.log(lamb) / m * np.sqrt(mass_in * r**3 / G)
    tdf /= 86400. * 365.2422 * 1e6
    return tdf

# Calculates the dynamical friction time scale for the star to collapse.
def t_df2(r, m, sigma, m_c, m_s):
    G = 6.67e-11
    r *= 3.08578e19
    m *= 1.989e30
    lamb = 0.1 * m_c / m_s
    tdf = r**2 * sigma / (0.86 * G * m * np.log(lamb))
    tdf /= 86400. * 365.2422 * 1e6
    return tdf


# Calculates the relaxation time scale.
def t_rlx(r_c, m_c, m_s):
    G = 6.67e-11
    r_c *= 3.08578e19
    m_c *= 1.989e30
    m_s *= 1.989e30
    lamb = 0.1 * m_c / m_s
    trlx = np.sqrt(r_c**3 / G / m_c) * m_c / m_s /(8.0 * np.log(lamb))
    trlx /= 86400. * 365.2422 * 1e6
    return trlx


# Returns the main sequence time scale of the star.
def t_ms(m):
    if m < 4.95249:
        return 1.0e10 / m**2.5 /1e6
    return 10.**(-0.3533*np.log10(m)**3. + 2.6422*np.log10(m)**2. - 6.4453*np.log10(m) + 5.5842)


# Gives the mass evolution of the star given its initial mass.
# This should be considered more detailly
def m_t(m, t):
    return m
    
t00 = time.time()
##### Read mass density info #####
nfiles=len(glob.glob1(os.getcwd(),"Cluster*-cdf.txt"))
#filename = 'Clusters_'+str(nfiles-1).zfill(2)+'.hdf5'
#filename = 'Cluster'+str(0).zfill(2)+'-cdf.txt'
filename = 'cluster-cdf0-703-703.txt'
mass_cdf = np.loadtxt(filename)
with open(filename, 'r') as f:
    line = f.readline()
    mass_cls = float(line[1:])*1e10
    line = f.readline()
    sigma = float(line[1:])*1e3
    print('sigma_v =', sigma)
    # Should be verified.
    
##### Get sigma_v #####
#filename = 'Clusters_'+str(nfiles-1).zfill(3)+'.hdf5'
#filename = 'Clusters_034.hdf5'
#with h5py.File(filename, 'r') as f:
#    v3d = f['Cluster02']['Velocities']
#    v1d = LA.norm(v3d, axis=1)
#    sigma = np.std(v1d) * 1000
#    print ('sigma_v = ', sigma)

##### Generate a catalog #####
    # mass; position; tdf; tms
star_catalog = np.empty((0, 4), float)
mass_tot=0.
m_cut = 100.
# mass cut-off for the IMF
r_c = radius(mass_cdf, 0.5)
print('half-mass radius =', r_c)
m_s = 0.376176
trlx = t_rlx(r_c, mass_cls, m_s)

while mass_tot < mass_cls:
    m = m_cut + 100.
    while m > m_cut:
        xi=random.random()
        m = imf(xi)
    xi=random.random()
    r = radius(mass_cdf, xi)
    mass_in = mass_cls * xi
    #sigma = sigma_v()
    #tdf = t_df(r, m, mass_in, sigma)
    
    tdf = 3.3 * m_s / m * t_rlx(r, mass_in, m_s)
    #tdf = t_df2(r, m, sigma, mass_cls, m_s)
    tms = t_ms(m)
    star_catalog = np.append(star_catalog, np.array([[m, r, tdf, tms]]), axis=0)
    mass_tot += m
    #print(time.time()-t00, m, mass_tot/mass_cls)

print('time =', time.time()-t00, 'mass_tot/mass_cls =', mass_tot/mass_cls)

star_catalog = np.transpose(star_catalog)
m = star_catalog[0]
tdf = star_catalog[2]
tms = star_catalog[3]


##### Trace the mass growth #####
tmax = 1
dt = .001
t = 0.1
mass_growth = np.empty((0, 2), float)
while t < tmax:
    #print(time.time()-t00)
    dm = sum(m_t(m, t) * np.heaviside(t - tdf, 1.) * np.heaviside(tms - tdf, 1.))
    mass_growth = np.append(mass_growth, np.array([[t, dm]]), axis = 0)
    t += dt

t -= dt    
tmax = 10000
dt = 1
while t < tmax:
    #print(time.time()-t00)
    dm = sum(m_t(m, t) * np.heaviside(t - tdf, 1.) * np.heaviside(tms - tdf, 1.))
    mass_growth = np.append(mass_growth, np.array([[t, dm]]), axis = 0)
    t += dt

mass_growth = np.transpose(mass_growth) 

print('time =', time.time()-t00)
  
fig, ax = plt.subplots(figsize = [4, 3])
ax.loglog(mass_growth[0], mass_growth[1], label=r'\rm Mass accumulated')
ax.loglog(mass_growth[0], np.repeat(mass_tot, len(mass_growth[0])), linestyle= '--', c = 'm', label=r'\rm Mass of the culster')
ax.loglog(np.repeat(trlx, len(mass_growth[0])), mass_growth[1], linestyle = '-.', c = 'y', label=r'$t_{\mathrm{rlx}}=%.2f\mathrm{Myr}$'%trlx)
#ax.text(2, 2e4, r'$t_{\mathrm{rlx}}=%f\mathrm{Myr}$'%trlx)
ax.set_xlabel(r"$t[\mathrm{Myr}]$")
ax.set_ylabel(r"$\Delta M_c[M_{\odot}]$")
ax.legend()
plt.tight_layout()
plt.draw()
plt.savefig('massgrowth.pdf')
plt.show()


##### Plot of central mass growth rate
fig, ax = plt.subplots(figsize = [4, 3])
skip = 500
mgx = mass_growth[0][::skip]
mgy = mass_growth[1][::skip]
mgrx = mgx[:-1]
mgry = (mgy[1:] - mgy[:-1])/(mgx[1:] - mgx[:-1])/1e6
mgr = interp1d(mgrx, mgry)
mg = interp1d(mass_growth[0], mass_growth[1]/3e6)
l = 0.1
h = 10000
while h-l > 0.01:
    m = (h+l)/2
    if mgr(m) < mg(m):
        h = m
    else:
        l = m
ax.loglog(mgrx, mgry, label = r'\rm Accumulation rate')
ax.loglog(mass_growth[0], mass_growth[1]/3e6, linestyle = '--', label=r'$M_{\mathrm{c}}/3\mathrm{Myr}$')
ax.text(m, mgr(m), r'\rm $t_{\rm e}=%.2f\mathrm{Myr}$, $M_{\rm c}(t_{\rm e})= %.2f M_{\odot}$'%(m, mg(m)*3e6))
ax.set_xlabel(r"$t[\mathrm{Myr}]$")
ax.set_ylabel(r"$\Delta \dot{M}_c[M_{\odot}/yr]$")
ax.legend()
plt.tight_layout()
plt.savefig('massgrowth-rate.pdf')
plt.show()


##### Plot of median mass and radius of collapsed stars
star_catalog = np.transpose(star_catalog)
star_catalog = [x for x in star_catalog if x[3] > x[2]]
star_catalog=sorted(star_catalog, key = operator.itemgetter(2))

mass_mean = np.empty((0, 2), float)
radius_mean = np.empty((0, 2), float)
l = 0
h = l
hm = len(star_catalog)
m = np.transpose(star_catalog)[0]
r = np.transpose(star_catalog)[1]

tmax = 1
dt = 0.3
t = 0.1
while t <= tmax:
    #print(time.time()-t00)
    for i in range(l,hm):
        if star_catalog[i][2] > t + dt:
            break
    h = i
    mass_mean = np.append(mass_mean, np.array([[t, np.mean(m[l:h])]]), axis = 0)
    radius_mean = np.append(radius_mean, np.array([[t, np.mean(r[l:h])*1000]]), axis = 0)
    l = h
    t += dt
    
t -= dt
tmax = 10000
dt = 50
while t < tmax:
    #print(time.time()-t00)
    for i in range(l,hm):
        if star_catalog[i][2] > t + dt:
            break
    h = i
    mass_mean = np.append(mass_mean, np.array([[t, np.mean(m[l:h])]]), axis = 0)
    radius_mean = np.append(radius_mean, np.array([[t, np.mean(r[l:h])*1000]]), axis = 0)
    l = h
    t += dt

mass_mean = np.transpose(mass_mean)
radius_mean = np.transpose(radius_mean)

fig, ax = plt.subplots(figsize = [4, 3])
ax.semilogx(mass_mean[0], mass_mean[1],label=r'\rm Mean mass of collapsed stars')
ax.semilogx(mass_mean[0], np.repeat(m_s, len(mass_mean[0])), linestyle= '--', c = 'm', label=r'\rm Mean mass of all stars')
ax.set_xlabel(r"$t[\mathrm{Myr}]$")
ax.set_ylabel(r"$\langle M \rangle /M_{\odot}$")
ax.legend()
plt.savefig('meanmass.pdf')
plt.show()

fig, ax = plt.subplots(figsize = [4, 3])
ax.semilogx(radius_mean[0], radius_mean[1], label = r'\rm Mean radius of collapsed stars')
ax.semilogx(radius_mean[0], np.repeat(r_c*1000, len(radius_mean[0])), linestyle= '--', c = 'm', label=r'\rm Effective radius $r_{\rm h}$')
ax.set_xlabel(r"$t[\mathrm{Myr}]$")
ax.set_ylabel(r"$\langle r \rangle /\mathrm{pc}$")
ax.legend(loc='center left')
plt.savefig('meanradius.pdf')
plt.show()