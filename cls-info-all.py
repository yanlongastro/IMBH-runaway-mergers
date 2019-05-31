# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:40:37 2019

Extract cluster information from the hdf5 cluster data
- params: center, m_tot, n_particle, r_h, rho_c, eta_1, eta_2, r_c, m_rc, m_tot_fitted; 
- raw_cdf;
- cdf;
- raw_pdf;
- fitted_pdf.

@author: Yanlong
"""

from __future__ import unicode_literals
import numpy as np
import operator

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
import h5py
#from hfof import fof
import glob
import os
import sys
from numpy import linalg as LA
import cls_py

def release_list(a):
   del a[:]
   del a

##### Read data #####
pwd = os.getcwd()
path = os.path.join(pwd, *sys.argv[1:])
pw = False
print(path)
nfiles=len(glob.glob1(path, "snapshot*.hdf5"))
nfiles = 38
filename = 'Clusters_'+str(nfiles-1).zfill(3)+'.hdf5'
f = h5py.File(os.path.join(path, filename), 'r')
n_cls = len(f.keys())
#dat = np.loadtxt('bound_'+str(nfiles-1).zfill(3)+'.dat')
dat = np.loadtxt(os.path.join(path, 'bound_'+str(nfiles-1).zfill(3)+'-fixed.dat'))
if dat.ndim == 1:
    dat = dat.reshape((1, dat.shape[0]))

#n_cls
outfilename = "-".join(sys.argv[1:])+'-'+filename[:-5]+('-info-pw.hdf5' if pw else '-info.hdf5')
hdf = h5py.File(os.path.join(pwd, 'clusters-density', outfilename), 'w')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[12,16], tight_layout=True)
#fig, ax1 = plt.subplots(figsize=[12,8])
for i in range(n_cls):
    clst = list(f.keys())[i]
    center = dat[i][1:4]
    m_tot = dat[i][0] * 1.0e10
    n_particle = len(f[clst]['Masses'])
    if n_particle < 300:
        break
    r_h = dat[i][4] * 1.0e3
    pos = np.array(f[clst]['Coordinates'])
    radius = LA.norm(pos - center, axis = 1) * 1.0e3
    masses = np.array(f[clst]['Masses']) * 1.0e10
    r_m = np.array(list(zip(radius, masses)))
    r_m = sorted(r_m, key=operator.itemgetter(0))
    r_m = np.array(r_m)[1:]
    radius = r_m[:,0]
    masses = r_m[:,1]
    mass_cum = np.cumsum(masses)
    raw_cdf = np.array(list(zip(radius, mass_cum)))
    cdf = cls_py.cdf_smooth(raw_cdf)
    #cdf = cls_py.cdf_smooth_cheby(raw_cdf[5:])
    cdf_s = cls_py.cdf_sample(r_m)
    pdf_s = cls_py.pdf_sample(r_m)
    pdf = cls_py.pdf_cal(cdf)
    pdf = pdf[pdf[:, 1]>0, :]
    
    for j in range(len(cdf_s)):
        if cdf_s[j][1] > cdf_s[-1, 1]/2.:
            break
    r_h_s = cdf_s[j, 0]
    #print(r_h, r_h_s)
    
    try:
        #fits, params = cls_py.fit_pdf(pdf[len(pdf)//100:-len(pdf)//1000])
        #pw = True
        tofit_cdf = cdf_s
        
        if pw==True:
            fits, params = cls_py.fit_cdf_pw(tofit_cdf)
            func_cdf = cls_py.func_cdf_pw
        else:
            #fits, params = cls_py.fit_cdf(tofit_cdf, cdf_s[j,0], cdf_s[j,1])
            fits, params = cls_py.fit_cdf_scipy(tofit_cdf, cdf_s[j,0], cdf_s[j,1], cdf_s[-1,0], cdf_s[-1,1])
            func_cdf = cls_py.func_cdf
        
        #fits, params = cls_py.fit_cdf(tofit_cdf)
        #fits, params = cls_py.fit_cdf_pw(tofit_cdf)
        #fits, params = cls_py.fit_cdf(cls_py.cdf_sample(r_m))
        if params[2]<0:
            continue
        fits_pdf = np.zeros((tofit_cdf.shape[0]-1, tofit_cdf.shape[1]))
        fits_pdf[:,0] = tofit_cdf[:-1,0]
        fits_pdf[:,1] = np.diff(fits[:,1])/np.diff(fits[:,0])/4/np.pi/fits[:-1,0]**2
        for k in range(len(tofit_cdf)):
            if tofit_cdf[k][0]>params[-1]:
                break
        mrc_s = tofit_cdf[k][1]
        mrc_fit = np.exp(func_cdf(np.log(params[3]), np.log(params[0]), params[1], params[2], np.log(params[3])))
        m_tot_fit = np.exp(func_cdf(np.log(1e8), np.log(params[0]), params[1], params[2], np.log(params[3])))
        r_h_fit = cls_py.func_cdf_inv(0.5, np.log(params[0]), params[1], params[2], np.log(params[3]))
        #print(r_h, cdf_s[j,0], r_h_fit)
        params = np.concatenate((params, [mrc_fit, m_tot_fit, r_h_fit]))
        
        r_mc_min = cls_py.func_cdf_inv(0.5/m_tot_fit, np.log(params[0]), params[1], params[2], np.log(params[3]))
        r_mc_max = cls_py.func_cdf_inv(0.99, np.log(params[0]), params[1], params[2], np.log(params[3]))
        n_mc_cdf = 5000
        mc_cdf = np.zeros((n_mc_cdf, 2))
        mc_cdf[:,0] = np.logspace(np.log10(r_mc_min), np.log10(r_mc_max), num=n_mc_cdf)
        mc_cdf[:,1] = np.exp(func_cdf(np.log(mc_cdf[:,0]), np.log(params[0]), params[1], params[2], np.log(params[3])))
        
        clst_hdf = hdf.create_group(clst)
        clst_hdf.create_dataset('center', data = center)
        clst_hdf.create_dataset('params0', data = [m_tot, n_particle, r_h])
        clst_hdf.create_dataset('params', data = params)
        clst_hdf.create_dataset('raw_cdf', data = raw_cdf)
        clst_hdf.create_dataset('cdf', data = cdf)
        clst_hdf.create_dataset('raw_pdf', data = pdf)
        clst_hdf.create_dataset('fitted_cdf', data = fits)
        clst_hdf.create_dataset('fitted_pdf', data = fits_pdf)
        clst_hdf.create_dataset('mc_cdf', data = mc_cdf)
        clst_hdf.create_dataset('cdf_s', data = cdf_s)
        clst_hdf.create_dataset('pdf_s', data = pdf_s)
        
        
        #print(cdf[-1, 1], m_tot)
        #plt.loglog(radius, mass_cum)
        #plt.loglog(cdf[:,0], cdf[:,1])
        label = r'$\rho_c=%.2e M_{\odot}/pc^3$, $\eta_1=%.2f $, $\eta_2=%.2f$, $r_c = %.2f pc$, $M_{\rm c} = %.2e M_{\odot}$, $M_{\rm tot} = %.2e M_{\odot}$, $r_{\rm h} = %.2f pc$'%tuple(params)
        ax1.loglog(pdf[:,0], pdf[:,1], label=label)
        ax1.loglog(fits_pdf[:, 0], fits_pdf[:, 1], color='k', linestyle='--')
        ax1.loglog(pdf_s[:, 0], pdf_s[:, 1], color='grey', linestyle=':')
        ax2.loglog(raw_cdf[:,0], raw_cdf[:,1])
        ax2.loglog(fits[:,0], fits[:,1], color='k', linestyle='--')
        #ax2.loglog(mc_cdf[:,0], mc_cdf[:,1], color='r', linestyle='--')
        ax2.loglog(cdf_s[j,0], cdf_s[j,1], 'r.', label=r'$r_h$')
        ax2.loglog(r_h_fit, m_tot_fit/2., 'g+', label=r'$r_h$')
        ax2.loglog(params[3], mrc_fit, 'b*', label=r'$r_c$')
        #ax.loglog(cdf[:,0], cdf[:,1])
    except:
        pass

f.close()
hdf.close()
   
ax1.legend(loc='lower left')
ax1.set_xlabel(r"$r[\mathrm{pc}]$")
ax1.set_ylabel(r"$\rho[M_{\odot}\mathrm{pc}^{-3}]$")
#ax2.legend(loc='lower right')
ax2.set_xlabel(r"$r[\mathrm{pc}]$")
ax2.set_ylabel(r"$M[M_{\odot}]$")
#plt.tight_layout() 
pt = "-".join(sys.argv[1:])+"-"+filename[:-5]
pt = pt.replace('_', '-')
fig.suptitle(pt)
equ = (r'$\rho(r) = \frac{\rho_c}{\left(\frac{r}{r_c}\right)^{\eta_1} + \left(\frac{r}{r_c}\right)^{\eta_2}}$' if pw else
       r'$\rho(r) = \frac{\rho_c}{\left(\frac{r}{r_c}\right)^{\eta_1} \left(1+\frac{r}{r_c}\right)^{\eta_2-\eta_1}}$')
ax1.text(0.8, 0.8,equ, ha='center', va='center', fontsize=20, transform=ax1.transAxes)
plt.savefig(os.path.join(pwd, 'clusters-density', "-".join(sys.argv[1:])+'-'+filename[:-5]+('-densities-pw.pdf' if pw else '-densities.pdf')))
#plt.close()   