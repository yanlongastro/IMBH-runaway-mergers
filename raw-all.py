# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 22:41:40 2019

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

##### Read data #####
pwd = os.getcwd()
path = os.path.join(pwd, *sys.argv[1:])
print(path)
nfiles=len(glob.glob1(path, "snapshot*.hdf5"))
nfiles = 38
filename = 'Clusters_'+str(nfiles-1).zfill(3)+'.hdf5'
f = h5py.File(os.path.join(path, filename), 'r')
n_cls = len(f.keys())
dat = np.loadtxt(os.path.join(path, 'bound_'+str(nfiles-1).zfill(3)+'-fixed.dat'))
if dat.ndim == 1:
    dat = dat.reshape((1, dat.shape[0]))
    
outfilename = "-".join(sys.argv[1:])+'-'+filename[:-5]+'-raw.hdf5'
hdf = h5py.File(os.path.join(pwd, 'clusters-density', outfilename), 'w')

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
    raw_cdf = r_m = np.array(sorted(r_m, key=operator.itemgetter(0)))
    raw_cdf = np.array(raw_cdf)[1:]
    raw_cdf[:,1] = np.cumsum(raw_cdf[:,1])
    
    smooth_cdf = cls_py.cdf_smooth(raw_cdf)
    #cdf = cls_py.cdf_smooth_cheby(raw_cdf[5:])
    sample_cdf = cls_py.cdf_sample(r_m)
    sample_pdf = cls_py.pdf_sample(r_m)
    smooth_pdf = cls_py.pdf_cal(smooth_cdf)
    smooth_pdf = smooth_pdf[smooth_pdf[:, 1]>0, :]
        
    clst_hdf = hdf.create_group(clst)
    clst_hdf.create_dataset('center', data = center)
    clst_hdf.create_dataset('m_tot', data = m_tot)
    clst_hdf.create_dataset('n_particle', data = n_particle)
    clst_hdf.create_dataset('r_h', data = r_h)
    
    clst_hdf.create_dataset('raw_cdf', data = raw_cdf)
    clst_hdf.create_dataset('smooth_cdf', data = smooth_cdf)
    clst_hdf.create_dataset('smooth_pdf', data = smooth_pdf)
    clst_hdf.create_dataset('sample_cdf', data = sample_cdf)
    clst_hdf.create_dataset('sample_pdf', data = sample_pdf)

f.close()
hdf.close()