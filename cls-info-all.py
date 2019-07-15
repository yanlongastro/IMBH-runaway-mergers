# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:58:49 2019

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
import gw_nu


cwd = os.getcwd()
#pwd = os.path.dirname(os.getcwd())
path = os.path.join(cwd, 'clusters-density-2')
files = glob.glob1(path, '*-raw.hdf5')
minusn = 9

hdf = h5py.File('cls-info-all.hdf5', 'w')
for fs in files:
    filename = os.path.join(path, fs)
    try:
        f = h5py.File(filename, 'r')
    except:
        pass
        continue
    n_cls = len(f.keys())
    print('\n', fs, '\n')
    hdf_cloud = hdf.create_group(fs[:-minusn])
    for i in range(n_cls):
        clst = list(f.keys())[i]
        hdf_cls = hdf_cloud.create_group(clst)
        print('\n', clst, '\n')
        m_tot = f[clst]['m_tot'][()]
        cdf = f[clst]['sample_cdf'][()]
        r_h = f[clst]['r_h'][()]
        for j in range(len(cdf)):
            if cdf[j][1] > cdf[-1, 1]/2.:
                break
        r_h_s = cdf[j, 0]
        #plt.loglog(cdf[:,0], cdf[:,1])
        
        models = ['double_power_free', 'double_power_a_fixed', 'double_power_a_b_fixed', 'single_power', 'piecewise']
        for model in models:
            if model == 'double_power_free':
                a_con, b_con, rc_con = 0, 0, 0
                func_cdf = cls_py.func_cdf
                func_cdf_inv = cls_py.func_cdf_inv
                tdfc_func = gw_nu.tdfc_phy
                cdf_s = cdf
            if model == 'double_power_a_fixed':
                a_con, b_con, rc_con = 0, 0, 0
                func_cdf = cls_py.func_cdf
                func_cdf_inv = cls_py.func_cdf_inv
                tdfc_func = gw_nu.tdfc_phy
                cdf_s = cdf
            if model == 'double_power_a_b_fixed':
                a_con, b_con, rc_con = 0, 3.35, 0
                func_cdf = cls_py.func_cdf
                func_cdf_inv = cls_py.func_cdf_inv
                tdfc_func = gw_nu.tdfc_phy
                cdf_s = cdf
            if model == 'single_power':
                a_con, b_con, rc_con = 0, 0, r_h
                func_cdf = cls_py.func_cdf
                func_cdf_inv = cls_py.func_cdf_inv
                tdfc_func = gw_nu.tdfc_phy
                cdf_s = cdf[:j+1]
            if model == 'piecewise':
                a_con, b_con, rc_con = 0, 0, 0
                func_cdf = cls_py.func_cdf_pw
                func_cdf_inv = cls_py.func_cdf_pw_inv
                tdfc_func = gw_nu.tdfc_pw_phy
                cdf_s = cdf
            
            try:
                fits, params = cls_py.fit_cdf_general_scipy(cdf_s, cdf_s[j,0], cdf_s[j,1], cdf_s[-1,0], cdf_s[-1,1], model, a_con, b_con, rc_con)
            except:
                pass
                continue
#            fits, params = cls_py.fit_cdf_general_scipy(cdf_s, cdf_s[j,0], cdf_s[j,1], cdf_s[-1,0], cdf_s[-1,1], model, a_con, b_con, rc_con)
            if params[2]<0:
                print('Fitting declined.')
                continue
            print(params)
            fits_pdf = np.zeros((cdf_s.shape[0]-1, cdf_s.shape[1]))
            fits_pdf[:,0] = cdf_s[:-1,0]
            fits_pdf[:,1] = np.diff(fits[:,1])/np.diff(fits[:,0])/4/np.pi/fits[:-1,0]**2
            mrc_fit = np.exp(func_cdf(np.log(params[3]), np.log(params[0]), params[1], params[2], np.log(params[3])))
            m_tot_fit = np.exp(func_cdf(np.log(1e8), np.log(params[0]), params[1], params[2], np.log(params[3])))
            r_h_fit = func_cdf_inv(0.5, np.log(params[0]), params[1], params[2], np.log(params[3]))
            t_dfc = tdfc_func(*params)
            
            hdf_model = hdf_cls.create_group(model)
            hdf_model.create_dataset('params', data = params)
            hdf_model.create_dataset('mrc_fit', data = mrc_fit)
            hdf_model.create_dataset('m_tot_fit', data = m_tot_fit)
            hdf_model.create_dataset('r_h_fit', data = r_h_fit)
            hdf_model.create_dataset('t_dfc', data=t_dfc)
            hdf_model.create_dataset('fits_cdf', data=fits)
            hdf_model.create_dataset('fits_pdf', data=fits_pdf)
            
            #plt.loglog(fits[:,0], fits[:,1])

hdf.close()     