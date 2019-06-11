# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:48:22 2019

Main monte-carlo code

@author: Yanlong
"""
import numpy as np
import glob
import sys
import os
import time
import h5py
from scipy.interpolate import interp1d
from scipy import signal
import operator
import pandas  as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib
matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True

import mc_cy

import argparse
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--savfig', action='store_true', help='Save the figures of evulution')
parser.add_argument('--pw', action='store_true', help='Use the piecewise density model')
args = parser.parse_args()
savfig = args.savfig
pw = args.pw


t0 = time.time()
cwd = os.getcwd()
pwd = os.path.dirname(os.getcwd())
path = os.path.join(pwd, 'clusters-density')

if pw==True:
    files = glob.glob1(path, '*-info-pw.hdf5')
    minusn = 13
else:
    files = glob.glob1(path, '*-info.hdf5')
    minusn = 10
#filename = os.path.join(path, sys.argv[1])
#f = h5py.File(filename, 'r')
#n_cls = len(f.keys())
#print("Read file:", time.time() - t0)
#data = np.zeros((n_cls, 11))
m_s = 0.376176

hdf = h5py.File('mc-data-all'+('-pw' if pw else '')+'.hdf5', 'w')
cloud=[]
try:
    os.remove('data'+('-pw' if pw else '')+'.txt')
except:
    pass

for fs in files:
    filename = os.path.join(path, fs)
    try:
        f = h5py.File(filename, 'r')
    except:
        pass
        continue
    n_cls = len(f.keys())
    print('\n', fs, '\n')
    print("Read file:", time.time() - t0)
    data = np.zeros((n_cls, 14))
    hdf_cloud = hdf.create_group(fs[:-minusn])
    for i in range(n_cls):
        clst = list(f.keys())[i]
        print('\n', clst, '\n')
        mass_cls = f[clst]['params'][-2]
        if np.isfinite(f[clst]['params']).all()==False:
            continue
        mass_cdf = np.array(f[clst]['raw_cdf'])
        #mass_cdf = np.array(f[clst]['mc_cdf'])
        mass_cdf = np.transpose(mass_cdf)
        mass_cdf[0] = mass_cdf[0]/1.0e3     #pc to kpc
        mass_cdf_max = mass_cdf[1][-1]
        mass_cdf[1] = mass_cdf[1]/mass_cdf_max
        mass_cdf = np.transpose(mass_cdf)
        star_catalog = mc_cy.sample_partial(mass_cls, mass_cdf)
        #r_h = f[clst]['params0'][-1]/1.0e3  #pc to kpc
        r_h = f[clst]['params'][-1]/1.0e3  #pc to kpc
        t_rlx = mc_cy.t_rlx_py(r_h, mass_cls/2.0, m_s)
        r_c = f[clst]['params'][3]/1.0e3
        m_rc = f[clst]['params'][-3]
        t_dfc = 3.3 * mc_cy.t_rlx_py(r_c, m_rc, 100)* np.log(0.1* m_rc/100) / np.log(0.1* mass_cls/2./m_s)
        print("Sample:", time.time() - t0)
        
        m_cut_l = 0.5
        star_catalog = np.array(star_catalog)
        #star_catalog_sorted = [row for row in star_catalog if row[0] > m_cut_l and row[3] > row[2]]
        print("Select:", time.time() - t0)
        
        star_catalog_sorted = sorted(star_catalog, key = operator.itemgetter(2))
        star_catalog_sorted = np.array(star_catalog_sorted)
        print("Sort:", time.time() - t0)
        
        n_step = 10000
        times = np.logspace(-1, 5, n_step)
        mass_growth = mc_cy.evolve(star_catalog_sorted, times)
        mass_growth = np.array(mass_growth)
        print("Evolve:", time.time() - t0)
        
    
        x = np.log(times)
        y = np.log(mass_growth)
        #print(y[:5])
        #print(all(np.isfinite(y)))
        x = x[np.isfinite(y)]
        times_ = times[np.isfinite(y)]
        mass_growth_ = mass_growth[np.isfinite(y)]
        y = y[np.isfinite(y)]
        cheby = np.polynomial.Chebyshev.fit(x, y, 10)
        #y = signal.savgol_filter(y, len(x)//8*2+1, 3)
        y = cheby(x)
        
        mass_growth_rate  = np.diff(y)/np.diff(x) * np.exp(y[:-1]) / np.exp(x[:-1])/1e6
        #mass_growth_rate  = cheby.deriv()(x[:-1]) * np.exp(y[:-1]) / np.exp(x[:-1])/1e6
        times_gr = times_[:-1][mass_growth_rate>0]
        mass_growth_rate = mass_growth_rate[mass_growth_rate>0]
        
        #np.savetxt('mg.txt', np.transpose([times, mass_growth]), fmt='%e')
        #np.savetxt('mgr.txt', np.transpose([times_[:-1], mass_growth_rate]), fmt='%e')
        
        intp_mgr = interp1d(np.log(times_gr), np.log(mass_growth_rate), fill_value='extrapolate')
        intp_mg = interp1d(np.log(times_), np.log(mass_growth_/3e6), fill_value='extrapolate')
        l = -4
        h = 4
        while h-l > 0.001:
            m = (h+l)/2
            if intp_mgr(m) < intp_mg(m):
                h = m
            else:
                l = m
        t_e = np.exp(m)
        m_c_te = np.exp(intp_mg(m))*3e6
        m_c_te3 = np.exp(intp_mg(np.log(np.exp(m)+3)))*3e6
        print("t_e/Myr:", t_e)
        print("M_c(t_e)/M_sun", m_c_te)
        print("M_c(t_e+3)/M_sun", m_c_te3)
        
        if savfig == True:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[8,10], tight_layout=True)
            ax1.loglog(times, mass_growth, label='Data')
            ax1.loglog(times_, np.exp(y), label='Fit')
            ax1.loglog(np.repeat(t_rlx, len(y)), np.exp(y), '--', label=r'$t_{\rm rlx} = %f {\rm Myr}$'%(t_rlx))
            ax1.loglog(np.repeat(t_dfc, len(y)), np.exp(y), ':', label=r'$t_{\rm df,c} = %f {\rm Myr}$'%(t_dfc))
            ax1.loglog(times, np.repeat(mass_cls, len(times)), '-.', label=r'$M_{\rm cls} = %f M_{\odot}$'%(mass_cls))
            ax1.loglog(times, np.repeat(m_rc, len(times)), '-.', label=r'$M(r_{\rm c}) = %f M_{\odot}$'%(m_rc))
            ax1.set_xlim(0.1, 1e4)
            ax1.set_xlabel(r"$t[\mathrm{Myr}]$")
            ax1.set_ylabel(r"$\mathcal{M}[M_{\odot}]$")
            ax1.legend(loc='lower right')
    
            ax2.loglog(times_gr, mass_growth_rate, label='Growth rate')
            ax2.loglog(times, mass_growth/3e6, label='Fueling rate')
            ax2.loglog(t_e, m_c_te/3.0e6, '*', label=r'$t_{\rm e}=%f {\rm Myr},~M_{\rm c}(t_{\rm e})=%f M_{\odot}$'%(t_e, m_c_te))
            ax2.loglog(t_e+3, m_c_te3/3.0e6, 'o', label=r'$M_{\rm c}(t_{\rm e}+3{\rm Myr})=%f M_{\odot}$'%(m_c_te3))
            ax2.loglog(np.repeat(t_rlx, len(y)), np.exp(y)/3.0e6, '--', label=r'$t_{\rm rlx} = %f {\rm Myr}$'%(t_rlx))
            ax2.loglog(np.repeat(t_dfc, len(y)), np.exp(y)/3.0e6, ':', label=r'$t_{\rm df,c} = %f {\rm Myr}$'%(t_dfc))
            ax2.set_xlim(0.1, 1e4)
            ax2.set_ylim(1e-6)
            ax2.set_xlabel(r"$t[\mathrm{Myr}]$")
            ax2.set_ylabel(r"$\dot{\mathcal{M}}[M_{\odot}/{\rm Myr}]$")
            ax2.legend(loc='lower left')
            
            pt = fs[:-minusn]+"-"+clst
            pt = pt.replace('_', '-')
            fig.suptitle(pt)
            plt.savefig(os.path.join(os.getcwd(), 'plots', pt+'-mc'+('-pw' if pw else '')+'.pdf'))
            plt.close()
            
        cloud.append(fs[:-minusn])
        
        
        data[i][0] = list(f[clst]['params0'])[0]
        data[i][1] = list(f[clst]['params0'])[-1]
        data[i][2:9] = list(f[clst]['params'])
        data[i][9] = t_e
        data[i][10] = m_c_te
        data[i][11] = m_c_te3
        data[i][12] = t_rlx
        data[i][13] = t_dfc
        
        hdf_cls = hdf_cloud.create_group(clst)
        hdf_cls.create_dataset('mc_data', data = data[i])
        
        #hdf_cls.create_dataset('mass_growth', data=np.array(list(zip(times_, mass_growth_))))
        #hdf_cls.create_dataset('mass_growth_rate', data=np.array(list(zip(times_[:-1], mass_growth_rate))))
        print("Done:", time.time() - t0)
    f.close()
    dt=open('data'+('-pw' if pw else '')+'.txt','ab')
    np.savetxt(dt, data)
    dt.close()
hdf.close()
data = np.loadtxt('data'+('-pw' if pw else '')+'.txt')
names = ['mcls', 'rh', 'rhoc', 'eta1', 'eta2', 'rc', 'mrc-fit', 'mcls-fit', 'rh-fit', 'te', 'mte', 'mte3', 't-rlx', 't-dfc']
df = pd.DataFrame(data, columns=names)
df['cloud'] = cloud
df.to_csv('dataframe'+('-pw' if pw else '')+'.csv', encoding='utf-8')
    
