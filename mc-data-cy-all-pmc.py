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
import mc_cy_pmc

import argparse
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--savfig', action='store_true', help='Save the figures of evulution')
#parser.add_argument('--pw', action='store_true', help='Use the piecewise density model')
args = parser.parse_args()
savfig = args.savfig
#pw = args.pw


t0 = time.time()
cwd = os.getcwd()
pwd = os.path.dirname(os.getcwd())
path = os.path.join(pwd, 'clusters-density-2')

#if pw==True:
#    files = glob.glob1(path, '*-info-pw.hdf5')
#    minusn = 13
#else:
files = glob.glob1(path, '*-raw.hdf5')
minusn = 9

m_s = 0.376176

npart_max = 700

hdf = h5py.File('mc-data-all.hdf5', 'a')
info = h5py.File('../cls-info-all.hdf5', 'r')
cloud=[]
#try:
#    os.remove('data.txt')
#except:
#    pass

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
    #data = np.zeros((n_cls, 14))
    if fs[:-minusn] not in list(hdf.keys()):
        hdf_cloud = hdf.create_group(fs[:-minusn])
    else:
        hdf_cloud = hdf[fs[:-minusn]]
    info_cloud = info[fs[:-minusn]]
    for i in range(0, n_cls):
        clst = list(f.keys())[i]
        print('\n', clst, '\n')
        mass_cls = f[clst]['m_tot'][()]
#        if np.isfinite(f[clst]['params']).all()==False:
#            continue
        mass_cdf = np.array(f[clst]['raw_cdf'])
        if info_cloud[clst]['n_particle'][()] > npart_max:
            print('To massive, skip at this time...')
            continue
        if 'double_power_cons' in list(info_cloud[clst].keys()):
            params = info_cloud[clst]['double_power_cons']['params'][()]
            print(params)
        else:
            continue
        #mass_cdf = np.array(f[clst]['mc_cdf'])
        mass_cdf = np.transpose(mass_cdf)
        mass_cdf[0] = mass_cdf[0]/1.0e3     #pc to kpc
        mass_cdf_max = mass_cdf[1][-1]
        mass_cdf[1] = mass_cdf[1]/mass_cdf_max
        mass_cdf = np.transpose(mass_cdf)
        star_catalog = mc_cy.sample_partial(mass_cls, mass_cdf)
        #r_h = f[clst]['params0'][-1]/1.0e3  #pc to kpc
        r_h = f[clst]['r_h'][()]/1.0e3  #pc to kpc
        t_rlx = mc_cy.t_rlx_py(r_h, mass_cls/2.0, m_s)
        #r_c = f[clst]['params'][3]/1.0e3
        #m_rc = f[clst]['params'][-3]
        #t_dfc = 3.3 * mc_cy.t_rlx_py(r_c, m_rc, 100)* np.log(0.1* m_rc/100) / np.log(0.1* mass_cls/2./m_s)
        print("Sample:", time.time() - t0)
        
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
        #tm = mc_pmc.evolve(star_catalog_sorted, times[-1], 1e8, 1, 4 ,5)
        #tm_fit = interp1d(tm[:,0], tm[:,1])
        #mass_growth = tm_fit(times)
        print("Evolve:", time.time() - t0)
        
        plt.loglog(times, mass_growth)
        
        if np.isnan(mass_growth).any()==True or (mass_growth<=0).any()==True:
            continue
    
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

        # check if we need to create new sets in hdf5 file
        if clst not in list(hdf_cloud.keys()):
            hdf_cls = hdf_cloud.create_group(clst)
            if 'mass_growth_rate' not in list(hdf_cloud[clst].keys()):
                print("t_e/Myr:", t_e)
                print("M_c(t_e)/M_sun", m_c_te)
                print("M_c(t_e+3)/M_sun", m_c_te3)
                hdf_cls.create_dataset('t_e', data = t_e)
                hdf_cls.create_dataset('m_c_te', data = m_c_te)
                hdf_cls.create_dataset('m_c_te3', data = m_c_te3)
                
                hdf_cls.create_dataset('t_rlx', data = t_rlx)
                hdf_cls.create_dataset('mass_growth', data=np.array(list(zip(times_, mass_growth_))))
                hdf_cls.create_dataset('mass_growth_rate', data=np.array(list(zip(times_[:-1], mass_growth_rate))))
        else:
            hdf_cls = hdf_cloud[clst]
            print('No PMC condition is already done.')
        if 'mass_growth_rate_pmc' not in list(hdf_cls.keys()):
            pass
        else:
            print('PMC condition is already done.')
            continue
        
        star_catalog_sorted = sorted(star_catalog, key = operator.itemgetter(2))
        star_catalog_sorted = np.array(star_catalog_sorted)
        print("Sort:", time.time() - t0)
        
        n_step = 10000
        times = np.logspace(-1, 5, n_step)
        #mass_growth = mc_cy.evolve(star_catalog_sorted, times)
        #mass_growth = np.array(mass_growth)
        tm = mc_cy_pmc.evolve(star_catalog_sorted, times[-1]*1e6, *params)
        # check if the PMC evolution fails
        if len(tm)<10 or np.isnan(tm).any()==True or (tm<0).any()==True:
            continue
        tm_fit_log = interp1d(np.log10(tm[:,0]/1e6), np.log10(tm[:,1]), fill_value="extrapolate" )
        tm_fit = lambda x: 10**tm_fit_log(np.log10(x))
        mass_growth = tm_fit(times)
        
        print("Evolve -- PMC:", time.time() - t0)
        
        plt.loglog(times, mass_growth)
        plt.show()
        
    
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
        t_e_pmc = np.exp(m)
        m_c_te_pmc = np.exp(intp_mg(m))*3e6
        m_c_te3_pmc = np.exp(intp_mg(np.log(np.exp(m)+3)))*3e6
        mass_growth_pmc_ = mass_growth_
        mass_growth_rate_pmc = mass_growth_rate
        print("t_e_pmc/Myr:", t_e_pmc)
        print("M_c_pmc(t_e)/M_sun", m_c_te_pmc)
        print("M_c_pmc(t_e+3)/M_sun", m_c_te3_pmc)
        
        
        
        if savfig == True:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[8,10], tight_layout=True)
            ax1.loglog(times, mass_growth, label='Data')
            ax1.loglog(times_, np.exp(y), label='Fit')
            ax1.loglog(np.repeat(t_rlx, len(y)), np.exp(y), '--', label=r'$t_{\rm rlx} = %f {\rm Myr}$'%(t_rlx))
            #ax1.loglog(np.repeat(t_dfc, len(y)), np.exp(y), ':', label=r'$t_{\rm df,c} = %f {\rm Myr}$'%(t_dfc))
            ax1.loglog(times, np.repeat(mass_cls, len(times)), '-.', label=r'$M_{\rm cls} = %f M_{\odot}$'%(mass_cls))
            #ax1.loglog(times, np.repeat(m_rc, len(times)), '-.', label=r'$M(r_{\rm c}) = %f M_{\odot}$'%(m_rc))
            ax1.set_xlim(0.1, 1e4)
            ax1.set_xlabel(r"$t[\mathrm{Myr}]$")
            ax1.set_ylabel(r"$\mathcal{M}[M_{\odot}]$")
            ax1.legend(loc='lower right')
    
            ax2.loglog(times_gr, mass_growth_rate, label='Growth rate')
            ax2.loglog(times, mass_growth/3e6, label='Fueling rate')
            ax2.loglog(t_e, m_c_te/3.0e6, '*', label=r'$t_{\rm e}=%f {\rm Myr},~M_{\rm c}(t_{\rm e})=%f M_{\odot}$'%(t_e, m_c_te))
            ax2.loglog(t_e+3, m_c_te3/3.0e6, 'o', label=r'$M_{\rm c}(t_{\rm e}+3{\rm Myr})=%f M_{\odot}$'%(m_c_te3))
            ax2.loglog(np.repeat(t_rlx, len(y)), np.exp(y)/3.0e6, '--', label=r'$t_{\rm rlx} = %f {\rm Myr}$'%(t_rlx))
            #ax2.loglog(np.repeat(t_dfc, len(y)), np.exp(y)/3.0e6, ':', label=r'$t_{\rm df,c} = %f {\rm Myr}$'%(t_dfc))
            ax2.set_xlim(0.1, 1e4)
            ax2.set_ylim(1e-6)
            ax2.set_xlabel(r"$t[\mathrm{Myr}]$")
            ax2.set_ylabel(r"$\dot{\mathcal{M}}[M_{\odot}/{\rm Myr}]$")
            ax2.legend(loc='lower left')
            
            pt = fs[:-minusn]+"-"+clst
            pt = pt.replace('_', '-')
            fig.suptitle(pt)
            plt.savefig(os.path.join(os.getcwd(), 'plots', pt+'-mc'+'.pdf'))
            plt.close()
            
        cloud.append(fs[:-minusn])
        
        
        #data[i][0] = list(f[clst]['params0'])[0]
        #data[i][1] = list(f[clst]['params0'])[-1]
        #data[i][2:9] = list(f[clst]['params'])
        #data[i][9] = t_e
        #data[i][10] = m_c_te
        #data[i][11] = m_c_te3
        #data[i][12] = t_rlx
        #data[i][13] = t_dfc
        

        
        hdf_cls.create_dataset('t_e_pmc', data = t_e_pmc)
        hdf_cls.create_dataset('m_c_te_pmc', data = m_c_te_pmc)
        hdf_cls.create_dataset('m_c_te3_pmc', data = m_c_te3_pmc)
        
        
        hdf_cls.create_dataset('mass_growth_pmc', data=np.array(list(zip(times_, mass_growth_pmc_))))
        hdf_cls.create_dataset('mass_growth_rate_pmc', data=np.array(list(zip(times_[:-1], mass_growth_rate_pmc))))
        print("Done:", time.time() - t0)
        
        hdf.flush()
    f.close()
#    dt=open('data.txt','ab')
#    np.savetxt(dt, data)
#    dt.close()
hdf.close()
#data = np.loadtxt('data.txt')
#names = ['mcls', 'rh', 'rhoc', 'eta1', 'eta2', 'rc', 'mrc-fit', 'mcls-fit', 'rh-fit', 'te', 'mte', 'mte3', 't-rlx', 't-dfc']
#df = pd.DataFrame(data, columns=names)
#df['cloud'] = cloud
#df.to_csv('dataframe.csv', encoding='utf-8')
    
