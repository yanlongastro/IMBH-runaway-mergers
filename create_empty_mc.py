# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 00:22:08 2020

@author: yanlo
"""

import h5py
import os
import glob


cwd = os.getcwd()
pwd = os.path.dirname(os.getcwd())
path = os.path.join(pwd, 'clusters-density-2')
files = glob.glob1(path, '*-raw.hdf5')
minusn = 9
m_s = 0.376176
npart_max = 700


hdf = h5py.File('mc-data-all-parallel.hdf5', 'a')
info = h5py.File('../cls-info-all.hdf5', 'r')

# create all the data groups required.
for fs in files:
    filename = os.path.join(path, fs)
    try:
        f = h5py.File(filename, 'r')
    except:
        pass
        continue
    n_cls = len(f.keys())
    if fs[:-minusn] not in list(hdf.keys()):
        hdf_cloud = hdf.create_group(fs[:-minusn])
    else:
        hdf_cloud = hdf[fs[:-minusn]]
    cloud = fs[:-minusn]
    for clst in list(f.keys()):
        if clst not in list(hdf_cloud.keys()):
            hdf_cls = hdf_cloud.create_group(clst)
        else:
            hdf_cls = hdf_cloud[clst]
            
        if 'raw_cdf' not in list(hdf_cls.keys()):
            hdf_cls.create_dataset('m_tot', data=f[clst]['m_tot'][()])
            hdf_cls.create_dataset('r_h', data=f[clst]['r_h'][()]/1e3)
            hdf_cls.create_dataset('raw_cdf', data=f[clst]['raw_cdf'][()])
        
        if 'params' not in list(hdf_cls.keys()):
            if 'double_power_cons' in list(info[cloud][clst].keys()):
                hdf_cls.create_dataset('params', data=info[cloud][clst]['double_power_cons']['params'][()])

        if 't_rlx' not in list(hdf_cls.keys()):
            hdf_cls.create_dataset('t_e', data = 0.0)
            hdf_cls.create_dataset('m_c_te', data = 0.0)
            hdf_cls.create_dataset('m_c_te3', data = 0.0)
            hdf_cls.create_dataset('t_rlx', data = 0.0)
            #hdf_cls.create_dataset('mass_growth', data=np.array(list(zip(times_, mass_growth_))))
            #hdf_cls.create_dataset('mass_growth_rate', data=np.array(list(zip(times_[:-1], mass_growth_rate))))
        if 'no_PMC'not in list(hdf_cls.keys()):
            hdf_cls.create_dataset('no_PMC', data = False)
    print(cloud)
    hdf.flush()
    f.close()
# hdf.close()
info.close()

hdf.close()