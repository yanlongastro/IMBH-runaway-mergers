# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:48:22 2019

Main monte-carlo code

MPI task ref: 
    https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py
install h5py with mpi: 
    conda install -c conda-forge "h5py>=2.10=mpi*"
debug runs:
    mpirun --use-hwthread-cpus -np 4 xterm -e python <script>

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

from mpi4py import MPI

import mc_cy
import mc_cy_pmc


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object
num_workers = size-1


t0 = time.time()
m_s = 0.376176
npart_max = 400


hdf = h5py.File('mc-data-all-parallel.hdf5', 'a', driver='mpio', comm=MPI.COMM_WORLD)
    

if rank == 0:
    # Master
    print("Monte-Carlo simulations of runaway-merger IMBH formation\n")
    print("Master starting with %d workers" % num_workers)

    #cloud = list(hdf.keys())[0]
    clusters = []
    for cloud in list(hdf.keys()):
        for clst in list(hdf[cloud].keys()):
            clusters.append(cloud+'/'+clst)
    #hdf_cloud = hdf[cloud]
    #n_cls = len(hdf_cloud.keys())
    n_cls = len(clusters)

    closed_workers = 0
    task_index = 0
    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # assign some work
            if task_index < n_cls:
                # send info to workers
                #clst = list(hdf_cloud.keys())[task_index]
                cloud_clst = clusters[task_index]

                comm.send(cloud_clst, dest=source, tag=tags.START)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            results = data
            print("MC simulation from worker %d"%source, end=' ')
            if results==1:
                print("succeeded")
            elif results==-1:
                print("failed")
            elif results==2:
                print("done before")
            elif results==0:
                print("skipped")
        elif tag == tags.EXIT:
            print("Worker %d exited." % source)
            closed_workers += 1
    print("Master finishing")

#hdf.close()

else:
    #Workers
    while True:
        comm.send(None, dest=0, tag=tags.READY)

        cloud_clst = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
        if tag == tags.START:
            # MC simulation
            hdf_cls = hdf[cloud_clst]
            print('\n', 'Rank %d/%d: %s'%(rank, num_workers, cloud_clst))
            if hdf[cloud_clst]['PMC'][()] == True:
                print('PMC already done.')
                comm.send(2, dest=0, tag=tags.DONE)
                continue
            n_particle = hdf[cloud_clst]['n_particle'][()] 
            if n_particle > npart_max:
                print('Too massive, skip for now...')
                hdf_cls['PMC'][...] = False
                comm.send(0, dest=0, tag=tags.DONE)
                continue
            print("Particle numbers: %d"%n_particle)
            mass_cls = hdf[cloud_clst]['m_tot'][()]
            r_h = hdf[cloud_clst]['r_h'][()]
            mass_cdf = hdf[cloud_clst]['raw_cdf'][()]
            if 'params' in list(hdf[cloud_clst].keys()):
                params = hdf[cloud_clst]['params'][()]
            else:
                print('MJ not fitted for the cluster.')
                hdf_cls['PMC'][...] = True
                comm.send(-1, dest=0, tag=tags.DONE)
                continue

            print(params)
            mass_cdf = np.transpose(mass_cdf)
            mass_cdf[0] = mass_cdf[0]/1.0e3     #pc to kpc
            mass_cdf_max = mass_cdf[1][-1]
            mass_cdf[1] = mass_cdf[1]/mass_cdf_max
            mass_cdf = np.transpose(mass_cdf)
            star_catalog = mc_cy.sample_partial(mass_cls, mass_cdf)
            # r_h = f[clst]['r_h'][()]/1.0e3  #pc to kpc
            t_rlx = mc_cy.t_rlx_py(r_h, mass_cls/2.0, m_s)
            print("Sample:", time.time() - t0)
            star_catalog = np.array(star_catalog)
            star_catalog_sorted = sorted(star_catalog, key = operator.itemgetter(2))
            star_catalog_sorted = np.array(star_catalog_sorted)
            print("Sort:", time.time() - t0)

            n_step = 10000
            times = np.logspace(-1, 5, n_step)
            tm = mc_cy_pmc.evolve(star_catalog_sorted, times[-1]*1e6, *params)
            print("Evolve (PMC):", time.time() - t0)
            if len(tm)<10 or np.isnan(tm).any()==True or (tm<0).any()==True:
                print("PMC evolve failed")
                hdf_cls['PMC'][...] = True
                comm.send(-1, dest=0, tag=tags.DONE)
                continue
            else:
                tm_fit_log = interp1d(np.log10(tm[:,0]/1e6), np.log10(tm[:,1]), fill_value="extrapolate" )
                tm_fit = lambda x: 10**tm_fit_log(np.log10(x))
                mass_growth = tm_fit(times)

                if np.isnan(mass_growth).any()==True or (mass_growth<=0).any()==True:
                    print("Nan occured in mass_growth")
                    hdf_cls['no_PMC'][...] = True
                    comm.send(-1, dest=0, tag=tags.DONE)
                    continue

                x = np.log(times)
                y = np.log(mass_growth)
                x = x[np.isfinite(y)]
                times_ = times[np.isfinite(y)]
                mass_growth_ = mass_growth[np.isfinite(y)]
                y = y[np.isfinite(y)]
                cheby = np.polynomial.Chebyshev.fit(x, y, 10)
                y = cheby(x)
        
                mass_growth_rate  = np.diff(y)/np.diff(x) * np.exp(y[:-1]) / np.exp(x[:-1])/1e6
                times_gr = times_[:-1][mass_growth_rate>0]
                mass_growth_rate = mass_growth_rate[mass_growth_rate>0]

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

                # write data
                
                print("t_e/Myr:", t_e_pmc)
                print("M_c(t_e)/M_sun", m_c_te_pmc)
                print("M_c(t_e+3)/M_sun", m_c_te3_pmc)
                print('Time used: %.2f s'%(time.time()-t0))
                sys.stdout.flush()

                hdf_cls['t_e_pmc'][...] = t_e_pmc
                hdf_cls['m_c_te_pmc'][...] = m_c_te_pmc
                hdf_cls['m_c_te3_pmc'][...] = m_c_te3_pmc
                hdf_cls['PMC'][...] = True

                comm.send(1, dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)
    
#hdf.flush()
hdf.close()