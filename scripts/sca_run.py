
from tqdm.auto import tqdm
from copy import deepcopy
import itertools as it
import h5py
import glob
import gc
import os

import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

# SET YOUR PROJECT ROOT DIR HERE
PROJ_RT = "/root/deep-EE-opt-master/src/SCA/"
import sys; sys.path.append(PROJ_RT)

from sca import *
from utils import *
import dataset as ds

"""
classical SCA
"""
def classical_SCA(X, mu, Pc, init='last', **kwargs):
    
    kwargs['SolverMaxIter'] = kwargs['SolverMaxIter'] if 'SolverMaxIter' in kwargs else 100
    kwargs['MaxIter'] = kwargs['MaxIter'] if 'MaxIter' in kwargs else 100
    kwargs['parm_alpha'] = kwargs['parm_alpha'] if 'parm_alpha' in kwargs else 0.01
    kwargs['RelTolFun'] = kwargs['RelTolFun'] if 'RelTolFun' in kwargs else 1e-12
    kwargs['RelTolVal'] = kwargs['RelTolVal'] if 'RelTolVal' in kwargs else 1e-12
    kwargs['InnerOpt'] = kwargs['InnerOpt'] if 'InnerOpt' in kwargs else 'sgd'
    
    n, d = X.shape
    nue = int(d**.5)
    assert nue**2+nue+1 == d

    xx = None
    gc.collect()
    xx = deepcopy(X)
    
    p0_valid, hs_valid, pmax_valid = [item.cpu().numpy() for item in decouple_input(xx, nue)]

    eeval, ppred = [],[]
    with tqdm(total=n)  as pbar:       
        for ni in range(n):
            if ni==0 or pmax_valid[ni-1] > pmax_valid[ni] or init in ['full','max'] :
                p0 = p0_valid[ni]
            elif init=='last':
                p0 = P[-1]
            else:
                raise ValueError('Initialization error!')
                
            h = hs_valid[ni]
            O, P = SCA(h, mu = mu, Pc = Pc, Pmax = pmax_valid[ni], pt = p0, **kwargs)
            
            if np.isnan(O[-1]):
                P[-1] = p0_valid[ni]
                O[-1] = f_wsee(P[-1], h, mu, Pc)

            if np.isnan(O[-1]): # first one in a roll?
                raise

            eeval.append(O[-1])
            ppred.append(P[-1])
            
            pbar.update(1)
            pbar.set_description("WSEE %.6f" % np.mean(eeval))
            
        #     print(O[-1], P[-1])
    return np.array(eeval), np.array(ppred)


if __name__ == '__main__':
    
    #num_ue=10
    init = 'last'
    device = torch.device('cpu')

    dfns = sorted(glob.iglob(PROJ_RT+r'../../data_my/test-channels-HataUrban-noSF-6.h5'))
    #dfns = sorted(glob.iglob(PROJ_RT+r'../../data/results_hata*.h5'))

    for dfn in dfns:
        num_ue = int(dfn.split('-')[-1].split('.')[0])
        print(dfn, num_ue)

        X, y, cinfo = None, None, None
        gc.collect()

        #--- data---
        X, y, cinfo = ds.load_data_unsup(dfn)
        print(X.shape, y.shape)
        assert int((X.shape[1]-1)**.5)==num_ue

        # add initial pt (max)
        attach_pt = lambda x: torch.from_numpy(np.hstack((init_p(x[:,-1], num_ue, method='full'), x))).float().to(device)
        X_ = attach_pt(X)

        # move channel info to device
        dict_to_device = lambda x,dev: {k:v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        cinfo = dict_to_device(cinfo, device)
        
        save_to = dfn.split('.h5')[0]+'_fSCA+obj+p.npz'

        if not os.path.exists(save_to):
            
            ni = 500#len(inner_layers)
            no = 1000#len(outer_layers)

            args = {'mu':cinfo['mu'], 'Pc':cinfo['Pc'], 'init':init, 'SolverMaxIter':ni, 'MaxIter':no}

            wsee_sca_full, p_pred_full = classical_SCA(X_, **args)
            print(wsee_sca_full.shape, p_pred_full.shape)

            np.savez(dfn.split('.h5')[0]+'_fSCA+obj+p.npz', wsee = wsee_sca_full, p = p_pred_full)
                        