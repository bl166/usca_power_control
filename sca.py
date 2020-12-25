#!/usr/bin/env python3

# Copyright (C) 2018-2020 Bho Matthiesen, Karl-Ludwig Besser
# 
# This program is used in the article:
# 
# Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and
# Merouane Debbah, "A Globally Optimal Energy-Efficient Power Control Framework
# and its Efficient Implementation in Wireless Interference Networks,"
# submitted to IEEE Transactions on Signal Processing
# 
# License:
# This program is licensed under the GPLv2 license. If you in any way use this
# code for research that results in publications, please cite our original
# article listed above.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy as np
import cvxpy as cp
from utils import *

it_count1  = []

def SCA(h, mu, Pc, Pmax, pt = None, MaxIter = 10000, SolverMaxIter = 1000,
         parm_alpha = 1e-8, parm_beta = 0.01, 
         RelTolFun = 1e-12, RelTolVal = 1e-12, InnerOpt='sgd'):
    """
    InnerOpt: sgd, cvx, ...
    Init    : last, full, ...
    """

    if pt is None:
        pt = np.full(h.shape[-1], Pmax)

    def f(p): # verified
        s = h * p # (4,4) * (4,) --> (4,4)
        
        direct = np.diag(s)
        ifn = 1 + np.sum(s, axis=-1) - direct
        rates = np.log(1+direct/ifn)
        ee = rates / (mu * p + Pc)

        return np.sum(ee)

    def gradr(p): # verified
        s = h * p
        tmp = 1 + np.sum(s, axis=-1) # 1 + sum beta + a
        tmp2 = tmp - np.diag(s)
        fac = np.diag(s) / (tmp * tmp2)
        
        grad = h.copy()      
        grad = -(fac * grad.T).T
    
        grad[np.diag_indices_from(grad)] = 1/tmp * np.diag(h)#tmp2/(tmp*tmp2) * np.diag(h)

        return grad

    def gradf(p): # verified
        tmp = 1 / (mu * p + Pc)
        gr = gradr(p)
        
        t1 = np.sum((gr.T * tmp).T, axis=0)

        s = h * p
        direct = np.diag(s)
        ifn = 1 + np.sum(s, axis=-1) - direct
        rates = np.log(1+direct/ifn)
        
        t2 = mu * rates * tmp**2

        return t1 - t2

    # gradient step parameter
    if InnerOpt.lower() == "sgd":
        inner_opt = inner_optim_sgd
        kargws = {"learning_rate":.1}
        
    elif InnerOpt.lower() == "cvx":
        inner_opt = inner_optim_cvx
        kargws = {"verbose": False}
        
    else:
        raise NotImplemented
        
        
    OBJ, PT = [f(pt)],[pt]
    cnt = 0
    while True:
                
        cnt += 1
        pvar = inner_opt(pt, h, mu, Pc, Pmax, eps=1e-8, max_iters=SolverMaxIter, **kargws)
                
        # calculate gradient step
        Bpt = pvar - pt
        gamma = 1

        old_obj = f(pt)
        old_pt = pt
        while f(pt + gamma * Bpt) < old_obj + parm_alpha * gamma * gradf(pt) @ Bpt:
            gamma *= parm_beta

#         pt = np.clip(gamma * Bpt + pt, 0, Pmax)
        pt += gamma * Bpt
        obj = f(pt)
        
        OBJ.append(obj)
        PT.append(pt)

        with np.errstate(divide='ignore'):
            if abs(obj/old_obj - 1) < RelTolFun and np.linalg.norm(pt-old_pt, np.inf) / np.linalg.norm(pt, np.inf) < RelTolVal:
                break
        
        if cnt > MaxIter:
#             print('MaxIter')
            break

    it_count1.append(cnt)
    return (OBJ, PT)



def SCA_randinit(num, h, mu, Pc):
    dim = h.shape[-1]

    obj = -np.inf
    for cnt in range(num):
        pt = np.random.rand(dim)
        obj1, popt1 = SCA(h, mu, Pc, pt = pt)

        if obj1 > obj:
            obj = obj1
            popt = popt1

    return (obj, popt)


if __name__ == "__main__":
#     import progressbar as pb
    from tqdm import tqdm
    from tqdm.auto import trange
    import itertools as it
    import h5py
    
    num_ue = 4

    dfn = '../../data/wsee%d-processed.h5'%num_ue
    mu = 4
    Pc = 1

    f = h5py.File(dfn, 'r')
    dset = f

    Plin = 10**(np.asarray(dset['input/PdB'][...]/10))

    try:
        obj = dset.create_dataset('SCA', shape = dset['objval'].shape, fillvalue = np.nan, dtype = dset['objval'].dtype)
        popt = dset.create_dataset('SCA_xopt', shape = dset['objval'].shape + (4,), fillvalue = np.nan, dtype = dset['objval'].dtype)
        obj2 = dset.create_dataset('SCAmax', shape = dset['objval'].shape, fillvalue = np.nan, dtype = dset['objval'].dtype)
        popt2 = dset.create_dataset('SCAmax_xopt', shape = dset['objval'].shape + (4,), fillvalue = np.nan, dtype = dset['objval'].dtype)
    except:# RuntimeError:
        print('IN EXCEPTION!!')
        obj = dset['SCA']
        popt = dset['SCA_xopt']
        obj2 = dset['SCAmax']
        popt2 = dset['SCAmax_xopt']
        
    obj_ = np.full_like(obj, np.nan)
    popt_ =  np.full_like(popt, np.nan)
    obj2_ = np.full_like(obj2, np.nan)
    popt2_ =  np.full_like(popt2, np.nan)
    
#     for cidx, pidx in pb.progressbar(it.product(range(11907,20000), range(obj.shape[1])), widget = pb.ETA, max_value = (20000-11907)*obj.shape[1]):
#     for cidx, pidx in tqdm(it.product(range(11907,12200), range(obj.shape[1])), total = (12200-11907)*obj.shape[1]):
    for cidx in trange(2,desc='channel index:'):
        for pidx in trange(obj.shape[1], desc='power index:'):

            # cidx- channel matrix realization ~xxx
            # pidx- power limit realization ~51

            h = np.asarray(dset['input/channel_to_noise_matched'][cidx], dtype = float)
            p = Plin[pidx]

            if pidx == 0:
                pt = None

            if pt is not None:
                o1,p1 = SCA(h, mu, Pc, Pmax = p, pt = pt, 
                            MaxIter = 10000, parm_alpha = 1e-8, parm_beta = 0.01, RelTolFun = 1e-12, RelTolVal = 1e-12)
            else:
                o1 = -np.inf

            o2,p2 = SCA(h, mu, Pc, Pmax = p, pt = np.full(num_ue, p), 
                        MaxIter = 10000, parm_alpha = 1e-8, parm_beta = 0.01, RelTolFun = 1e-12, RelTolVal = 1e-12)

            obj2_[cidx,pidx] = o2 # f(pt)
            popt2_[cidx,pidx,:] = p2

            if o1 > o2:
                obj_[cidx,pidx] = o1
                pt = p1
            else:
                obj_[cidx,pidx] = o2
                pt = p2

            popt_[cidx,pidx,:] = pt

            if pidx > 51:
                print(cidx, pidx)
                break

#         if cidx > 51
#             print(cidx, pidx)
#             break

    print(obj_[~np.isnan(obj_)])




    f.close()
