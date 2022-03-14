import numpy as np
import h5py
from tqdm.auto import trange
import torch
from torch_geometric.utils import dense_to_sparse


def load_data(dpath):
    X, y = [],[]
    with h5py.File(dpath, "r") as handle:
        input = handle['input']
        Hs = input["channel_to_noise_matched"]
        Plin = 10**(np.asarray(input['PdB'][...], dtype=float)/10)
        xopt = handle['xopt'][...].astype(float)
        objval_opt = handle['objval'][...].astype(float) # or wsee 
        
        ns, nu, _ = Hs.shape # eg:(1000,4,4)

        for hidx in range(ns):
            edge_index, h = dense_to_sparse(torch.from_numpy(Hs[hidx].astype(float)))
            
            x1 = np.hstack([(h.reshape((-1,1))*Plin).T, # -->(h1p1, h2p1, h3p1, ...)
                             Plin.reshape(-1,1)])
            X.append( x1 )
            y.append( xopt[hidx] )
            
        cinfo = {'mu': input['PA inefficency'][...].item(),
                 'Pc': input['Pc'][...].item(),
                 'edge_index': edge_index,
                 'objval_opt': objval_opt} # or wsee 

    y = np.concatenate((y))
    X = np.concatenate((X))
    y = y[~np.any(np.isnan(X),-1)]     
    X = X[~np.any(np.isnan(X),-1)]
    
    return X, y, cinfo


def load_data_unsup(dpath, **kwargs):
    # parameters; ref: https://github.com/bmatthiesen/deep-EE-opt/blob/062093fde6b3c6edbb8aa83462165265deefce1a/src/globalOpt/run_wsee.py#L30
    extract_args = lambda a, k: a if not k in kwargs else kwargs[k]
    PdB = extract_args(np.array(range(-40,10+1,1)), 'PdB')   
    
    mu = extract_args(4.0, 'mu')
    Pc = extract_args(1.0, 'Pc')
    hxp = extract_args(True, 'hxp')
    num_stab = extract_args(0., 'num_stab')    
    
    Plin = 10**(PdB/10)
    if hxp:
        Ph = Plin
    else:
        Ph = torch.empty(Plin.shape).fill_(1)
    
    X = []
    with h5py.File(dpath, "r") as handle:
        Hs = handle['input']["channel_to_noise_matched"]
        
        ns, nu, _ = Hs.shape # eg:(1000,4,4)

        for hidx in range(ns):
            edge_index, h = dense_to_sparse(torch.from_numpy(Hs[hidx].astype(float)))
            h += num_stab
            
            x1 = np.hstack([(h.reshape((-1,1))*Ph).T, # -->(h1p1, h2p1, h3p1, ...)
                             Plin.reshape(-1,1)])
            X.append( x1 )
            
        cinfo = {'mu': mu,
                 'Pc': Pc,
                 'edge_index': edge_index} # or wsee 

    X = np.concatenate((X))
    X = X[~np.any(np.isnan(X),-1)]
    y = np.full([X.shape[0], nu], np.nan)
    
    return X, y, cinfo