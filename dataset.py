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
        
        cinfo = {'mu': input['PA inefficency'][...].item(),
                 'Pc': input['Pc'][...].item() }
        
        ns, nu, _ = Hs.shape # eg:(1000,4,4)

        for hidx in range(ns):
            edge_index, h = dense_to_sparse(torch.from_numpy(Hs[hidx].astype(float)))
            
            x1 = np.hstack([(h.reshape((-1,1))*Plin).T, # -->(h1p1, h2p1, h3p1, ...)
                             Plin.reshape(-1,1)])
            X.append( x1 )
            y.append( xopt[hidx] )

    y = np.concatenate((y))
    X = np.concatenate((X))
    y = y[~np.any(np.isnan(X),-1)]     
    X = X[~np.any(np.isnan(X),-1)]
    
    return X, y, cinfo

