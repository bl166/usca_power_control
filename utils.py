import torch
import numpy as np
import cvxpy as cp
import torch.nn.functional as F
from datetime import datetime
from prettytable import PrettyTable
import glob

import builtins as __builtin__
def print(*args, **kwargs):
    # My custom print() function: Overload print function to get time logged
    __builtin__.print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), end = ' | ')
    return __builtin__.print(*args, **kwargs)

def print_update(msg, pbar=None):
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg)
        
        
def anyvals(string, anyval):
    substrings = string.split('_')
    for si, substr in enumerate(substrings):
        if any(substr.startswith(aval) for aval in anyval):
            substrings[si] = substr.split('+')[0]+'*'
    return '_'.join(substrings)


def glob_escape_except(string, exp='*'):
    estring = glob.escape(string)
    for c in exp:
        estring = estring.replace('['+c+']',c)
    return estring


def plot_shaded(*, x, y, ax, label, color):
    m = np.nanmean(y,0)
    e = np.nanstd(y,0)

    #axis.plot(mse_m, label=exper, linewidth=1, alpha=.8)
    ax.plot(x, m, '-', linewidth=1, alpha=1, label=label, color=color)
    ax.fill_between(x, m-e, m+e,
        alpha=.3, facecolor=color, #edgecolor='#3F7F4C', facecolor='#7EFF99',
        linewidth=0, label=None)    


# convert a list of lists of different lengths to np array
def list2arr(lst):
    pad = len(max(lst, key=len))
    arr = np.array([i + [np.nan]*(pad-len(i)) for i in lst])
    #np.array([np.pad(i, ((0,pad-len(i)),(0,0))) for i in lst])
    return arr


# reinitialize model parameters     
def reset_model_parameters(m):
    # check if m is iterable or if m has children
    # base case: m is not iterable, then check if it can be reset; if yes --> reset; no --> do nothing

    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
    else:
        for l in m.children():
            reset_model_parameters(l)


def count_parameters(model, verbose=0):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: 
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
    if verbose >= 2:
        print(table)
    if verbose >= 1:
        print(f"Total Trainable Params: {total_params}")
    return total_params


def scale_to_range(x, constraints):
    if constraints is None:
        return x

    device = x.device
    
    lo, hi = constraints
    n,d = x.shape
    
    if not hasattr(lo, "__len__"): # hi is a scalar
        lo *= torch.ones(n).view(-1).to(device)
        
    if not hasattr(hi, "__len__"): # hi is a scalar
        hi *= torch.ones(n).view(-1).to(device)
        
    lo = lo.repeat(d).view((d,-1)).T
    hi = hi.repeat(d).view((d,-1)).T
    x_clip = torch.max(torch.min(x, hi), lo)
    
    return x_clip


def init_p(pmax, nue, method='full'):
    full_init = np.tile(pmax,(nue,1)).T
    if method=='full':
        return full_init
#     rand_init = np.log10(np.random.uniform(low=0.0, high=10**full_init))
    rand_init = np.random.uniform(low=0.0, high=full_init)
    return rand_init


def decouple_input(x, n):
    cpt1, cpt2, cpt3 = -(n+n**2+1), -(n**2+1), -1  # with/without y_pred as start
    de_pmax = x[:,cpt3]
    de_h = (x[:,cpt2:cpt3]/de_pmax.view(*de_pmax.shape, 1)).view(-1, n , n )
    de_p = x[:,:cpt2]
        
    if x.shape[1]!=-cpt1 and x.shape[1]!=-cpt2 :
        raise ValueError('check size of input!')
    return de_p, de_h, de_pmax
        
def f_wsee_torch(y_pred, x, mu, Pc, reduce='vector', **kwargs):
#     with torch.no_grad():
    n = y_pred.shape[-1]
    _, de_h, de_pmax = decouple_input(x, n)
                            
#     y_pred = torch.stack([torch.clamp(y_pred[pi], 0, pmax_) for pi,pmax_ in enumerate(de_x[:,-1])])
#     print(y_pred)
    y_pred = scale_to_range(y_pred, [0, de_pmax.view(-1)])
#     print(y_pred)
   
    s = de_h*y_pred.view((-1,1,n)) # (4,4) * (4,) --> (4,4)
    direct = s.diagonal(dim1=-2, dim2=-1)
    ifn =  torch.sum(s, axis=-1) - direct + 1
#     ifn = 1 + torch.sum(s, axis=-1) - direct
    rates = torch.log(1+direct/ifn)
    ee = rates / (mu * y_pred + Pc)  

    if reduce=='vector':
        loss = torch.mean(ee, dim=0)
    elif reduce=='mean':
        loss = torch.sum(ee)/len(ee)
    elif reduce=='sum':
        loss = torch.sum(ee)
    elif reduce=='none':   
        loss = ee
    else:
        raise ValueError
    return loss


def f_wsee(p, h, mu, Pc): # verified
    s = h * p # (4,4) * (4,) --> (4,4)

    direct = np.diag(s)
#     ifn = np.sum(s, axis=-1) - direct + 1
    ifn = np.sum(s-np.diag(direct), axis=-1) + 1
    rates = np.log(1+direct/ifn)
    ee = rates / (mu * p + Pc)

    return np.sum(ee)


def gradr(p,h): # verified
    s = h * p
    tmp = 1 + np.sum(s, axis=-1) # 1 + sum beta + a
    tmp2 = tmp - np.diag(s)
    fac = np.diag(s) / (tmp * tmp2)

    grad = h.copy()      
    grad = -(fac * grad.T).T

    grad[np.diag_indices_from(grad)] = 1/tmp * np.diag(h)#tmp2/(tmp*tmp2) * np.diag(h)

    return grad


def gradf(p, h, mu, Pc): # verified
    tmp = 1 / (mu * p + Pc)
    gr = gradr(p)

    t1 = np.sum((gr.T * tmp).T, axis=0)

    s = h * p
    direct = np.diag(s)
    ifn = np.sum(s, axis=-1) - direct + 1
    rates = np.log(1+direct/ifn)

    t2 = mu * rates * tmp**2

    return t1 - t2




"""
pytorch implementation for batches 
(FOR ORIGINAL IMPLEMENTATION: SEE SCA.PY)
"""

# s = de_h*y_pred.view((-1,1,n)) # (4,4) * (4,) --> (4,4)
# direct = s.diagonal(dim1=-2, dim2=-1)
# ifn =  torch.sum(s, axis=-1) - direct + 1
# #     ifn = 1 + torch.sum(s, axis=-1) - direct
# rates = torch.log(1+direct/ifn)
# ee = rates / (mu * y_pred + Pc)      
    
def inner_optim_helper_torch(pt, h, mu, Pc):
    
    n = pt.shape[-1]

    # grad r (without main diagonal)
    s = h * pt.view((-1,1,n))
    
    tmp_1 = torch.sum(s, axis=-1) # 1 + sum beta + a
    tmp = 1+tmp_1
    
    direct = s.diagonal(dim1=-2, dim2=-1)
    tmp2 = tmp_1 - direct + 1
    fac = direct / (tmp * tmp2)
    
    beta = h.clone()
    beta.diagonal(dim1=-2, dim2=-1)[:] = 0     
    
    grad = -(fac.view((-1,1,n)) * beta.transpose(-2, -1)).transpose(-2, -1) # verified
    
    # r tilde constants
    txp = 1.0/(mu * pt + Pc)
    c1 = torch.sum(grad * txp.view((-1,1,n)) , axis=1)  # verified
    c2 = -mu * torch.log(direct/tmp2+1)*txp**2

    c = c1+c2

    d = -c * pt    
    
    return txp, tmp, tmp2, c, d


def inner_optim_sgd_torch(pt, h, mu, Pc, Pmax, eps=1e-8, max_iters=10, learning_rate=0.1):
        
    pvar = torch.nn.Parameter(
            pt.clone().detach().requires_grad_(True) # should be trainable parameter
        )        
    txp, tmp, tmp2, c, d = inner_optim_helper_torch(pt, h, mu, Pc)
    opt = torch.optim.SGD([pvar], lr=learning_rate)

    for i in range(max_iters):

        direct = h.diagonal(dim1=-2, dim2=-1)
        inlog = (direct/tmp2) * pvar + 1

        obj_nl = torch.log(inlog) * txp
        obj_l  = c * pvar
        loss = -1*(obj_nl+obj_l+d) 
        
#         print(loss)
        
        opt.zero_grad()
        loss.backward(torch.ones(loss.shape), retain_graph=True) #vector loss
        opt.step()

        with torch.no_grad():
            pvar.copy_(scale_to_range(pvar, [0,Pmax]))
                
    return pvar.detach()    
    
    
