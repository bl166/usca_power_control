import os
import re
import glob
import numpy as np
import cvxpy as cp
from datetime import datetime
from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F
from collections.abc import Iterable
import warnings

import builtins as __builtin__

# np.random.seed(0)
# torch.manual_seed(0)


def print(*args, **kwargs):
    # My custom print() function: Overload print function to get time logged
    __builtin__.print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), end = ' | ')
    return __builtin__.print(*args, **kwargs)

def print_update(msg, pbar=None):
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg)

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
        
def glob_re(pattern, strings):
    return list(filter(re.compile(pattern).match, strings))

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

# convert a list of lists of different lengths to np array
def list2arr(lst):
    pad = len(max(lst, key=len))
    arr = np.array([i + [np.nan]*(pad-len(i)) for i in lst])
    return arr

# Returns True if all the elements are equal to each other
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def ordertest(A):
    for i in range( len(A) - 1 ):
        if A[i] < A[i+1]:
            return False
    return True


""" monotonic check """

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L,tol=1e-10):
    return all(x>y or np.isclose(x,y,tol) for x, y in zip(L, L[1:]))

def non_decreasing(L,tol=1e-10):
    return all(x<y or np.isclose(x,y,tol) for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)


""" pytorch model utils """

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
    if not hasattr(lo, "__len__"): # lo is a scalar
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
    rand_init = np.random.uniform(low=0.0, high=full_init)
    return rand_init

def decouple_input(x, n):
    cpt1, cpt2, cpt3 = -(n+n**2+1), -(n**2+1), -1 
    de_pmax = x[:,cpt3]
    de_h = x[:,cpt2:cpt3].view(-1, n , n )
    de_p = x[:,:cpt2]
    if x.shape[1]!=-cpt1 and x.shape[1]!=-cpt2 :
        raise ValueError('check size of input!')
    return de_p, de_h, de_pmax


""" ee metrics """

def f_pointwise_r_torch(y_pred, x, *args):
    nue = y_pred.shape[-1]
    _, de_h, de_pmax = decouple_input(x, nue)
    y_pred_ = scale_to_range(y_pred, [0, de_pmax.view(-1)])
    if not torch.all(y_pred_==y_pred):
        warnings.warn('prediction out of range! clipping...')
    y_pred = y_pred_
    s = de_h*y_pred.view((-1, 1, nue)) 
    direct = s.diagonal(dim1=-2, dim2=-1)
    ifn =  torch.sum(s, axis=-1) - direct + 1
    rates = torch.log(1+direct/ifn)
    return rates

def f_pointwise_ee_torch(y_pred, x, mu, Pc):
    rates = f_pointwise_r_torch(y_pred, x, mu, Pc)
    pt_ee = rates / (mu * y_pred + Pc)  
    return pt_ee

def f_wsee_torch(y_pred, x, mu, Pc, reduce='vector', func=f_pointwise_ee_torch, **kwargs):
    uval = func(y_pred, x, mu, Pc)
    if reduce=='vector':
        return torch.mean(uval, dim=0)
    elif reduce=='mean':
        return torch.sum(uval)/len(uval)
    elif reduce=='min':
        return torch.min(uval)[0]
    elif reduce=='sum':
        return torch.sum(uval)
    elif reduce=='none':   
        return uval
    else:
        raise ValueError
f_utility_torch = f_wsee_torch # alias

def f_wsee(p, h, mu, Pc): 
    s = h * p # (4,4) * (4,) --> (4,4)
    direct = np.diag(s)
    ifn = np.sum(s-np.diag(direct), axis=-1) + 1
    rates = np.log(1+direct/ifn)
    ee = rates / (mu * p + Pc)
    return np.sum(ee)

def gradr(p,h): 
    s = h * p
    tmp = 1 + np.sum(s, axis=-1) 
    tmp2 = tmp - np.diag(s)
    fac = np.diag(s) / (tmp * tmp2)
    grad = h.copy()      
    grad = -(fac * grad.T).T
    grad[np.diag_indices_from(grad)] = 1/tmp * np.diag(h)
    return grad

def gradf(p, h, mu, Pc):
    tmp = 1 / (mu * p + Pc)
    gr = gradr(p)
    t1 = np.sum((gr.T * tmp).T, axis=0)
    s = h * p
    direct = np.diag(s)
    ifn = np.sum(s, axis=-1) - direct + 1
    rates = np.log(1+direct/ifn)
    t2 = mu * rates * tmp**2
    return t1 - t2


""" pytorch implementation for sca optimization """  
    
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
    
    grad = -(fac.view((-1,1,n)) * beta.transpose(-2, -1)).transpose(-2, -1) 
    
    # r tilde constants
    txp = 1.0/(mu * pt + Pc)
    c1 = torch.sum(grad * txp.view((-1,1,n)) , axis=1)  
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
                
        opt.zero_grad()
        loss.backward(torch.ones(loss.shape), retain_graph=True) #vector loss
        opt.step()

        with torch.no_grad():
            pvar.copy_(scale_to_range(pvar, [0,Pmax]))
                
    return pvar.detach()    
    
    
""" other utils """

def get_cv_indices(nfold, shuffle=True, random_state=42, **kwargs):
    """
    kwargs: 
        n_observ: number of total observations; e.g. 51000
        n_interv: number of Pmax sample intervals; e.g. 51
    """
    npmax = kwargs['n_interv']
    nsamples = kwargs['n_observ']//npmax
    f_ret = kwargs['which_fold'] if 'which_fold' in kwargs else list(range(nfold))
    
    skf = StratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=random_state)

    train_index, valid_index = [],[]
    for k, (train_index_, valid_index_) in enumerate(
        skf.split(np.ones((nsamples, 1)), np.ones(nsamples, dtype=bool))):
        
        if k in f_ret:
            train_index.append( np.concatenate([np.arange(ti*npmax, (ti+1)*npmax) for ti in train_index_]) )
            valid_index.append( np.concatenate([np.arange(vi*npmax, (vi+1)*npmax) for vi in valid_index_]) )
            
    return train_index, valid_index



""" visualization utils """

def allocate(model, data, edge_index=None, batch_size=None, end=None):
    if batch_size is None:
        y_pred= _allocate(model, data, edge_index, end)
    else:
        y_pred = []
        nsamples = data.shape[0]
        for bi in range(int(np.ceil(nsamples/batch_size))):
            i_s, i_e = bi*batch_size, min((bi+1)*batch_size, nsamples)
            y_pred.append(_allocate(model, data[i_s: i_e], edge_index, end))
        y_pred = torch.cat(y_pred)
    return y_pred
    
def _allocate(model, data, edge_index, end):
    bs = data.shape[0] 
    nue = int((data.shape[1]-1)**.5)
    
    if edge_index is not None:
        shift = torch.Tensor(
                np.array([np.arange(bs)*nue,]*nue**2).T.reshape(-1)
            ).repeat(1, 2).view(2,-1).long().to(edge_index.device) 
        edge_index_batch = edge_index.repeat(1, bs)+shift 

        y_init =  data[:,:nue].reshape(-1,1)  # inital signals (p_init)
        y_constr = data[:,:nue].reshape(-1,1) # upper power constraint (p_max)

        edge_weight_batch = (data[:,nue:-1]).reshape(-1)
        input = [ (y_init, y_constr), edge_index_batch, edge_weight_batch ] 
    else:
        input = [ data[:,:] ]
    
    model.eval()
    try:
        pred = model(*input, end=end)
    except:
        pred = model(*input)
    
    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(pred, list):
        pred = pred[-1]
    return pred.view(-1, nue)

def assure_mono(arr, reduce='mean'):
    if arr.ndim < 2:
        arr = arr.reshape(1, -1)
    if reduce.lower()=='none':
        arr_mono = np.array(arr)
        for i in range(len(arr)):
            pk = max(arr[i])
            ss = arr[i].tolist().index(pk)
            arr_mono[i, ss:] = pk
        return np.squeeze(arr_mono)
    elif reduce.lower()=='mean':
        arr_mono = arr.mean(0)
        pk = max(arr_mono)
        ss = arr_mono.tolist().index(pk)
        arr_mono[ss:] = pk
        return np.squeeze(arr_mono)
    else:
        raise
        
        
def plot_shaded(x, y, ax, *, label, color, **kwargs):
    m = np.nanmean(y,0)
    e = np.nanstd(y,0)
    #ax.plot(x, m, '-', linewidth=1, alpha=1, label=label, color=color, **kwargs)
    ax.fill_between(x, m-e, m+e, alpha=.3, facecolor=color, linewidth=0, label=None)    
    
    
def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), loc='lower right')
    else:
        return bars, data.keys()

