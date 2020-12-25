import torch
import numpy as np
import cvxpy as cp
import torch.nn.functional as F
from datetime import datetime
from prettytable import PrettyTable



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
#     return x
    lo, hi = constraints
    n,d = x.shape
    
    if not hasattr(lo, "__len__"): # hi is a scalar
        lo *= torch.ones(n).view(-1)
        
    if not hasattr(hi, "__len__"): # hi is a scalar
        hi *= torch.ones(n).view(-1)
        
    lo = lo.repeat(d).view((d,-1)).T
    hi = hi.repeat(d).view((d,-1)).T
    x_clip = torch.max(torch.min(x, hi), lo)
    
    return x_clip


def init_p(pmax, nue, method='rand'):
    full_init = np.tile(pmax,(nue,1)).T
    if method=='full':
        return full_init
#     rand_init = np.log10(np.random.uniform(low=0.0, high=10**full_init))
    rand_init = np.random.uniform(low=0.0, high=full_init)
    return rand_init


def decouple_input(x, n):
    de_pmax = x[:,-1]
    if x.shape[1]==n**2+1:# without y_pred as start
        de_h = (x[:,:-1]/de_pmax.view(*de_pmax.shape, 1)).view(-1, n , n )
    elif x.shape[1]==n**2+n+1: # with y_pred as start
        de_h = (x[:,n:-1]/de_pmax.view(*de_pmax.shape, 1)).view(-1, n , n )
    else:
        raise ValueError('check size of input!')
    return de_h, de_pmax
        
def f_wsee_torch(y_pred, x, mu, Pc, reduce='vector', **kwargs):
#     with torch.no_grad():
    n = y_pred.shape[-1]
    de_h, de_pmax = decouple_input(x, n)
                            
#     y_pred = torch.stack([torch.clamp(y_pred[pi], 0, pmax_) for pi,pmax_ in enumerate(de_x[:,-1])])
    y_pred = scale_to_range(y_pred, [0, de_pmax.view(-1)])
   
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



def inner_optim_helper(pt, h, mu, Pc):
    
    # grad r (without main diagonal)
    s = h * pt

    tmp = 1 + np.sum(s, axis=-1) # 1 + sum beta + a
    tmp2 = tmp - np.diag(s)

    fac = np.diag(s) / (tmp * tmp2)

    beta = h.copy()
    beta[np.diag_indices_from(beta)] = 0
    grad = -(fac * beta.T).T

    # r tilde constants
    txp = 1.0/(mu * pt + Pc)

    c1 = np.sum(grad * txp , axis=0)
    c2 = -mu * np.log(np.diag(s)/tmp2+1)*txp**2

    c = c1+c2

    d = -c * pt    
    
    return txp, tmp, tmp2, c, d


class grad_solver_1step(torch.nn.Module):
    def __init__(self, pt, h, mu, Pc, Pmax, init='rand'):
        super().__init__()
        
        self.h = torch.from_numpy(h).float()
        self.mu = torch.tensor(mu).float() # check!!
        self.Pc = torch.tensor(Pc).float()
        self.Pmax = torch.empty(1).fill_(Pmax).float()
        
        txp, tmp, tmp2, c, d = inner_optim_helper(pt, h, mu, Pc)
        
        self.txp = torch.from_numpy(txp).float()
        self.tmp = torch.from_numpy(tmp).float()
        self.tmp2 = torch.from_numpy(tmp2).float()
        self.c = torch.from_numpy(c).float()
        self.d = torch.from_numpy(d).float()

        # initialize 
        self.pvar = torch.nn.Parameter(
            torch.tensor(pt, requires_grad=True).float() # should be trainable parameter
        )
        if pt is None:
            torch.nn.init.uniform_(self.pvar, a=0.0, b=self.Pmax)
        else:# init=='full':
            with torch.no_grad():
                self.pvar.copy_(torch.from_numpy(pt))
    
        
    def forward(self):
        #obj_nl = cp.log(cp.multiply(np.diag(h)/tmp2, pvar)+1) @ txp
        obj_nl = torch.log((torch.diag(self.h)/self.tmp2) * self.pvar + 1) @ self.txp
        #obj_l  = cp.multiply(c, pvar)
        obj_l  = self.c * self.pvar
                
        return -1*(obj_nl+obj_l+self.d) 
    
    def ee_eval(self, p):
        return f_wsee(p, self.h.numpy(), self.mu, self.Pc)
    

def inner_optim_sgd(pt, h, mu, Pc, Pmax, eps=1e-8, max_iters=10, learning_rate=0.1):
        
    # solve innner optim pvar
    model = grad_solver_1step(pt, h, mu, Pc, Pmax)
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(max_iters):

        loss = model()
        opt.zero_grad()
        loss.backward(torch.ones(h.shape[-1])) #vector loss
        opt.step()

        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(0, Pmax)
                
    return model.pvar.detach().numpy()


"""
pytorch implementation for batches
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


# class grad_solver_1step_torch(torch.nn.Module):
#     def __init__(self, pt, h, mu, Pc, Pmax, init='rand'):
#         super().__init__()
        
#         self.h = h        
#         self.txp, self.tmp, self.tmp2, self.c, self.d = inner_optim_helper_torch(pt, h, mu, Pc)

#         # initialize 
#         self.pvar = torch.nn.Parameter(
#             pt.clone().detach().requires_grad_(True) # should be trainable parameter
#         )
# #         if pt is None:
# #             torch.nn.init.uniform_(self.pvar, a=0.0, b=Pmax)    
        
#     def forward(self):
#         direct = self.h.diagonal(dim1=-2, dim2=-1)
#         obj_nl = torch.log((direct/self.tmp2) * self.pvar + 1) * self.txp
#         obj_l  = self.c * self.pvar
#         loss = -1*(obj_nl+obj_l+self.d) 
#         return loss
    

# def inner_optim_sgd_torch(pt, h, mu, Pc, Pmax, eps=1e-8, max_iters=10, learning_rate=0.1):
        
#     # solve innner optim pvar
#     model = grad_solver_1step_torch(pt, h, mu, Pc, Pmax)
#     opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

#     for i in range(max_iters):

#         loss = model()
#         opt.zero_grad()
#         loss.backward(torch.ones(loss.shape), retain_graph=True) #vector loss
#         opt.step()

#         with torch.no_grad():
#             model.pvar.copy_(scale_to_range(model.pvar, [0,Pmax]))
                
#     return model.pvar.detach()

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
    
    
def inner_optim_cvx(pt, h, mu, Pc, Pmax, **kwargs):
    # e.g. kwargs = {'eps':1e-8, 'max_iters':SolverMaxIter, 'verbose':True}
    
    txp, tmp, tmp2, c, d = inner_optim_helper(pt, h, mu, Pc)

    # ----- solve inner problem with cvxpy -----
    pvar = cp.Variable(pt.shape, nonneg=True)
    obj_nl = cp.log(cp.multiply(np.diag(h)/tmp2, pvar)+1) @ txp ##
    obj_l  = cp.multiply(c, pvar)

    objective = cp.Maximize(cp.sum(obj_nl + obj_l + d))
    constraints = [0 <= pvar, pvar <= Pmax]
    prob = cp.Problem(objective, constraints)
    
#     print('before solver:', pvar.value, pt)
    prob.solve(requires_grad=True,**kwargs)#, eps=1e-8, max_iters=SolverMaxIter, verbose=True)
#     print('after solver:', pvar.value, pt,'\n')
    
    return pvar.value