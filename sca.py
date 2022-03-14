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

"""
OPTIMIZATION HELPER FUNCTIONS
"""


def inner_optim_helper(pt, h, mu, Pc):   
    # grad r (without main diagonal)
    s = h * pt

    degr = np.sum(s, axis=-1)
    diag = np.diag(s)
    
    tmp = 1 + degr
    tmp2 = degr - diag + 1
    
    fac = diag / (tmp * tmp2)
       
    beta = h.copy()
    beta[np.diag_indices_from(beta)] = 0
    grad = -(fac * beta.T).T

    # r tilde constants
    txp = 1.0/(mu * pt + Pc)

    c1 = np.sum(grad * txp , axis=0)
    c2 = -mu * np.log(diag/tmp2+1)*txp**2

    c = c1+c2

    d = -c * pt    
    
    return txp, tmp, tmp2, c, d

           
class grad_solver_1step(torch.nn.Module):
    def __init__(self, pt, h, mu, Pc, Pmax):
        super().__init__()
        
        self.h = torch.from_numpy(h).float()
        self.mu = torch.tensor(mu, requires_grad=False).float() # check!!
        self.Pc = torch.tensor(Pc, requires_grad=False).float()
        self.Pmax = torch.empty(1).fill_(Pmax).float()
        
        txp, tmp, tmp2, c, d = inner_optim_helper(pt, h, mu, Pc)
        
        self.txp = torch.from_numpy(txp).float()
        self.tmp = torch.from_numpy(tmp).float()
        self.tmp2 = torch.from_numpy(tmp2).float()
        self.c = torch.from_numpy(c).float()
        self.d = torch.from_numpy(d).float()

        # initialize trainable parameter
        self.pvar = torch.nn.Parameter(
            torch.tensor(pt, requires_grad=True).float()
        )
        if pt is None:
            torch.nn.init.uniform_(self.pvar, a=0.0, b=self.Pmax)
        else:# init=='full':
            with torch.no_grad():
                self.pvar.copy_(torch.from_numpy(pt))
        
    def forward(self):
        obj_nl = torch.log((torch.diag(self.h)/self.tmp2) * self.pvar + 1) @ self.txp
        obj_l  = self.c * self.pvar
        return -(obj_nl+obj_l+self.d) 
    
    def ee_eval(self, p):
        return f_wsee(p, self.h.numpy(), self.mu, self.Pc)
    


def inner_optim(pt, h, mu, Pc, Pmax, eps=1e-8, max_iters=10, learning_rate=0.1, terminate=3, solver=torch.optim.SGD):        
    # solve innner optim pvar
    model = grad_solver_1step(pt, h, mu, Pc, Pmax)
    opt = solver(model.parameters(), lr=learning_rate)

    min_track = np.inf
    stop_count = 0
    power = pt
        
    for i in range(max_iters):
        
        loss = model()
        opt.zero_grad()
        loss.backward(torch.ones(h.shape[-1])) #vector loss
        opt.step()

        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(0, Pmax)
        
        l = loss.mean().item()
        if l < min_track:
            min_track = l
            stop_count = 0
            power = model.pvar.detach().numpy()
        else:
            stop_count += 1
            
        if stop_count>terminate:
            break
                            
    return power


"""
SCA MAIN FUNCTION
"""

def SCA(h, mu, Pc, Pmax, 
        pt = None, MaxIter = 1000, SolverMaxIter = 1000, InnerSolver = torch.optim.SGD,
        parm_alpha = 1e-8, parm_beta = 0.01, LearningRate = 1e-1, 
        RelTolFun = 1e-12, RelTolVal = 1e-12, **kwargs):
    """
    Init    : last, full, ...
    """

    if pt is None:
        pt = np.full(h.shape[-1], Pmax)

    def f(p): # verified
        s = h * p # (4,4) * (4,) --> (4,4)
        
        direct = np.diag(s)
        ifn = np.sum(s, axis=-1) - direct + 1
        rates = np.log(1+direct/ifn)
        ee = rates / (mu * p + Pc)

        return np.sum(ee)

    def gradr(p): # verified
        s = h * p
        
        degr = np.sum(s, axis=-1)
        diag = np.diag(s)
        
        tmp = 1 + degr # 1 + sum beta + a
        tmp2 = degr - diag + 1
        fac = diag / (tmp * tmp2)
        
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
        ifn = np.sum(s, axis=-1) - direct + 1
        rates = np.log(1+direct/ifn)
        
        t2 = mu * rates * tmp**2

        return t1 - t2
        
    OBJ, PT = [f(pt)],[pt]
    cnt = 0
    while True:
        cnt += 1
        pvar = inner_optim(pt, h, mu, Pc, Pmax, eps=1e-8, 
                           max_iters=SolverMaxIter, learning_rate=LearningRate, solver=InnerSolver)
                
        # calculate gradient step
        Bpt = pvar - pt
        gamma = 1

        old_obj = f(pt)
        old_pt = pt
        while f(pt + gamma * Bpt) < old_obj + parm_alpha * gamma * gradf(pt) @ Bpt:
            gamma *= parm_beta

        pt += gamma * Bpt
        pt = np.clip(pt, 0, Pmax)
        obj = f(pt)
        
        OBJ.append(obj)
        PT.append(pt)

#         with np.errstate(divide='ignore'):
        if abs(obj/old_obj - 1) < RelTolFun and np.linalg.norm(pt-old_pt, np.inf) / np.linalg.norm(pt, np.inf) < RelTolVal:
            break
        
        if cnt >= MaxIter:
            break
    
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

