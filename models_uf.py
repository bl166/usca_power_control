import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
from utils import *


# def form_input_mlp_numpy(pt, h, Pmax, Pc, mu):
#     x = torch.Tensor(
#         np.log10(
#             np.concatenate((pt, h.flatten()*Pmax, [Pmax]))
#         ).reshape((1,-1)).astype(float)
#     )
#     return x


# def form_input_mlp_torch(pt, h, Pmax, Pc, mu):
#     x = torch.log10(
#             torch.cat((pt.view(-1), 
#                        h.view(-1)*Pmax, 
#                        torch.empty((n,1)).fill_(Pmax)
#                       ))
#         ).view((1,-1)).float()
#     return x


# layer = ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])
class basic_mlp(nn.Module):
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.3):
        super(basic_mlp, self).__init__()
        
        # activation functions
        activations = {'elu': nn.ELU(),'relu': nn.ReLU()}
        activs = ['relu']*len(h_sizes) if activs is None else activs
        
        # modulelist container for hidden layers
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(in_size, h_sizes[0]))
        self.hidden.append(activations[activs[0]])
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.hidden.append(activations[activs[k+1]])
            self.hidden.append(nn.Dropout(dropout))
            
        self.hidden.append(nn.Linear(h_sizes[k+1], out_size))
        self.activ_fin = nn.Sigmoid()
        
    def forward(self, x, constraints=[0,1]):
        for i, hidden in enumerate(self.hidden):
            x = hidden(x)
        # scale to range 
        x = scale_to_range(x, constraints)
        return x

# class ClipTensorRange(nn.Module):
#     def forward(self, x, constraints=[0,1]):
#         return scale_to_range(x, constraints)
    
# layer = ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])
class basic_gcn(torch.nn.Module):
    """
    basic gcn block
    """
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0):
        super(basic_gcn, self).__init__()
        
        # activation functions
        activations = {'elu': nn.ELU(),'relu': nn.ReLU(), 'linear': nn.Linear(out_size, out_size)}
        activs = (['relu']*len(h_sizes) if activs is None else activs) + ['linear']
        self.activs = nn.ModuleList([activations[a] for a in activs])
        
        # modulelist container for hidden layers
        hidden_sizes = [in_size, *h_sizes, out_size]
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            self.hidden.append(GCNConv(hidden_sizes[k], hidden_sizes[k+1]))
        #self.final = nn.Linear(out_size, out_size) 
        self.activ_fin = nn.Sigmoid()
        
    def forward(self, x, edge_index, edge_weights, constraints=None):
        for i, (hidden, activa) in enumerate(zip(self.hidden, self.activs)):
            x = hidden(x, edge_index, edge_weight=edge_weights)
            x = activa(x)
        # scale to range 
        x = scale_to_range(x, constraints)
        return x

    
    
class unfold_block_mlp(nn.Module):
    def __init__(self, in_size, out_size, h_sizes, 
                 activs=None, dropout=0.3, inner_optim='learned-mlp', channel_info={'mu':4, 'Pc':1}):
        super(unfold_block_mlp, self).__init__()
        
        self.gamma = basic_mlp(in_size, out_size, h_sizes, activs, dropout)
        self.size  = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']
        
        m1, m2 = inner_optim.split('-')
        if m1=='fixed':
            if m2=='sgd':
#                 self.optim = inner_optim_sgd
#                 self.kwargs = {'learning_rate':.1, 'max_iters':10}
                self.forward = self.forward_fixed
            elif m2=='cvx':
                pass
            else:
                raise
        elif m1 == 'learned':
            if m2=='mlp':
                self.optim = basic_mlp(in_size, out_size, h_sizes, activs, dropout)
#                 self.kwargs = {}
                self.forward = self.forward_learned
            elif m2=='gcn':
                pass
            else:
                raise            
        else:
            raise
            
    def forward_learned(self, x, constraints):
        # optim p2/3
        x_sol = self.optim(x, constraints)
        # compute gamma
        gamma = self.gamma(x, [0,1])
        # update pt
        x_new = x[:,:self.size] + gamma * (x_sol - x[:,:self.size])
        
        return x_new, gamma
    
    
    def forward_fixed(self, x, constraints):
        pt = x[:,:self.size]
        _, h, Pmax = decouple_input(x, self.size)
#         assert constraints[1]==Pmax
        gamma = self.gamma(x, [0,1])
        x_sol = inner_optim_sgd_torch(pt, h, self.mu, self.Pc, Pmax, eps=1e-8, max_iters=7, learning_rate=.01) 
        x_new = x[:,:self.size] + gamma * (x_sol - x[:,:self.size])
        
        return x_new, gamma
        
        
class unfold_block_gcn(nn.Module):
    def __init__(self, in_size, out_size, h_sizes, 
                 activs=None, dropout=0, inner_optim='learned-gcn', channel_info={'mu':4, 'Pc':1, 'edge_index':None}):
        super(unfold_block_gcn, self).__init__()
        
        self.gamma = basic_gcn(in_size, out_size, h_sizes, activs, dropout) # Note: "dropout" not working yet
        self.size  = out_size # 1 --> NOTE!! THIS SIZE IS DIFFERENT; ACTUALLY IT'S THE "FEATURE DIMENTION"
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']     
        self.ei = channel_info['edge_index']
        
        m1, m2 = inner_optim.split('-')
        if m1 == 'learned':
            if m2=='mlp':
                raise
                #self.optim = basic_mlp(in_size, out_size, h_sizes, activs, dropout)
            elif m2=='gcn':
                self.optim = basic_gcn(in_size, out_size, h_sizes, activs, dropout)
            else:
                raise
            self.forward = self.forward_learned
        else:
            raise
            
    def forward_learned(self, x, edge_index, edge_weights, constraints):
        # optim p2/3
        x_sol = self.optim(x, edge_index, edge_weights, constraints=constraints)
        # compute gamma
        gamma = self.gamma(x, edge_index, edge_weights, constraints=constraints and [0,1])
        # update pt
        x_new = x + gamma * (x_sol - x)
        
        return x_new, gamma
    
    
class USCA_GCN(nn.Module):
    def __init__(self, num_layers, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1, 'edge_index':None}, 
                 activs=None, dropout=0.3, inner_optim='learned-gcn', **extra):
        super().__init__()
        # build a block
        self.sca = nn.ModuleList()
        for l in range(num_layers):
            self.sca.append(unfold_block_gcn(in_size, out_size, h_sizes, activs, dropout, inner_optim, channel_info))
        self.size = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']
        self.ei = channel_info['edge_index']
        
    def forward(self, x, edge_index, edge_weights):
        #print(x[0].shape, edge_index.shape, edge_weights.shape)
        
        pt, constraints = x[0], x[1]
        pt_list, gamma_list = [x[0]],[]
        for i, sca_block in enumerate(self.sca):
            pt, gamma = sca_block(pt, edge_index, edge_weights, constraints=None)#[0, x[1].view(-1)])
            
            pt = scale_to_range(pt, constraints=[0, x[1].view(-1)])
            gamma = scale_to_range(gamma, constraints=[0,1])
    
            pt_list.append(pt)
            gamma_list.append(gamma)
            
        if self.training:
            return pt, gamma
        else:
            return pt_list, gamma_list #pt, gamma
        
        
        
# class USCA_GAT(nn.Module):
#     def __init__(self, num_layers, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1, 'edge_index':None}, 
#                  activs=None, dropout=0.3, inner_optim='learned-mlp'):
#         super().__init__()
#         # build a block
#         self.sca = nn.ModuleList()
#         for l in range(num_layers):
#             self.sca.append(unfold_block_gcn(in_size, out_size, h_sizes, activs, dropout, inner_optim, channel_info))
#         self.size = out_size
#         self.mu = channel_info['mu']
#         self.Pc = channel_info['Pc']
#         self.ei = channel_info['edge_index']
        
#     def forward(self, x, edge_index, edge_weights):
#         #print(x[0].shape, edge_index.shape, edge_weights.shape)
        
#         pt, constraints = x[0], x[1]
#         pt_list, gamma_list = [x[0]],[]
#         for i, sca_block in enumerate(self.sca):
#             pt, gamma = sca_block(pt, edge_index, edge_weights, constraints=[0, x[1].view(-1)])
#             pt_list.append(pt)
#             gamma_list.append(gamma)
            
#         if self.training:
#             return pt, gamma
#         else:
#             return pt_list, gamma_list #pt, gamma        
        
        
        
class USCA_MLP(nn.Module):
    def __init__(self, num_layers, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1}, 
                 activs=None, dropout=0.3, inner_optim='learned-mlp', **extra):
        super().__init__()
        # build a block
        self.sca = nn.ModuleList()
        for l in range(num_layers):
            self.sca.append(unfold_block_mlp(in_size, out_size, h_sizes, activs, dropout, inner_optim, channel_info))
        self.size = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']
        
    def forward(self, x):  
        channels = x[..., self.size:]
        pt_list, gamma_list = [x[..., :self.size]],[]
        for i, sca_block in enumerate(self.sca):
            pt, gamma = sca_block(x, constraints=[0, channels[:,-1]])
            pt_list.append(pt)
            gamma_list.append(gamma)

            x = torch.cat((pt, channels), -1)

        if self.training:
            return pt, gamma
        else:
            return pt_list, gamma_list #pt, gamma
        
        
class USCA_MLP_R(nn.Module):
    def __init__(self, num_layers, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1},
                 activs=None, dropout=0.3, inner_optim='learned-mlp', **extra):
        super().__init__()
        # build a block
        self.sca = nn.ModuleList()
        sca_block = unfold_block_mlp(in_size, out_size, h_sizes, activs, dropout, inner_optim, channel_info)
        for l in range(num_layers):
            self.sca.append(sca_block)
        self.size = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']
        
    def forward(self, x):  
        channels = x[..., self.size:]
        pt_list, gamma_list = [x[..., :self.size]],[]
        for i, sca_block in enumerate(self.sca):
            pt, gamma = sca_block(x, constraints=[0, channels[:,-1]])
            pt_list.append(pt)
            gamma_list.append(gamma)

            x = torch.cat((pt, channels), -1)

        if self.training:
            return pt, gamma
        else:
            return pt_list, gamma_list #pt, gamma
     
    
    
    
"""
VANILLA E2E ARCHITECTURES 
"""
class MLP_ChPm(nn.Module): 
    """
    MLP with Channel parameters and Pmax as input 
    """
    def __init__(self, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1}, 
                 activs=None, dropout=0.3, inner_optim='vanilla', **extra):
        super(MLP_ChPm, self).__init__()
        self.model = basic_mlp(in_size, out_size, h_sizes, activs, dropout)
        self.size  = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']        
    
    def forward(self, x):
        x = self.model(x[:,self.size:], constraints=[0, x[:,-1]])
        return x,None
    
    
class GCN_ChPt(torch.nn.Module):
    """
    GCN with channel parameters as edge weights and pt as node signals
    """
    def __init__(self, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1}, 
                 activs=None, dropout=0.3, inner_optim='vanilla', **extra):
        super(GCN_ChPt, self).__init__()
        self.model = basic_gcn(in_size, out_size, h_sizes, activs, dropout)
        self.size  = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']     
        self.ei = channel_info['edge_index']

    def forward(self, x, edge_index, edge_weights):       
        #print(x[0].shape, edge_index.shape, edge_weights.shape)
        x = self.model(x[0], edge_index, edge_weights, constraints=[0, x[1].view(-1)])
        return x,None
    
