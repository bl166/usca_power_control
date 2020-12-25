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


def usca_decouple_input_shape(input, nue=4):
    n, d = input.shape
    assert nue**2+nue+1 == d
    input = 10**input
    
    pmax = input[:,-1].view((-1,1)) #(n, 1)
    pt = input[:,:nue].view((-1,nue))
    h = input[:,nue:-1].view((-1, nue, nue))/pmax.unsqueeze_(-1) #(n, 4, 4)
    return pt, h, pmax
    
    
# layer = ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])
class basic_gnn(torch.nn.Module):
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0):
        super(basic_gnn, self).__init__()
        
        # activation functions
        activations = {'elu': nn.ELU(),'relu': nn.ReLU()}
        activs = ['relu']*len(h_sizes) if activs is None else activs
        
        # modulelist container for hidden layers
        self.hidden = nn.ModuleList()
        self.hidden.append(GCNConv(in_size, h_sizes[0]))
        self.hidden.append(activations[activs[0]])
        for k in range(len(h_sizes)-1):
            self.hidden.append(GCNConv(h_sizes[k], h_sizes[k+1]))
            self.hidden.append(activations[activs[k+1]])
#             self.hidden.append(nn.Dropout(dropout))
            
        self.hidden.append(GCNConv(h_sizes[k+1], out_size))
        # self.final = nn.Linear(out_size, out_size) 
        self.activ_fin = nn.Sigmoid()
        

    def forward(self, x, edge_index, edge_weights):
        for i, hidden in enumerate(self.hidden):
            x = hidden(x, edge_index, edge_weight=edge_weights)
        # scale to range 
        x = scale_to_range(x, constraints)

        return x


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


class unfold_block_mlp(nn.Module):
    def __init__(self, in_size, out_size, h_sizes, 
                 activs=None, dropout=0.3, inner_optim='fixed-sgd', channel_info={'mu':4, 'Pc':1}):
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
            elif m2=='gnn':
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
        h, Pmax = decouple_input(x, self.size)
#         assert constraints[1]==Pmax
        gamma = self.gamma(x, [0,1])
        x_sol = inner_optim_sgd_torch(pt, h, self.mu, self.Pc, Pmax, eps=1e-8, max_iters=7, learning_rate=.01) 
        x_new = x[:,:self.size] + gamma * (x_sol - x[:,:self.size])
        
        return x_new, gamma
        
        
class unfold_block_gnn(nn.Module):
    def __init__(self, in_size, out_size, h_sizes, 
                 activs=None, dropout=0.3, inner_optim='fixed-sgd', channel_info={'mu':4, 'Pc':1}):
        super(unfold_block_gnn, self).__init__()
        
        self.gamma = basic_gnn(in_size, out_size, h_sizes, activs, dropout)
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
                self.optim = basic_gnn(in_size, out_size, h_sizes, activs, dropout)
#                 self.kwargs = {}
                self.forward = self.forward_learned
            elif m2=='gnn':
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
        h, Pmax = decouple_input(x, self.size)
#         assert constraints[1]==Pmax
        gamma = self.gamma(x, [0,1])
        x_sol = inner_optim_sgd_torch(pt, h, self.mu, self.Pc, Pmax, eps=1e-8, max_iters=7, learning_rate=.01) 
        x_new = x[:,:self.size] + gamma * (x_sol - x[:,:self.size])
        
        return x_new, gamma
    
    

class USCA_MLP_R(nn.Module):
    def __init__(self, num_layers, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1},
                 activs=None, dropout=0.3, inner_optim='learned-mlp'):
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
#         pt, h, Pmax = usca_decouple_input_shape(x, self.nue)
        channels = x[..., self.size:]
        for i, sca_block in enumerate(self.sca):
            pt, gamma = sca_block(x, constraints=[0, channels[:,-1]])#pt, h, Pmax, self.Pc, self.mu)
            x = torch.cat((pt, channels), -1)
#             pt = pt.data.numpy()[0]
#             print(gamma)

        return pt, gamma
    
    
    
class USCA_MLP(nn.Module):
    def __init__(self, num_layers, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1}, 
                 activs=None, dropout=0.3, inner_optim='learned-mlp'):
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
            pt, gamma = sca_block(x, constraints=[0, channels[:,-1]])#pt, h, Pmax, self.Pc, self.mu)
            pt_list.append(pt)
            gamma_list.append(gamma)

            x = torch.cat((pt, channels), -1)

        if self.training:
            return pt, gamma
        else:
            return pt_list, gamma_list #pt, gamma
        
    