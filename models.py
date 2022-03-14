import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import utils 


class basic_mlp(nn.Module):
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.3, **extra):
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
        x = utils.scale_to_range(x, constraints)
        return x
    

class unfold_block_mlp(nn.Module):
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.3, channel_info={'mu':4, 'Pc':1}, **extra):
        super(unfold_block_mlp, self).__init__()
        self.size  = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']
        self.build_model(in_size, out_size, h_sizes, activs, dropout)

    def build_model(self, in_size, out_size, h_sizes, activs, dropout):
        self.ssize = out_size//2
        self.gamma = basic_mlp(in_size+self.ssize, self.ssize, h_sizes, activs, dropout)
        self.optim = basic_mlp(in_size, out_size, h_sizes, activs, dropout)
        
    def forward(self, x, constraints=None):
        # optim p2/3
        x_sol = self.optim(x, constraints) # None
        # compute gamma
        gamma = self.gamma(torch.cat((x,x_sol[:,self.ssize:]),1), constraints=[0,1])
        
        # update pt
        s,g = x_sol.shape[1], gamma.shape[1]
        if s == g: 
            # no embedding -- sizes -- x:2, x_sol:1, gamma:1
            x_new = x[:, -g:] + gamma * (x_sol - x[:, -g:])
        else: 
            # embedding -- sizes -- x:2, x_sol:2, gamma:1
            x_new = x_sol
            x_new[:, -g:] = x[:, -g:] + gamma * (x_sol[:, -g:] - x[:, -g:])
        return x_new, gamma    
    
    
# layer = ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])
class basic_gcn(torch.nn.Module):
    """
    basic gcn block
    """
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0, **extra):
        super(basic_gcn, self).__init__()
        
        # activation functions
        activations = {'elu': nn.ELU(),'relu': nn.ReLU(), 'selu': nn.SELU(), 
                       'linear': nn.Linear(out_size, out_size)}
        activs = (['relu']*len(h_sizes) if activs is None else activs) + ['linear']
        self.activs = nn.ModuleList([activations[a] for a in activs])
        
        # modulelist container for hidden layers
        hidden_sizes = [in_size, *h_sizes, out_size]
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            self.hidden.append(GCNConv(hidden_sizes[k], hidden_sizes[k+1]))
        
    def forward(self, x, edge_index, edge_weights, constraints=None):
        for i, (hidden, activa) in enumerate(zip(self.hidden, self.activs)):
            x = hidden(x, edge_index, edge_weight=edge_weights)
            x = activa(x)
        # scale to range 
        if constraints is not None:
            x = utils.scale_to_range(x, constraints)
        return x

    
class unfold_block_gcn(nn.Module):
    def __init__(self, in_size, out_size, h_sizes, activs = None, dropout = 0, 
                 channel_info = {'mu':4, 'Pc':1, 'edge_index':None}, **extra):
        super(unfold_block_gcn, self).__init__()
        self.size  = out_size 
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']     
        self.ei = channel_info['edge_index']
        self.build_model(in_size, out_size, h_sizes, activs, dropout)
            
    def build_model(self, in_size, out_size, h_sizes, activs, dropout):
        self.gamma = basic_gcn(in_size+1, 1, h_sizes, activs, dropout)
        self.optim = basic_gcn(in_size, out_size, h_sizes, activs, dropout)
        
    def forward(self, x, edge_index, edge_weights, constraints=None):
        # optim p2/3
        x_sol = self.optim(x, edge_index, edge_weights, constraints=constraints) 
        # compute gamma
        gamma = self.gamma(torch.cat((x,x_sol[:,-1:]),1), edge_index, edge_weights, constraints=[0,1])
        
        # update pt
        s,g = x_sol.shape[1], gamma.shape[1]
        if s == g: 
            # no embedding -- sizes -- x:2, x_sol:1, gamma:1
            x_new = x[:, -g:] + gamma * (x_sol - x[:, -g:])
        else: 
            # embedding -- sizes -- x:2, x_sol:2, gamma:1
            x_new = x_sol
            x_new[:, -g:] = x[:, -g:] + gamma * (x_sol[:, -g:] - x[:, -g:])
        return x_new, gamma    
    

class USCA_GCN_Embed(nn.Module):
    def __init__(self, num_layers, in_size, out_size, h_sizes, activs = None, dropout = 0.3, 
                 channel_info = {'mu':4, 'Pc':1, 'edge_index':None}, **extra):
        super().__init__()
        include = ['in_size', 'out_size', 'h_sizes', 'activs', 'dropout', 'channel_info']
        self.saved_args = {k:v for k,v in locals().items() if k in include}
        self.nblocks = num_layers
        
        # build a block
        self.build_model(**self.saved_args) 
        
        self.size = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']
        self.ei = channel_info['edge_index']
            
    def build_model(self, **kwargs):
        first_kwargs = {k:v for k,v in kwargs.items() if k !='emded_size'}
        outsize = first_kwargs['out_size']
        embsize = kwargs['emded_size'] if 'emded_size' in kwargs else outsize
        first_kwargs['out_size'] = embsize
        self.embedding = basic_gcn(**first_kwargs)
        
        first_kwargs['out_size'] += 1 # add a place for concat constraint
        first_kwargs['in_size'] = first_kwargs['out_size']
        self.sca = nn.ModuleList()
        for l in range(self.nblocks):
            self.sca.append(unfold_block_gcn(**first_kwargs))
            
    def forward(self, x, edge_index, edge_weights, start=0, end=None):
        pt, constraints = x[0], x[1]
        end = self.nblocks if end is None else end
        
        pt_emb = self.embedding(torch.ones(pt.shape).to(pt.device), 
                        edge_index, edge_weights, constraints=None)   
        pt = torch.cat((pt_emb, constraints), 1)

        for i in range(end):
            pt, gamma = self.sca[i](pt, edge_index, edge_weights, constraints=None)#[0, x[1].view(-1)])
            pt_skip = pt[:,:gamma.shape[1]]
            pt_constr = utils.scale_to_range(pt[:,-gamma.shape[1]:], constraints=[0, x[1].view(-1)])
            if i<end-1:
                pt = torch.cat((pt_skip, pt_constr), -1)
            else: 
                pt = pt_constr
        return pt, None
    
    
class USCA_GCN_Embed_R(USCA_GCN_Embed):
    def build_model(self, **kwargs):
        first_kwargs = {k:v for k,v in kwargs.items() if k !='emded_size'}
        outsize = first_kwargs['out_size']
        embsize = kwargs['emded_size'] if 'emded_size' in kwargs else outsize
        first_kwargs['out_size'] = embsize
        self.embedding = basic_gcn(**first_kwargs)
        
        first_kwargs['out_size'] += 1 # add a place for concat constraint
        first_kwargs['in_size'] = first_kwargs['out_size']
        self.sca = nn.ModuleList()
        self.sca.append(unfold_block_gcn(**first_kwargs))
    
    def forward(self, x, edge_index, edge_weights, start=0, end=None):
        pt, constraints = x[0], x[1]
        end = self.nblocks if end is None else end

        pt_emb = self.embedding(torch.ones(pt.shape).to(pt.device), edge_index, edge_weights, constraints=None)   
        pt = torch.cat((pt_emb, constraints), 1)

        for i in range(end):
            pt, gamma = self.sca[0](pt, edge_index, edge_weights, constraints=None)
            pt_skip = pt[:,:gamma.shape[1]]
            pt_constr = utils.scale_to_range(pt[:,-gamma.shape[1]:], constraints=[0, x[1].view(-1)])
            if i<end-1:
                pt = torch.cat((pt_skip, pt_constr), -1)
            else: 
                pt = pt_constr
        return pt, None
    
        
class USCA_MLP_Embed(nn.Module):
    def __init__(self, num_layers, in_size, out_size, h_sizes, 
                 channel_info = {'mu':4, 'Pc':1}, activs = None, dropout = 0.3, **extra):
        super().__init__()
        
        include = ['in_size', 'out_size', 'h_sizes', 'activs', 'dropout', 'inner_optim']
        self.saved_args = {k:v for k,v in locals().items() if k in include}
        self.nblocks = num_layers
        
        # build a block
        self.build_model(**self.saved_args) 
        
        self.size = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']
                
    def build_model(self, **kwargs):
        print(kwargs)
        first_kwargs = {k:v for k,v in kwargs.items() if k !='emded_size'}
        
        outsize = kwargs['out_size']
        embsize = kwargs['emded_size'] if 'emded_size' in kwargs else outsize
        
        first_kwargs['in_size'] -= 1 
        first_kwargs['out_size'] = embsize
        self.embedding = basic_mlp(**first_kwargs)
        
        first_kwargs['in_size'] = kwargs['in_size']-1 + 2*kwargs['out_size']
        first_kwargs['out_size'] = 2*kwargs['out_size']
        self.sca = nn.ModuleList()
        for l in range(self.nblocks):
            self.sca.append(unfold_block_mlp(**first_kwargs))
        
    def forward(self, x, start=0, end=None):
        end = len(self.sca) if end is None else end
        channels = torch.log10(x[..., self.size:-1])
        pt = x[..., :self.size]
        
        pt_emb = self.embedding(channels, constraints=None)   
        pt = torch.cat((channels, pt_emb, pt), 1)
        
        for i in range(end):
            pt, gamma = self.sca[i](pt, constraints=None)#[0, x[:,-1]])
            pt_skip = pt[:,:gamma.shape[1]]
            pt_constr = utils.scale_to_range(pt[:,-gamma.shape[1]:], constraints=[0, x[:,-1]])

            if i < end-1:
                pt = torch.cat((channels, pt_skip, pt_constr), -1)
            else: 
                pt = pt_constr

        return pt, gamma       
    
    
class USCA_MLP_Embed_R(USCA_MLP_Embed):
    def build_model(self, **kwargs):
        first_kwargs = {k:v for k,v in kwargs.items() if k !='emded_size'}
        
        outsize = kwargs['out_size']
        embsize = kwargs['emded_size'] if 'emded_size' in kwargs else outsize
        
        first_kwargs['in_size'] -= 1 
        first_kwargs['out_size'] = embsize
        self.embedding = basic_mlp(**first_kwargs)
        
        first_kwargs['in_size'] = kwargs['in_size']-1 + 2*kwargs['out_size']
        first_kwargs['out_size'] = 2*kwargs['out_size']
        self.sca = nn.ModuleList()
        self.sca.append(unfold_block_mlp(**first_kwargs))
        
    def forward(self, x, start=0, end=None):
        end = len(self.sca) if end is None else end
        channels = torch.log10(x[..., self.size:-1])
        pt = x[..., :self.size]
        
        pt_emb = self.embedding(channels, constraints=None)   
        pt = torch.cat((channels, pt_emb, pt), 1)
        
        for i in range(end):
            pt, gamma = self.sca[0](pt, constraints=None)
            pt_skip = pt[:,:gamma.shape[1]]
            pt_constr = utils.scale_to_range(pt[:,-gamma.shape[1]:], constraints=[0, x[:,-1]])

            if i < end-1:
                pt = torch.cat((channels, pt_skip, pt_constr), -1)
            else: 
                pt = pt_constr

        return pt, gamma       
    
    
"""
VANILLA E2E ARCHITECTURES 
"""
class MLP_ChPm(nn.Module): 
    """
    MLP with Channel parameters and Pmax as input 
    """
    def __init__(self, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1}, activs=None, dropout=0.3, **extra):
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
    def __init__(self, in_size, out_size, h_sizes, channel_info={'mu':4, 'Pc':1}, activs=None, dropout=0.3, **extra):
        super(GCN_ChPt, self).__init__()        
        activs=None
        self.model = basic_gcn(in_size, out_size, h_sizes, activs, dropout)
        self.size  = out_size
        self.mu = channel_info['mu']
        self.Pc = channel_info['Pc']     
        self.ei = channel_info['edge_index']

    def forward(self, x, edge_index, edge_weights):       
        x = self.model(x[0], edge_index, edge_weights, constraints=[0, x[1].view(-1)])
        return x,None
    

