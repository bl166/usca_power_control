import torch
torch.backends.cudnn.benchmark =True
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.tensorboard import SummaryWriter
# PyTorch logger: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py

import os
import json
import time
import warnings
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

# # control randomness
# np.random.seed(0)
# torch.manual_seed(0)

import utils 

# Save model with epoch number and validation error
def save_network(model, epoch, err, optimizer, save_path):
    # deal with multi-gpu parallelism
    pa = type(model)
    if pa.__name__ == 'DataParallel':
        net_save = model.module
    else:
        net_save = model
    # format to save
    state = {
        'net'  : net_save,
        'err'  : err,  # error rate or negative loss (something to minimize)
        'epoch': epoch,
        'opt'  : optimizer,
    }
    torch.save(state, save_path)


# Load model with epoch number and validation error
def load_network(network, net_path):
    
    optimizer = None
    epoch = -1
    err = np.nan

    if network: # if the given template network is not none

        check_pt = torch.load(net_path)

        if 'net' in check_pt:
            checkpoint = check_pt['net']
            epoch = check_pt['epoch']
            err = check_pt['err']
            if 0:#'opt' in check_pt:
                optimizer = check_pt['opt']
        else:
            checkpoint = check_pt

        # assert checkpoint should be a nn.module or parallel nn.module
        pa_0 = type(checkpoint)
        if pa_0.__name__ == 'DataParallel':
            checkpoint = checkpoint.module
        else:
            pass

        new_dict = checkpoint.state_dict()

        # same to the network template
        pa_1 = type(network)
        if pa_1.__name__ == 'DataParallel':
            network = network.module

        old_dict = network.state_dict()

        # make sure new and old state_dict match each other
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())

        if old_keys == new_keys:
            network.load_state_dict(new_dict)

        else:
            if new_keys-old_keys:
                warnings.warn("Ignoring keys in new dict: {}".format(new_keys-old_keys))
            if old_keys-new_keys:
                warnings.warn("Missing keys in old dict: {}".format(old_keys-new_keys))

            # filter out unnecessary keys
            new_dict = {k: v for k, v in new_dict.items() if k in old_dict}
            # overwrite entries in the existing state dict
            old_dict.update(new_dict)
            # load the new state dict
            network.load_state_dict(old_dict)

        # if network used to be DataParallel, now it's time to convert it back to DP
        if pa_0.__name__ == 'DataParallel':
            network = pa_0(network)
        elif pa_1.__name__ == 'DataParallel':
            network = pa_1(network)

    else: # if not given any network template

        checkpoint = torch.load(net_path)

        if 'net' in checkpoint:
            network = checkpoint['net']
            epoch = checkpoint['epoch']
            err = checkpoint['err']
            if 'opt' in checkpoint:
                optimizer = checkpoint['opt']
        else:
            network = checkpoint

    return network, epoch, err, optimizer


# ----------------------------------
# -------- Trainer Wrapper ---------
# ----------------------------------

class _trainer(object):
    def __init__(self, model, check_path, optimizer, resume=True, ft_path=None, **kwargs):
        super().__init__()

        self.bs = None # placeholder for batch_size
        self.pbar = None
        
        self.cp = check_path
        self.jlogs_path = self.cp.replace('model.pt','logs.json')
       
        self.global_step = 0

        if model:
            self.model, self.epoch, self.track_minimize, self.optimizer = model, 0, np.inf, None
        else:
            self.model, self.epoch, self.track_minimize, self.optimizer = load_network(model, self.cp)
        self.mu, self.Pc = self.model.mu, self.model.Pc
        
        # optimizer
        if self.optimizer is None:
            self.set_optimizer(optimizer)

        self.device = next(self.model.parameters()).device
    
        # tensorboard
        self.logger = SummaryWriter(self.cp+'-logs')
        
        # optional configs
        extract_args = lambda a, k: a if not k in kwargs else kwargs[k]
        self.l2 = extract_args( 0, 'l2' ) # l2 regularization
        self.ds = extract_args( np.inf, 'display_step' )
        self.gc = extract_args( None, 'gradient_clipping' )
        self.autostop = extract_args( np.inf, 'auto_stop' )
                    
        decay_step_ = extract_args( None, 'decay_step' )
        if decay_step_ is not None:
            steps, gamma = decay_step_
            if isinstance(steps, list):
                self.lr_scheduler = MultiStepLR(self.optimizer, milestones=steps, gamma=gamma)
            elif isinstance(steps, int):
                self.lr_scheduler = StepLR(self.optimizer, step_size=steps, gamma=gamma)
        else:
            self.lr_scheduler = None
        
        self.trackstop = 0
        self.stop = False
        self.logs = {}

        if ft_path:
            self._resume_other(ft_path)
            
        if resume:
            self._resume()
                
    #------------------------------      
    # JSON log files i/o
    #------------------------------
    def save_json_logs(self, desc=''):
        if len(self.logs):
            with open(self.jlogs_path+desc, 'w') as fp:
                json.dump(self.logs, fp)
            return True 
        return False    
            
    def load_json_logs(self, desc=''):
        if os.path.exists(self.jlogs_path):
            with open(self.jlogs_path+desc, 'r') as fp:
                self.logs = json.load(fp)
                return True
        return False
    
    #------------------------------
    # Model and checkpoints saving
    #------------------------------
    def _save_latest(self, desc=None):
        if not desc:
            desc = "-latest"
        else:
            desc = "-ep%d"%self.epoch
        # save latest checkpoint upon exit
        # ref : http://effbot.org/zone/stupid-exceptions-keyboardinterrupt.htm
        save_network(self.model, self.epoch, self.track_minimize, self.optimizer, self.cp+desc)
        sflag = self.save_json_logs()
        utils.print_update("Saving to {} ... (jlogs:{})".format(self.cp+desc, sflag), self.pbar)

    # Alias of _save_latest
    def save_checkpoint(self, desc=None):
        return self._save_latest(desc)

    # Save & restore model
    def save(self, epoch):
        # normal save
        save_network(self.model, epoch, self.track_minimize, self.optimizer, self.cp)
        utils.print_update("Saving to {} ...".format(self.cp), self.pbar)

    # Externally: restore the best epoch
    def restore(self):
        assert self.model is not None, "Must initialize a model!"
        if os.path.exists(self.cp):
            self.model, self.epoch, self.track_minimize, optimizer_loaded = load_network(self.model, self.cp)
            if optimizer_loaded is not None:
                self.optimizer = optimizer_loaded
            jflag = self.load_json_logs()   
            print(f"Loading from {self.cp} at epoch {self.epoch} (jlogs:{jflag}) ...")
            if self.lr_scheduler is not None:
                self.lr_scheduler._step_count = self.epoch

    # Internally: resume from the latest checkpoint. Update epoch and track_min values
    def _resume(self):
        res_path = self.cp+'-latest'
        if os.path.exists(res_path):
            self.model, self.epoch, self.track_minimize, optimizer_loaded = load_network(self.model, res_path)
            if optimizer_loaded is not None:
                self.optimizer = optimizer_loaded
            jflag = self.load_json_logs()   
            if self.lr_scheduler is not None:
                self.lr_scheduler._step_count = self.epoch
            print(f"Resuming training from {res_path} at epoch {self.epoch} (jlogs:{jflag}) ...")
        else:
            print("Starting fresh at epoch 0 ...")
         
    # Internally: reload model from some location. Keep fresh epoch and track_min values
    def _resume_other(self, res_path):
        if os.path.exists(res_path):
            self.model, old_epoch, _, _ = load_network(self.model, res_path)
            jflag = False#self.load_json_logs()   
            print(f"Finetuning from {res_path} (epoch:{old_epoch}); Starting at epoch {self.epoch} (jlogs:{jflag}) ...")
        else:
            raise ValueError("No prior model to resume! Check path or turn off FT mode. Exiting..")

    #-------------------------------------
    # Monotonic regularizer implementation
    #-------------------------------------
        
    # Data augmentation for monotonic regularizer
    def unsup_augmentation(self, x, additive, p_shape, cpu=False):
        
        # 1. raplce x new (interpolate)
        plin_new = 10**((torch.log10(x[:,-1])*10 + additive)/10)
        x_new = torch.cat((x[:,:-1], plin_new.view(-1,1)), 1)
        inputs = self.batch_data_handler(x_new)

        # 2. predict wsee
        y_pred_aug, _ = self.model(*inputs)
        y_pred_aug = y_pred_aug.view(p_shape)
        if not cpu:
            wsee_aug = utils.f_pointwise_ee_torch(y_pred_aug, x_new, self.mu, self.Pc)
        else:
            device = x.device
            wsee_aug = utils.f_pointwise_ee_torch(y_pred_aug.cpu(), x_new.cpu(), self.mu, self.Pc)
            wsee_aug = wsee_aug.to(device)
        
        return y_pred_aug, wsee_aug
    
    # Stand alone augmentation loss
    def _augmentation_loss(self, y_pred, x, wsee, cpu=False):
        additive = torch.empty(x.shape[0]).uniform_(-1 , 1).to(device)
        y_pred_new, wsee_new = self.unsup_augmentation( x, additive, y_pred.shape, cpu )
        wsee_loss_aug = -wsee_new.sum(1).mean()/ y_pred.shape[-1]
        return wsee_loss_aug
        
    def _monotonicity_loss(self, y_pred, x, wsee, factor=1e+3, cpu=False):
        additive = -1*torch.ones(x.shape[0]).to(self.device)
        with torch.no_grad():
            y_pred_new, wsee_new = self.unsup_augmentation( x, additive, y_pred.shape, cpu )
            wsee_bk = wsee_new.detach()
            y_bk = y_pred_new.detach()
            
        # 3. check monotonicity
        delta_wsee = wsee - wsee_bk.sum(1) # positive positions: wsee new < wsee old
        delta_pmax = torch.sign(additive) # positive positions: p_max new > p_max old
        
        # 4. scale with loss
        delta = delta_pmax*delta_wsee
        mask_incorr = delta>0 # where monotonicity is violated
        mask_incorr_increm = delta_wsee[mask_incorr]<0 # violations where wsee new > wsee old
        
        incorr = torch.mean((delta[mask_incorr][mask_incorr_increm])**2)
        
        pt_increm = torch.nn.functional.smooth_l1_loss(
            y_pred[mask_incorr][mask_incorr_increm], 
            y_bk[mask_incorr][mask_incorr_increm]
        )
        
        mono_loss = incorr + factor * pt_increm
                
        return mono_loss        
            
    #--------------------------------
    # Set which loss functions to use
    #--------------------------------
    def set_criteria(self, crit, cweights=None):
        # parse criteria, e.g. self.criterion = { 'mse':1, 'wsee':1, 'augm':1, 'mono':1 }
        cpool = ['mse', 'rmse', 'mae', 'huber', 'huberh', 'mseh', 'wsee', 'augm', 'mono']
        if isinstance(crit, str):
            crit = [ct.lower() for ct in crit.split('+')]
        elif isinstance(crit, list):
            crit = [ct.lower() for ct in crit]
        assert np.all([ct in cpool for ct in crit])
        
        if cweights is not None:
            wnorm = 1#sum(cweights)
            cweights = [w/wnorm for w in cweights]
        else:
            wnorm = len(crit)
            cweights = [1/wnorm for _ in crit]
        self.criterion = {k:v for k,v in zip(crit, cweights)}
        
    def compute_objective(self, y, x, cpu, func=utils.f_pointwise_ee_torch):
        if not cpu:
            uval_pt = func(y, x, self.mu, self.Pc)
        else:
            device = y_pred.device
            uval_pt = func(y.cpu(), x.cpu(), self.mu, self.Pc)
            uval_pt = uval_pt.to(device)
        return uval_pt    
    
    # loss and backprop step (in training and prediction)
    def loss_func(self, y_pred, y_true, x, weights=None, cpu=False):
                   
        sup, obj, mono = [torch.tensor(0.).to(self.device) for _ in range(3)] # scalar
        
        # unsupvised loss
        if 'wsee' in self.criterion:
            wsee_pt = self.compute_objective(y_pred, x, cpu, utils.f_pointwise_ee_torch)
            wsee = wsee_pt.sum(1)
            obj = -wsee.mean()/ y_pred.shape[-1]
            obj *= self.criterion['wsee']
        else:
            raise NotImplemented
            
        # mono penalty
        if 'mono' in self.criterion:# and self.model.training:
            assert 'wsee' in self.criterion
            mono = self._monotonicity_loss(y_pred, x, wsee, cpu)
            mono *= self.criterion['mono']
            
        # supervised loss
        sup_scale = 1 
        filtered = torch.any(y_true.isnan(),dim=1)     
        ytf, ypf = y_true[~filtered], y_pred[~filtered]
        
        if 'mse' in self.criterion and torch.any(~filtered):
            sup = sup_scale*((ypf - ytf)**2).mean() # scalar
            sup *= self.criterion['mse']
        elif 'mae' in self.criterion and torch.any(~filtered):
            sup = sup_scale*(torch.abs(ypf - ytf)).mean() # scalar
            sup *= self.criterion['mae']               
        elif 'rmse' in self.criterion and torch.any(~filtered):
            sup = sup_scale*((ypf - ytf)**2).mean()**.5 # scalar
            sup *= self.criterion['rmse']   
        elif 'huber' in self.criterion and torch.any(~filtered):   
            sup = sup_scale*torch.nn.functional.smooth_l1_loss(ypf, ytf)
            sup *= self.criterion['huber']
        elif 'huberh' in self.criterion and torch.any(~filtered):   
            wsee_supv = self.compute_objective(ytf, x[~filtered], cpu).sum(1)
            sup_filtered = (wsee_supv - wsee[~filtered])>0
            if torch.any(sup_filtered):
                sup = sup_scale*torch.nn.functional.smooth_l1_loss(ypf[sup_filtered], ytf[sup_filtered])
                sup *= self.criterion['huberh']
        elif 'mseh' in self.criterion and torch.any(~filtered):   
            wsee_supv = self.compute_objective(ytf, x[~filtered], cpu).sum(1)
            sup_filtered = (wsee_supv - wsee[~filtered])>0
            if torch.any(sup_filtered):
                sup = sup_scale*((ypf[sup_filtered] - ytf[sup_filtered])**2).mean()
                sup *= self.criterion['mseh']
            
        loss = sup + obj + mono
             
        l = loss.item()
        m = sup.item()
        w = wsee.mean().item()
                
        return loss, (l, m, w)
    
    #------------------------------------------------
    # Set the optimizer for training, simple or decay
    #------------------------------------------------        
    def set_optimizer_simple(self, configs):
        if configs is None or 'torch.optim' in str(type(configs)):
            self.optimizer = configs
        else:
            opt_inst, opt_config = configs
            self.optimizer = opt_inst(filter(lambda p: p.requires_grad, self.model.parameters()), **opt_config)

    def set_optimizer_decay(self, configs):
        if configs is None:
            pass
        
        elif 'torch.optim' in str(type(configs)):
            raise

        else:
            opt_inst, opt_config = configs
            module_names = self.model._modules.keys() 
            # confirm module_names are legal
            assert all([m in ['embedding', 'sca', 'final'] for m in module_names])
            param_groups = []

            if 'embedding' in module_names:
                cflag = True
                param_groups.append({'params':filter(lambda p: p.requires_grad, self.model.embedding.parameters()),
                                     'name': 'emb'})
            else:
                cflag = False

            if 'sca' in module_names:
                nblocks = len(self.model.sca)
                for t in range(nblocks):
                    param_groups.append({'params':filter(lambda p: p.requires_grad, self.model.sca[t].parameters()),
                                         'name': f'sca_{t}'})
            else:
                raise ValueError('The model must have a module named sca!')

            if 'final' in module_names:
                param_groups.append({'params':filter(lambda p: p.requires_grad, self.model.final.parameters()),
                                     'name': 'fin'})

            if cflag: # combine "emb" and "sca_0"; remove "emb"
                assert param_groups[1]['name']=='sca_0'
                param_groups[1]['params']=list(param_groups[0]['params'])+list(param_groups[1]['params'])
                param_groups.pop(0)

            self.optimizer = opt_inst(param_groups, **opt_config)     
            
    #---------------------------------------------
    # Optimizer step methods, simple or decay
    #---------------------------------------------
    def opt_step_handler_simple(self, loss):
        self.optimizer.zero_grad()
         
        # scalar loss (or vector loss)
        loss.backward() #(torch.ones(self.nu).to(self.device)) 
        
        # gradient clipping
        if self.gc: 
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)
        self.optimizer.step()   
        
    def opt_step_handler_decay(self, loss, decay):
        # decay : a list of learning rate decay rates
        decay = self.block_decay if decay is None else decay
        ng = len(self.optimizer.param_groups)
        decay = [decay]*ng if not hasattr(decay, "__len__") else decay
        
        # decay learning rate
        for g,d in zip(self.optimizer.param_groups, decay[:ng]):
            g['lr'] = self.lr_init * d
            
        self.opt_step_handler_simple(loss)
        
    #------------------------------
    # Data handling for training
    #------------------------------
    
    # Returns the numbers of nodes and steps
    def calc_numbers(self, loader):
        max_nodes = loader.dataset[0].x.shape[0]
        self.T = min(loader.dataset[0].y.shape[1], self.max_steps)
        self.bs = loader.batch_size
        n = max_nodes*self.bs*self.T
        return max_nodes, n
    
    # Return data loaders
    def get_loaders(self, dict_loaders):
        # get optimal wsee
        self.logs['wsee_opt'] = {k:utils.f_wsee_torch(*v[1::-1], self.mu, self.Pc, 'mean').item() for k,v in dict_loaders.items()}
        
        # get the loaders
        if 'simple' in self.__class__.__name__.lower():
            return dict_loaders
        elif 'seq' in self.__class__.__name__.lower():
            return dict_loaders
        else:
            return {k: DataLoader(TensorDataset(*v[:-1]), batch_size=v[-1], shuffle= k=='tr') for k,v in dict_loaders}      

    #--------------------------------------
    # Load data and labels, simple or graph
    #--------------------------------------
    def batch_data_handler_simple(self, data):
        if isinstance(data, list):
            [d.to(self.device) for d in data]
            for dd in range(len(data)): # update x steps
                data[dd].y = data[dd].y[...,:self.T]
            x_true = torch.cat([d.x for d in data],0)
            y_true = torch.cat([d.y for d in data],0)
            A_coo = torch.cat([d.edge_index for d in data],0)
        else:
            data.to(self.device)
            x_true = data.x
            y_true = data.y[...,:self.T]
            A_coo = data.edge_index

        # if 'CVAE', considering both parallel/nonparallel model cases
        try:
            assert 'CVAE' in self.model.__class__.__name__ or 'CVAE' in self.model.module.__class__.__name__
            return data, y_true, A_coo
        except:
            return x_true, y_true, A_coo
        
    def batch_data_handler_graph(self, data):
        edge_weight_batch = data[:,self.nu:-1].reshape(-1) 
        y_init =  data[:,:self.nu].reshape(-1,1) # inital signals (p_init)
        y_constr = data[:,:self.nu].reshape(-1,1) # upper power constraint (p_max)
        return (y_init, y_constr), self.edge_index_batch, edge_weight_batch
    
    #-------------------------------------------
    # Training data preperation, simple or graph
    #-------------------------------------------
    
    # Set configs; return x_train, y_train and index (permutated)
    def training_prep_simple(self, epoch, loader):
        torch.cuda.empty_cache()
        self.model.train() # set to train mode
        self.epoch = epoch
        self.global_step = epoch*len(loader) if not self.global_step else self.global_step
        
        # get number of users and batch size
        X_train, y_train, self.bs = loader
        if y_train is not None:
            ns, self.nu = y_train.shape
        else:
            ns = X_train.shape[0]
            self.nu = np.floor((X_train.shape[1]-1)**.5).astype(int)
        perm_i = np.random.permutation(ns)
        
        return X_train, y_train, perm_i  

    def training_prep_graph(self, epoch, loader):
        X_train, y_train, perm_i = self.training_prep_simple(epoch, loader)

        # edge index for mini batches
        self.edge_index_batch = self.edge_info_proc(self.ei, self.nu, self.bs)   
        
        return X_train, y_train, perm_i
    
    #------------------------
    # Performance evaluation
    #------------------------
    def calculate_metric(self, loader, metric='mse'):
        y_true, y_hat, A_coo = self.predict(self.epoch, criterion=None, loader=loader, save=False)

        if metric == 'mse':
            m = np.mean((y_true- y_hat)**2)
        elif metric == 'auc':
            from sklearn import metrics
            m = metrics.roc_auc_score(y_true.flatten(), y_hat.flatten())
        else:
            raise NotImplemented

        return m, (y_true, y_hat, A_coo)

    # Tensorboard logging
    def logging(self, info, phase):
        step = self.global_step*self.bs if phase.startswith('tr') else self.epoch
        
        # Log scalar values (scalar summary)
        for k,v in info.items():
            self.logger.add_scalar('%s/%s'%(k,phase), v, step)        
            
        if phase.startswith('tr'):
            # Log values and gradients of the parameters (histogram summary)
            for tag, value in self.model.named_parameters():
                if value.requires_grad:#'enc' in tag.lower() or 'lstm' in tag.lower():
                    try:
                        tag = '.'.join(tag.split('.')[:-1])+'/'+tag.split('.')[-1]
                        self.logger.add_histogram(tag, value.data.cpu().numpy(), step)
                        self.logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step)
                    except:
                        continue
   
    #-----------------------------------------------------------
    # Train and Predict methods, and other utility to be implemented
    #-----------------------------------------------------------
    def train(self):
        raise NotImplemented

    def predict(self):
        raise NotImplemented

    def batch_data_handler(self, data):
        return self.batch_data_handler_simple(data)
        
    def opt_step_handler(self, loss):
        return self.opt_step_handler_simple(loss)

    def set_optimizer(self, configs):
        return self.set_optimizer_simple(configs)

    def training_prep(self, epoch, loader):
        return self.training_prep_simple(epoch, loader)


class Simple_Trainer(_trainer):
    """
    load all data first; not using pytorch's dataloader
    """   
    def batch_data_handler(self, data):
        return [data]

    # returns the numbers 
    def calc_numbers(self, loader):
        self.nu = loader.dataset[0][1].shape[-1] # number of users
        self.bs = loader.batch_size # batch size

    # simple logging via dict: self.logs
    def logging(self, info, phase):
        pkey = '%s'%phase[:2]
        for k,v in info.items():
            if k not in self.logs:
                self.logs[k] = {}
            if pkey not in self.logs[k]:
                self.logs[k][pkey] = [v]
            else:
                self.logs[k][pkey].append(v)
                            
            if not len(self.logs[k][pkey]) < self.epoch+1:
                self.logs[k][pkey] = self.logs[k][pkey][:self.epoch+2]
            else:
                print(self.logs[k][pkey])
                raise ValueError

    # Train & test functions for a single epoch
    def train(self, epoch, loader, pbar=None, **kwargs):
        self.pbar = pbar # simple progress bar handling
        x_train, y_train, perm_i = self.training_prep(epoch, loader) # set train here
        
        # tracking loss and other stats
        running_loss, running_mse, running_wsee = 0,0,0
        for i in range(len(perm_i)//self.bs):
            i_s, i_e = i*self.bs, (i+1)*self.bs

            y_true = y_train[perm_i[i_s:i_e]]
            x = x_train[perm_i[i_s:i_e]]

            y_pred, gamma = self.model( x )     
            
            loss, (l,m,w) = self.loss_func(y_pred.view(y_true.shape), y_true, x, x[:,-1])
            self.opt_step_handler(loss)  

            running_loss += l
            running_mse += m
            running_wsee += w
            
            # update global step
            self.global_step += 1
            
        if self.lr_scheduler is not None:
            self.lr_scheduler.step() # _step_count

        # training logs
        trn_log = {'loss': running_loss/(i+1),
                   'mse': running_mse/(i+1),
                   'wsee': running_wsee/(i+1)}
        self.logging(trn_log, 'train')

        # if increment, reset loss etc. information
        if not self.epoch%self.ds:
            utils.print_update('==> Epoch %d: training avg '%self.epoch +
                  ', '.join(['{}: {:.6f}'.format(k,v) for k,v in trn_log.items()]), pbar)
            
    def predict(self, epoch, loader=None, save=False, pbar=None, return_serial=False, **kwargs):
        phase='val' if save else 'test'
        self.pbar = pbar
        
        x, y, self.bs = loader
        ns, self.nu = y.shape
        
        self.model.eval()
        # validation
        y_pred, gamma = self.model( x )
        yp = y_pred if gamma is None else y_pred[-1]
        ga = gamma if gamma is None else gamma[-1]
                            
        loss, (l,m,w) = self.loss_func(yp, y, x, x[:,-1])
        val_log = {'loss':l, 'mse':m, 'wsee':w} # wsee/mse loss

        self.logging(val_log, phase)
        if not self.epoch%self.ds:
            utils.print_update('==> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in val_log.items()]) , pbar)

        if save: 
            # if validation, save the model if it results in a new lowest error/loss
            track = val_log['loss'] #test_err
            if track < self.track_minimize:
                # update min err track
                self.trackstop = 0
                self.track_minimize = track
                self.save(epoch) # save model
            else:
                self.trackstop += 1
                if self.trackstop > self.autostop:
                    self.stop = True

        if return_serial:
            return (y_pred, gamma), y    
        else:
            return (yp, ga), y    
        
        
class Simple_Trainer_G(Simple_Trainer):
    """
    load all data first; not using pytorch's dataloader
    """
    def __init__(self, model, check_path, optimizer, resume=True, **kwargs):
        super(Simple_Trainer_G, self).__init__(model, check_path, optimizer, resume, **kwargs)
        extract_args = lambda a, k: a if not k in kwargs else kwargs[k]
        self.ft_flag = extract_args( False, 'ft_flag' ) 
        self.ei = self.model.ei

    def batch_data_handler(self, data):
        return self.batch_data_handler_graph(data)
    
    def training_prep(self, epoch, loader):
        return self.training_prep_graph(epoch, loader)

    # edge index for mini batches
    def edge_info_proc(self, edge_index, nu, bs):
        shift = torch.Tensor(
                np.array([np.arange(bs)*nu,]*nu**2).T.reshape(-1)
            ).repeat(1, 2).view(2,-1).long().to(edge_index.device) 
        edge_index_batch = edge_index.repeat(1, bs)+shift #edge_index_tr=edge_index_va
        return edge_index_batch        
        
    # Train & test functions for a single epoch
    def train(self, epoch, loader, pbar=None, **kwargs):
        self.pbar = pbar # simple progress bar handling
        x_train, y_train, perm_i = self.training_prep(epoch, loader) # set train here

        # tracking loss and other stats
        running_loss, running_mse, running_wsee = 0,0,0
        for i in range(len(perm_i)//self.bs):
            # get batch
            idx = perm_i[(i*self.bs) : ((i+1)*self.bs)]
            y_true = y_train[idx]
            x = x_train[idx]
            
            # format inputs and propogate forward
            inputs = self.batch_data_handler(x)
            y_pred, gamma = self.model(*inputs)
                        
            # training step back prop
            loss, (l,m,w) = self.loss_func(y_pred.view(y_true.shape), y_true, x, x[...,-1])

            self.opt_step_handler(loss)

            running_loss += l
            running_mse += m
            running_wsee += w
                        
            # update global step and lr scheduler step
            self.global_step += 1
            
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        # check if stuck at a local minimum; if yes, reset the parameters
        if not self.ft_flag and np.isclose(running_wsee, 0, rtol=1e-05, atol=1e-08, equal_nan=True):
            utils.print_update('WARNING: re-initializing the points...')
            utils.reset_model_parameters(self.model)

        # training logs
        try:
            trn_log = {'loss': running_loss/(i+1),
                       'mse': running_mse/(i+1),
                       'wsee': running_wsee/(i+1)}
        except:
            trn_log = {'loss': running_loss,
                       'mse': running_mse,
                       'wsee': running_wsee}            
        self.logging(trn_log, 'train')

        # if increment, reset loss etc. information
        if not self.epoch%self.ds:
            utils.print_update('==> Epoch %d: training avg '%self.epoch +
                  ', '.join(['{}: {:.6f}'.format(k,v) for k,v in trn_log.items()]), pbar)

            
    def predict(self, epoch, loader=None, save=False, pbar=None, **kwargs):
        # predict in single batch
        phase='val' if save else 'test'
        self.pbar = pbar
        self.model.eval()
        
        x, y, bs = loader
        n, nu = y.shape[0], y.shape[-1]
        
        # ------ validation # HERE ASSUME: SAME GRAPH TOPOLOGY!!! (self.ei DOES NOT CHANGE)
        self.nu = nu
        self.edge_index_batch = self.edge_info_proc(self.ei, nu, bs) 
        
        yp = []
        for i in range(int(np.ceil(n/bs))):
            i_s, i_e = i*bs, min((i+1)*bs, n)
            inputs = self.batch_data_handler(x[i_s: i_e])
            yp_, gamma = self.model(*inputs)        
            if isinstance(yp_, list):
                yp_ = yp_[-1]
            yp.append(yp_)
        yp = torch.cat(yp)
        
        _, (l,m,w) = self.loss_func(yp.view(y.shape), y, x, x[...,-1], cpu=False)
        val_log = {'loss':l, 'mse':m, 'wsee':w} # wsee/mse loss

        self.logging(val_log, phase)
        if not self.epoch%self.ds:
            utils.print_update('==> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in val_log.items()]) , pbar)

        if save: 
            # if validation, save the model if it results in a new lowest error/loss
            track = -val_log['wsee'] 
            if track < self.track_minimize:
                # update min err track
                self.trackstop = 0
                self.track_minimize = track
                self.save(epoch) # save model
            else:
                self.trackstop += 1
                if self.trackstop > self.autostop:
                    self.stop = True

        return yp, y
    

class Decay_Seq_Trainer(Simple_Trainer):
    """
    for sequentially training usca-mlp with blockwise learning rate decay
    """      
    def __init__(self, model, check_path, optimizer, resume=True, **kwargs):
        super(Decay_Seq_Trainer, self).__init__(model, check_path, optimizer, resume, **kwargs)
        extract_args = lambda a, k: a if not k in kwargs else kwargs[k]
        self.block_decay = extract_args( .6, 'block_decay' ) # lr decay block-wise
       
    def set_optimizer(self, configs):
        return self.set_optimizer_decay(configs)

    def opt_step_handler(self, loss, decay):
        return self.opt_step_handler_decay( loss, decay)   

    def requires_grad_blocks(self, bidx, requires_grad):
        #set blocks indexed by bidx non-trainable
        for i in bidx:
            if i==0: # embedding as the same
                try:
                    for parameter in self.model.embedding.parameters():
                        parameter.requires_grad = requires_grad
                except:
                    pass
                
            # set sca[i] require_grad False
            for parameter in self.model.sca[i].parameters():
                  parameter.requires_grad = requires_grad
    
    def train(self, epoch, b_curr, loader, pbar=None, **kwargs):
        self.pbar = pbar # simple progress bar handling
        x_train, y_train, perm_i = self.training_prep(epoch, loader) # set train here

        db = b_curr[1] - b_curr[0]
        ft_flag = True if db>1 else False
        
        # tracking loss and other stats
        running_loss, running_mse, running_wsee = 0,0,0
        for i in range(len(perm_i)//self.bs):
            
            # get batch
            idx = perm_i[(i*self.bs) : ((i+1)*self.bs)]
            y_true = y_train[idx]
            x = x_train[idx]
            
            # format inputs and propogate forward
            inputs = self.batch_data_handler(x)
            y_pred, gamma = self.model(*inputs, start=b_curr[0], end=b_curr[1])
                        
            # training step back prop
            loss, (l,m,w) = self.loss_func(y_pred.view(y_true.shape), y_true, x, x[...,-1])

            self.opt_step_handler(loss, decay=self.block_decay)

            running_loss += l
            running_mse += m
            running_wsee += w
            
            # update global step and lr scheduler step
            self.global_step += 1
            
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        # check if stuck at a local minimum; if yes, reset the parameters
        if np.isclose(running_wsee, 0, rtol=1e-05, atol=1e-08, equal_nan=True):
            utils.print_update('WARNING: re-initializing the points...')
            utils.reset_model_parameters(self.model)

        # training logs
        trn_log = {'loss': running_loss/(i+1),
                   'mse': running_mse/(i+1),
                   'wsee': running_wsee/(i+1)}
        self.logging(trn_log, 'train')

        # if increment, reset loss etc. information
        if not self.epoch%self.ds:
            utils.print_update('==> Epoch %d: training avg '%self.epoch +
                  ', '.join(['{}: {:.6f}'.format(k,v) for k,v in trn_log.items()]), pbar)    
            
            
    def predict(self, epoch, b_curr, loader=None, save=False, pbar=None, **kwargs):
        # predict in single batch
        phase='val' if save else 'test'
        self.pbar = pbar
        self.model.eval()
        
        x, y, bs = loader
        n, nu = y.shape[0], y.shape[-1]
        
        yp = []
        for i in range(int(np.ceil(n/bs))):
            i_s, i_e = i*bs, min((i+1)*bs, n)
            inputs = self.batch_data_handler(x[i_s: i_e])
            yp_, gamma = self.model(*inputs, start=b_curr[0], end=b_curr[1])        
            if isinstance(yp_, list):
                yp_ = yp_[-1]
            yp.append(yp_)
        yp = torch.cat(yp)
        
        _, (l,m,w) = self.loss_func(yp.view(y.shape), y, x, x[...,-1], cpu=False)
        val_log = {'loss':l, 'mse':m, 'wsee':w} # wsee/mse loss

        self.logging(val_log, phase)
        if not self.epoch%self.ds:
            utils.print_update('==> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in val_log.items()]) , pbar)

        if save: 
            # if validation, save the model if it results in a new lowest error/loss
            track = -val_log['wsee'] #test_err
            if track < self.track_minimize:
                # update min err track
                self.trackstop = 0
                self.track_minimize = track
                self.save(epoch) # save model
            else:
                self.trackstop += 1
                if self.trackstop > self.autostop:
                    self.stop = True
        return yp, y    
      
        
class Decay_Seq_Trainer_G(Simple_Trainer_G):
    """
    for sequentially training usca-gcn with blockwise learning rate decay
    """
    def __init__(self, model, check_path, optimizer, resume=True, **kwargs):
        super(Decay_Seq_Trainer_G, self).__init__(model, check_path, optimizer, resume, **kwargs)
        extract_args = lambda a, k: a if not k in kwargs else kwargs[k]
        self.block_decay = extract_args( .6, 'block_decay' ) # lr decay block-wise
        self.weight_sharing = False
        self.reset_count = 0

    def batch_data_handler(self, data):
        return self.batch_data_handler_graph(data)

    def set_optimizer(self, configs):
        return self.set_optimizer_decay(configs)

    def opt_step_handler(self, loss, decay):
        return self.opt_step_handler_decay( loss, decay)   
    
    def training_prep(self, epoch, loader):
        return self.training_prep_graph(epoch, loader)
       
    def requires_grad_blocks(self, bidx, requires_grad):
        #set blocks indexed by bidx non-trainable
        for i in bidx:
            if i==0: # embegging as the same
                try:
                    for parameter in self.model.embedding.parameters():
                        parameter.requires_grad = requires_grad
                except:
                    pass
                
            # set sca[i] require_grad False
            for parameter in self.model.sca[i].parameters():
                  parameter.requires_grad = requires_grad
            
    # simple logging via dict: self.logs
    def logging(self, info, phase):
        pkey = '%s'%phase[:2]
        for k,v in info.items():
            if k not in self.logs:
                self.logs[k] = {}
            if pkey not in self.logs[k]:
                self.logs[k][pkey] = [v]
            else:
                self.logs[k][pkey].append(v)
                            
            if not len(self.logs[k][pkey]) < self.epoch+1:
                self.logs[k][pkey] = self.logs[k][pkey][:self.epoch+2]
            else:
                raise ValueError            
                                
    def train(self, epoch, b_curr, loader, pbar=None, **kwargs):
        self.pbar = pbar # simple progress bar handling
        x_train, y_train, perm_i = self.training_prep(epoch, loader) # set train here
        
        db = b_curr[1] - b_curr[0]
        ft_flag = True if db>1 and not self.weight_sharing else False
        
        # tracking loss and other stats
        running_loss, running_mse, running_wsee = 0,0,0
        for i in range(len(perm_i)//self.bs):
            
            # get batch
            idx = perm_i[(i*self.bs) : ((i+1)*self.bs)]
            y_true = y_train[idx]
            x = x_train[idx]
            
            # format inputs and propogate forward
            inputs = self.batch_data_handler(x)
            y_pred, gamma = self.model(*inputs, start=b_curr[0], end=b_curr[1])
                        
            # training step back prop
            loss, (l,m,w) = self.loss_func(y_pred.view(y_true.shape), y_true, x, x[...,-1])
            self.opt_step_handler(loss, decay=self.block_decay)

            running_loss += l
            running_mse += m
            running_wsee += w
            
            # update global step and lr scheduler step
            self.global_step += 1
            
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        # check if stuck at a local minimum; if yes, reset the parameters
        if b_curr[1]-1 ==0 and np.isclose(running_wsee, 0, rtol=1e-05, atol=1e-06, equal_nan=True):
            utils.print_update('WARNING: re-initializing the points...')
            utils.reset_model_parameters(self.model)
            
        # training logs
        trn_log = {'loss': running_loss/(i+1),
                   'mse': running_mse/(i+1),
                   'wsee': running_wsee/(i+1)}
        self.logging(trn_log, 'train')
        
        # if stuck, reset the parameters
        if not ft_flag and self.reset_count < 3 and len(self.logs['loss']['tr']) < 5 and len(self.logs['loss']['tr']) > 2:
            tr_progress = self.logs['loss']['tr'][-3:]
            if np.isclose(tr_progress[0], tr_progress[1], rtol=1e-05, atol=1e-08, equal_nan=True) and np.isclose(tr_progress[0], tr_progress[2], rtol=1e-05, atol=1e-08, equal_nan=True) :
                utils.print_update(f'WARNING: re-initializing the points in layer sca{b_curr[1]-1}...')
                utils.reset_model_parameters(self.model.sca[b_curr[1]-1])
                self.reset_count += 1

        # if increment, reset loss etc. information
        if not self.epoch%self.ds:
            utils.print_update('==> Epoch %d: training avg '%self.epoch +
                  ', '.join(['{}: {:.6f}'.format(k,v) for k,v in trn_log.items()]), pbar)    
            
                
    def predict(self, epoch, b_curr, loader=None, save=False, pbar=None, **kwargs):
        # predict in single batch
        phase='val' if save else 'test'
        self.pbar = pbar
        self.model.eval()
        
        x, y, bs = loader
        n, nu = y.shape[0], y.shape[-1]
        
        # ------ validation # HERE ASSUME: SAME GRAPH TOPOLOGY!!! (self.ei DOES NOT CHANGE)
        self.nu = nu
        self.edge_index_batch = self.edge_info_proc(self.ei, nu, bs) 
        
        yp = []
        for i in range(int(np.ceil(n/bs))):
            i_s, i_e = i*bs, min((i+1)*bs, n)
            inputs = self.batch_data_handler(x[i_s: i_e])
            yp_, gamma = self.model(*inputs, start=b_curr[0], end=b_curr[1])        
            if isinstance(yp_, list):
                yp_ = yp_[-1]
            yp.append(yp_)
        yp = torch.cat(yp)
        
        _, (l,m,w) = self.loss_func(yp.view(y.shape), y, x, x[...,-1], cpu=False)
        val_log = {'loss':l, 'mse':m, 'wsee':w} # wsee/mse loss

        self.logging(val_log, phase)
        if not self.epoch%self.ds:
            utils.print_update('==> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in val_log.items()]) , pbar)

        if save: 
            # if validation, save the model if it results in a new lowest error/loss
            track = -val_log['wsee'] 
            if track < self.track_minimize:
                # update min err track
                self.trackstop = 0
                self.track_minimize = track
                self.save(epoch) # save model
            else:
                self.trackstop += 1
                if self.trackstop > self.autostop:
                    self.stop = True

        return yp, y 
    
    
class DecayWS_Seq_Trainer(Decay_Seq_Trainer):
    """
    for sequentially training usca-mlp with blockwise learning rate decay
    """ 
    def opt_step_handler(self, loss, decay):
        decay = self.block_decay if decay is None else decay
        # decay learning rate
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr_init * decay
        self.opt_step_handler_simple(loss)
        
    
class DecayWS_Seq_Trainer_G(Decay_Seq_Trainer_G):
    """
    for sequentially training usca-gcn with blockwise learning rate decay
    """
    def __init__(self, model, check_path, optimizer, resume=True, **kwargs):
        super(DecayWS_Seq_Trainer_G, self).__init__(model, check_path, optimizer, resume, **kwargs)
        self.weight_sharing = True
    
    def opt_step_handler(self, loss, decay):
        decay = self.block_decay if decay is None else decay
        # decay learning rate
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr_init * decay
        self.opt_step_handler_simple(loss)