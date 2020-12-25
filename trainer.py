import torch
torch.backends.cudnn.benchmark =True
# torch.manual_seed(42)
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
# PyTorch logger tutorial: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

from utils import *#f_wsee_torch

# Save model with epoch number and validation error
def save_network(model, epoch, err, save_path):
    # deal with multi-gpu parallelism
    pa = type(model)
    if pa.__name__ == 'DataParallel':
        net_save = model.module
    else:
        net_save = model
    # format to save
    state = {
        'net': net_save,
        'err': err, # error rate or negative loss (something to minimize)
        'epoch': epoch
    }
    torch.save(state, save_path)


# Load model with epoch number and validation error
def load_network(network, net_path):

    if network: # if the given template network is not none

        check_pt = torch.load(net_path)

        if 'net' in check_pt:
            checkpoint = check_pt['net']
            epoch = check_pt['epoch']
            err = check_pt['err']
        else:
            checkpoint = check_pt
            epoch = -1
            err = np.nan

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

        return network, epoch, err

    else: # if not given any network template

        checkpoint = torch.load(net_path)

        if 'net' in checkpoint:
            network = checkpoint['net']
            epoch = checkpoint['epoch']
            err = checkpoint['err']
        else:
            network = checkpoint
            epoch = -1
            err = np.nan

        return network, epoch, err


# ----------------------------------
# -------- Trainer Wrapper ---------
# ----------------------------------

class _trainer(object):
    def __init__(self, model, check_path, optimizer, resume=True, **kwargs):
        super().__init__()

        self.bs = None # placeholder for batch_size
        self.pbar = None
        
        self.cp = check_path
        self.jlogs_path = os.path.join(os.path.dirname(self.cp),'logs.json')
        
        self.optimizer = optimizer
        self.global_step = 0

        if model:
            self.model, self.epoch, self.track_minimize = model, 0, np.inf
        else:
            self.model, self.epoch, self.track_minimize = load_network(model, self.cp)
        self.mu, self.Pc = self.model.mu, self.model.Pc

#         print(self.model)
        self.device = next(self.model.parameters()).device
    
        # tensorboard
        self.logger = SummaryWriter(self.cp+'-logs')
        
        # optional configs
        #self.l2 = 'l2' in kwargs and nn['l2'] or 0
        self.l2 = kwargs['l2'] if 'l2' in kwargs.keys() else 0 # l2 regularization
        self.ds = kwargs['display_step'] if 'display_step' in kwargs.keys() else np.inf
        self.gc = kwargs['gradient_clipping'] if 'gradient_clipping' in kwargs.keys() else None 
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=kwargs['decay_step'][0], gamma=kwargs['decay_step'][1]) \
            if 'decay_step' in kwargs.keys() and kwargs['decay_step'] else None # learning rate scheduler
        self.autostop = kwargs['auto_stop'] if 'auto_stop' in kwargs.keys() else np.inf
        
        self.trackstop = 0
        self.stop = False
        self.logs = {}

        if resume:
            self._resume()
            
    # ------  simple json logs handling ------     
    
    def save_json_logs(self):
        if len(self.logs):
            with open(self.jlogs_path, 'w') as fp:
                try:
                    json.dump(self.logs, fp)
                    return True
                except:
                    return False  
        return False    
            
    def load_json_logs(self):
        if os.path.exists(self.jlogs_path):
            with open(self.jlogs_path, 'r') as fp:
                try:
                    self.logs = json.load(fp)
                except:
                    return False
        return False
        
    # ----------------------------------------
    
    def _save_latest(self, desc=None):
        if not desc:
            desc = "-latest"
        else:
            desc = "-ep%d"%self.epoch
        # save latest checkpoint upon exit
        # http://effbot.org/zone/stupid-exceptions-keyboardinterrupt.htm
        save_network(self.model, self.epoch, self.track_minimize, self.cp+desc)
        sflag = self.save_json_logs()
        print_update("Saving to {} ... (jlogs:{})".format(self.cp+desc, sflag), self.pbar)

    # alias of _save_latest
    def save_checkpoint(self, desc=None):
        return self._save_latest(desc)

    # Save & restore model
    def save(self, epoch):
        # normal save
        save_network(self.model, epoch, self.track_minimize, self.cp)
        print_update("Saving to {} ...".format(self.cp), self.pbar)

    # Externally: restore the best epoch
    def restore(self):
        assert self.model is not None, "Must initialize a model!"
        if os.path.exists(self.cp):
            self.model, self.epoch, self.track_minimize = load_network(self.model, self.cp)
            jflag = self.load_json_logs()   
            print(f"Loading from {self.cp} at epoch {self.epoch} (jlogs:{jflag}) ...")

    # Internally: resume from the latest checkpoint
    def _resume(self):
        res_path = self.cp+'-latest'
        if os.path.exists(res_path):
            self.model, self.epoch, self.track_minimize = load_network(self.model, res_path)
            jflag = self.load_json_logs()   
            print(f"Resuming training from {res_path} at epoch {self.epoch} (jlogs:{jflag}) ...")
        else:
            print("Starting fresh at epoch 0 ...")

    # get data loaders
    def get_loaders(self, dict_loaders):
        # get optimal wsee
        self.logs['wsee_opt'] = {k:f_wsee_torch(*v[1::-1], self.mu, self.Pc, 'mean').item() for k,v in dict_loaders.items()}
        
        # get 'loaders'
        if 'simple' in self.__class__.__name__.lower():
            return dict_loaders
        else:
            return {k: DataLoader(TensorDataset(*v[:-1]), batch_size=v[-1], shuffle= k=='tr') for k,v in dict_loaders}            
    def train(self):
        raise NotImplemented

    def predict(self):
        raise NotImplemented
        
    def add_reg_l2(self):
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        return self.l2 * l2_reg        

    
    # Evaluate performance
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


    # tensorboard logging
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

    # load data and labels
    def data_handler(self, data):
        if isinstance(data, list):
            [d.to(self.device) for d in data]
            for dd in range(len(data)): # update x steps
                data[dd].y = data[dd].y[...,:self.T]
            x_true = torch.cat([d.x for d in data],0)
            y_true = torch.cat([d.y for d in data],0)
            A_coo = torch.cat([d.edge_index for d in data],0)
        else:
            data.to(self.device)
            data.y = data.y
            x_true = data.x
            y_true = data.y[...,:self.T]
            A_coo = data.edge_index

        # if 'CVAE', considering both parallel/nonparallel model cases
        try:
            assert 'CVAE' in self.model.__class__.__name__ or 'CVAE' in self.model.module.__class__.__name__
            return data, y_true, A_coo
        except:
            return x_true, y_true, A_coo

    # returns the numbers of nodes and steps.
    def calc_numbers(self, loader):
        max_nodes = loader.dataset[0].x.shape[0]
        self.T = min(loader.dataset[0].y.shape[1], self.max_steps)
        self.bs = loader.batch_size
        n = max_nodes*self.bs*self.T
        return max_nodes, n


# ----------------------------------------------
# ---- USCA MLP Trainer (classification) ----
# ----------------------------------------------

class GCVAE_Trainer(_trainer):

    # Train & test functions for a single epoch

    def train(self, epoch, criterion, loader):
        torch.cuda.empty_cache()

        # get nodes and steps
        max_nodes , n = self.calc_numbers(loader)

        # training logs
        trn_log = {'loss_%s'%k:[] for k in criterion.keys()} # kld/bce/mse loss
        trn_log['loss'] = [] # entire loss

        self.epoch = epoch
        self.global_step = epoch*len(loader) if not self.global_step else self.global_step

        # set to train mode
        self.model.train()

        with tqdm(loader, desc='Epoch#%d:'%epoch) as pbar:

            count_b = 0

            for data in pbar:
                if len(data) <= 1:
                    continue # for batchnorm the batchsize has to be greater than 1

                self.optimizer.zero_grad()

                data, y_true, A_coo = self.data_handler(data)

                # p (prior) and q (post) are multi-var gaussian distributions
                y_hat, (mu_q, sigma_q, mu_p, sigma_p)= self.model(data)

                kld  = criterion['kld'](mu_q, sigma_q, mu_p, sigma_p)/n
                bce  = criterion['bce'](y_hat, y_true)/n
                rule = criterion['rule'](y_hat, A_coo, num_nodes=max_nodes)/n
                loss = bce + kld + rule

                if self.l2: # l2 regularization
                    l2_reg = torch.tensor(0.).to(self.device)
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)
                    loss += self.l2 * l2_reg

                loss.backward()

                if self.gc is not None: # gradient clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)
                self.optimizer.step()

                trn_log['loss'] += [loss.item()]
                trn_log['loss_bce'] += [bce.item()]
                trn_log['loss_kld'] += [kld.item()]
                trn_log['loss_rule'] += [rule.item()]

                count_b += 1

                if not (self.global_step) % self.ds:
                    # record the last step
                    info = {k:v[-1] for k,v in trn_log.items() if v}
                    self.logging(info, 'train')

                    pbar.write('Train step {} at epoch {}: '.format(self.global_step, self.epoch) +
                               ', '.join(['{}: {:.6f}'.format(k, np.mean(v[-count_b:])) for k,v in trn_log.items()]))

                    count_b = 0

                # update global step
                self.global_step += 1

        # if increment, reset loss etc. information
        print('====> Epoch %d: Average training '%self.epoch +
              ', '.join(['{}: {:.6f}'.format(k, np.sum(v)/len(loader)) for k,v in trn_log.items()]))


    def predict(self, epoch, criterion=None, loader=None, save=False):

        # has to be a valid dataloader
        assert loader is not None

        # calculate n from dataloader
        max_nodes , n = self.calc_numbers(loader)

        phase='Val' if save else 'Test'

        self.model.eval()

        # prediction
        y_pr, y_gt, A_eidx = [],[],[]

        # track the loss
        if criterion:
            trn_log = {'loss_%s'%k:0 for k in criterion.keys()} # kld/bce/mse loss
            trn_log['loss'] = 0

        with torch.no_grad():
            with tqdm(loader, disable=1==len(loader)) as pbar:
                for data in pbar:

                    data, y_true, A_coo = self.data_handler(data)

                    # p (prior) and q (post) are multi-var gaussian distributions

                    y_hat, (mu_q, sigma_q, mu_p, sigma_p)= self.model(data)

                    # append true and predicted labels to lists
                    y_pr.append(y_hat.cpu().numpy().reshape((-1, max_nodes, self.T)))
                    y_gt.append(y_true.cpu().numpy().reshape((-1, max_nodes, self.T)))
                    A_eidx.append(A_coo.cpu().numpy())


                    if criterion: # if given loss functions

                        kld = criterion['kld'](mu_q, sigma_q, mu_p, sigma_p)/n
                        bce = criterion['bce'](y_hat, y_true)/n
                        rule = criterion['rule'](y_hat, A_coo, num_nodes=max_nodes)/n

                        loss = bce + kld + rule

                        if self.l2: # l2 regularization
                            l2_reg = torch.tensor(0.).to(self.device)
                            for param in self.model.parameters():
                                l2_reg += torch.norm(param)
                            loss += self.l2 * l2_reg

                        trn_log['loss'] += loss.item()
                        trn_log['loss_bce'] += bce.item()
                        trn_log['loss_kld'] += kld.item()
                        trn_log['loss_rule'] += rule.item()


            y_pr = np.concatenate(y_pr)
            y_gt = np.concatenate(y_gt)

            # if criterion is given, we can track loss and save if best and needed;
            # if criterion is not given, just return the true and predicted values
            if criterion:

                info = {k:v/len(loader) for k,v in trn_log.items()  if v}

                print('====> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in info.items()]) )

                if save: # if validation, save the model if it results in a new lowest error/loss
                    self.logging(info, 'val')

                    track = info['loss'] #test_err
                    if track < self.track_minimize:
                        # update min err track
                        self.trackstop = 0
                        self.track_minimize = track

                        self.save(epoch) # save model
                    else:
                        self.trackstop += 1
                        if self.trackstop > self.autostop:
                            self.stop = True

                else: # if test, do not save
                    pass

            return y_gt, y_pr, A_eidx


        
        
class Simple_Trainer(_trainer):
    """
    load all data first; not using pytorch's dataloader
    """
    
    # returns the numbers 
    def calc_numbers(self, loader):
        self.nu = loader.dataset[0][1].shape[-1] # number of users
        self.bs = loader.batch_size # batch size
    
    # load data and labels
    def data_handler(self, data):
        return data

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
            assert len(self.logs[k][pkey]) == self.epoch+1

    # Train & test functions for a single epoch
    def train(self, epoch, criterion, loader, pbar=None):
        torch.cuda.empty_cache()
        
        # simple progress bar handling
        self.pbar = pbar
        
        self.epoch = epoch
        self.global_step = epoch*len(loader) if not self.global_step else self.global_step
        
        # set to train mode
        self.model.train()
        
        # get number of users and batch size
        X_train, y_train, self.bs = loader
        self.nu = y_train.shape[-1]
        perm_i = np.random.permutation(y_train.shape[0])

        # tracking loss and other stats
        running_loss, running_mse, running_wsee = 0,0,0
        for i in range(len(perm_i)//self.bs):
            i_s, i_e = i*self.bs, (i+1)*self.bs

            y_true = y_train[perm_i[i_s:i_e]]
            x = X_train[perm_i[i_s:i_e]]

            y_pred, gamma = self.model(x)

            mse = torch.mean((y_pred-y_true)**2, dim=0)
            wsee = f_wsee_torch(y_pred, x, self.mu, self.Pc, 'vector')

            loss = 0.
            if 'mse' in criterion:
                loss += mse
            if 'wsee' in criterion:
                loss -= wsee
            if self.l2: # l2 regularization
                loss += self.add_reg_l2()

            self.optimizer.zero_grad()
            loss.backward(torch.ones(self.nu).to(self.device)) #vector loss

            if self.gc: # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)
            self.optimizer.step()    

            running_loss += torch.mean(loss).item()
            running_mse += torch.mean(mse).item()
            running_wsee += torch.sum(wsee).item()
            
            # update global step
            self.global_step += 1

        # training logs
        trn_log = {'loss': running_loss/(i+1),
                   'mse': running_mse/(i+1),
                   'wsee': running_wsee/(i+1)}
        self.logging(trn_log, 'train')

        # if increment, reset loss etc. information
        print_update('==> Epoch %d: training avg '%self.epoch +
              ', '.join(['{}: {:.6f}'.format(k,v) for k,v in trn_log.items()]), pbar)

            
    def predict(self, epoch, criterion=None, loader=None, save=False, pbar=None, return_serial=False):
        phase='val' if save else 'test'
        self.pbar = pbar
        
        X, y, self.bs = loader
        self.nu = y.shape[-1]
        
        self.model.eval()
        # validation
        y_p, gamma = self.model( X )        
        
        if criterion:
            mse = torch.mean((y_p[-1]-y)**2).item()
            wsee = f_wsee_torch(y_p[-1], X, self.mu, self.Pc, 'mean').item()
        
            val_log = {'loss': 0} # wsee/mse loss
            if 'mse' in criterion:
                val_log['mse'] = mse
                val_log['loss'] += mse
            if 'wsee' in criterion:
                val_log['wsee'] = wsee
                val_log['loss'] -= wsee/self.nu
            if self.l2: # l2 regularization
                val_log['loss'] += self.add_reg_l2()            

            self.logging(val_log, phase)
            print_update('==> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in val_log.items()]) , pbar)
            
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
                return (y_p, gamma), y    
            else:
                return (y_p[-1], gamma[-1]), y    
        
        
        
# ---------------------------
# ---- USCA MLP Trainer  ----
# ---------------------------

class MLP_Trainer(_trainer):
    
    # returns the numbers 
    def calc_numbers(self, loader):
        self.nu = loader.dataset[0][1].shape[-1] # number of users
        self.bs = loader.batch_size # batch size
    
    # load data and labels
    def data_handler(self, data):
        return data

    # Train & test functions for a single epoch
    def train(self, epoch, criterion, loader, pbar=None):
        torch.cuda.empty_cache()
        
        # get number of users and batch size
        self.calc_numbers(loader)
        self.ds = min(len(loader),self.ds)
    
        # training logs
        trn_log = {'%s'%k:[] for k in criterion} # wsee/mse loss
        trn_log['loss'] = [] # entire loss

        self.epoch = epoch
        self.global_step = epoch*len(loader) if not self.global_step else self.global_step

        # set to train mode
        self.model.train()

        # handle progress bar
        if pbar is None:
            pb_handler = (trange(len(loader), desc='Epoch#%d:'%epoch), True)
        else:
            pbar.set_description('Epoch#%d:'%epoch)
            pb_handler = (pbar, False)
        self.pbar = pb_handler[0]

        count_b = 0
        for data in loader:
#                 if len(data) <= 1:
#                     continue # for batchnorm the batchsize has to be greater than 1

            self.optimizer.zero_grad()

            x, y_true = self.data_handler(data)

            y_pred, gamma = self.model(x)

            mse = torch.mean((y_pred-y_true)**2, dim=0)
            wsee = f_wsee_torch(y_pred, x, self.mu, self.Pc, 'vector')

            loss = 0.
            if 'mse' in criterion:
                loss += mse
            if 'wsee' in criterion:
                loss -= wsee
            if self.l2: # l2 regularization
                loss += self.add_reg_l2()

            loss.backward(torch.ones(self.nu).to(self.device)) #vector loss

            if self.gc is not None: # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)
            self.optimizer.step()

            trn_log['loss'] += [torch.mean(loss).item()]
            trn_log['mse'] += [torch.mean(mse).item()]
            trn_log['wsee'] += [torch.sum(wsee).item()]

            count_b += 1
            
            if not (1+self.global_step) % self.ds:
                # record the last step
                info = {k:v[-1] for k,v in trn_log.items() if v}
                self.logging(info, 'train')
                
                print_update('Train step {} at epoch {}: '.format(self.global_step, self.epoch) +
                           ', '.join(['{}: {:.6f}'.format(k, np.mean(v[-count_b:])) for k,v in trn_log.items()]),
                            pb_handler[0])
                pb_handler[0].update(count_b)
                
                count_b = 0

            # update global step
            self.global_step += 1

        # if increment, reset loss etc. information
        print_update('====> Epoch %d: Average training '%self.epoch +
              ', '.join(['{}: {:.6f}'.format(k, np.sum(v)/len(loader)) for k,v in trn_log.items()]), pb_handler[0])

        if pb_handler[-1]:
            pb_handler[0].close()
            
    def predict(self, epoch, criterion=None, loader=None, save=False, pbar=None):
        
        # has to be a valid dataloader
        assert loader is not None

        # calculate n from dataloader
        self.calc_numbers(loader)
        phase='Val' if save else 'Test'

        # track the loss
        if criterion:
            trn_log = {'%s'%k:0 for k in criterion} # wsee/mse loss
            trn_log['loss'] = 0

        # prediction
        y_pr, y_gt = [],[]

        self.model.eval()
        with torch.no_grad():
            for data in loader:
                x, y_true = self.data_handler(data)
                yp, gamma = self.model( x )
                
                y_pr.append(yp[-1])
                y_gt.append(y_gt)
                
                mse = torch.mean((yp[-1]-y_true)**2).item()
                wsee = f_wsee_torch(yp[-1], x, self.mu, self.Pc, 'mean').item()

                if criterion: # if given loss functions
                    trn_log['mse'] += mse
                    trn_log['wsee'] += wsee
                    if 'mse' in criterion:
                        trn_log['loss'] += mse
                    if 'wsee' in criterion:
                        trn_log['loss'] -= wsee/self.nu
                    if self.l2: # l2 regularization
                        trn_log['loss'] += self.add_reg_l2()

            y_pr = np.concatenate(y_pr)
            y_gt = np.concatenate(y_gt)

            # if criterion is given, we can track loss and save if best and needed;
            # if criterion is not given, just return the true and predicted values
            if criterion:

                info = {k:v/len(loader) for k,v in trn_log.items()  if v}

                print_update('====> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in info.items()]) , pbar)

                if save: # if validation, save the model if it results in a new lowest error/loss
                    self.logging(info, 'val')

                    track = info['loss'] #test_err
                    if track < self.track_minimize:
                        # update min err track
                        self.trackstop = 0
                        self.track_minimize = track

                        self.save(epoch) # save model
                    else:
                        self.trackstop += 1
                        if self.trackstop > self.autostop:
                            self.stop = True

                else: # if test, do not save
                    self.logging(info, 'test')

            return y_gt, y_pr
        