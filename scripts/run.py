import os
import h5py
import shutil
import itertools as it
# from copy import deepcopy
from tqdm.auto import trange,tqdm
from sklearn.model_selection import StratifiedKFold

import numpy as np
# import cvxpy as cp
# import networkx as nx
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# SET YOUR PROJECT ROOT DIR HERE
PROJ_RT = "/root/deep-EE-opt-master/src/SCA/"
import sys; sys.path.append(PROJ_RT)

# from sca import *
from utils import *
import dataset as ds
from models_uf import USCA_MLP, USCA_MLP_R
from trainer import MLP_Trainer, Simple_Trainer


device = torch.device('cpu')

num_ue = 4
in_size = num_ue**2 + num_ue + 1
out_size = num_ue

architect = {'h_sizes': [128, 64, 32, 32, 32, 16, 8], 'activs': ['elu', 'relu', 'elu', 'relu', 'elu', 'relu',  'elu']}
architect = {'h_sizes': [32, 16, 16, 16, 16, 16, 8], 'activs': ['elu', 'relu', 'elu', 'relu', 'elu', 'relu',  'elu']}

k_fold = 2
bs = 512
num_l = 3 
learning_rate = 0.001
dropout = 0
l2 = 0
epochs = 200
save_freq = 50

init = 'full'
rseed= 42
loss_which= ['mse','wsee']
inner_optim='learned-mlp' 

models = [USCA_MLP] # USCA_MLP_R
trainers = [Simple_Trainer]


#--- data---

dfn_tr = PROJ_RT+'../../data/results_hataUrban_noSF.h5' # train / validation data
X_tr, y_tr, cinfo_tr = ds.load_data(dfn_tr)
print(X_tr.shape, y_tr.shape)

dfn_te = PROJ_RT+'../../data/results_hataUrban.h5' # test data
X_te, y_te, cinfo_te = ds.load_data(dfn_te)
print(X_te.shape, y_te.shape)

# to torch tensor
y_tr = torch.from_numpy(y_tr).float().to(device)
y_te = torch.from_numpy(y_te).float().to(device)

# add initial pt (max)
X_tr_ = torch.from_numpy(np.hstack((init_p(X_tr[:,-1], num_ue, method=init), X_tr))).float().to(device)
X_te_ = torch.from_numpy(np.hstack((init_p(X_te[:,-1], num_ue, method=init), X_te))).float().to(device)

attach_pt = lambda x: torch.from_numpy(np.hstack((init_p(x[:,-1], num_ue, method=init), x))).float().to(device)
X_tr_ = attach_pt(X_tr)
X_te_ = attach_pt(X_te)

for MODEL, TRAINER in zip(models, trainers):
    
    tr_loss, va_loss, te_loss = {},{},{}
    tr_wsee, va_wsee, te_wsee = {},{},{}
    wsee_opt = {}

    # k fold cross validation
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=rseed)
    
    for k, (train_index, valid_index) in enumerate(skf.split(X_tr, np.ones(y_tr.shape[0], dtype=bool))):

        # data loaders
        loaders_dict = {'tr': (X_tr_[train_index], y_tr[train_index], bs),
                        'va': (X_tr_[valid_index], y_tr[valid_index], len(valid_index) ),
                        'te': (X_te_[valid_index], y_te[valid_index], len(valid_index) )}
        
        # optimal wsee
        tr_loss[k], va_loss[k], te_loss[k] = [],[],[]
        tr_wsee[k], va_wsee[k], te_wsee[k] = [],[],[]
        
        # generate save paths
        m_str = MODEL.__name__.replace('_','.')+'+'+inner_optim.replace('-','.')
        print("\nmodel={ms} ; trainer={tn}".format(ms = m_str, tn = TRAINER.__name__))

        fix = '%s_kf+%d_bs+%d_nl+%d+%d_lr+%.2e_pinit+%s_loss+%s_rseed+%d'%(
            m_str, k_fold, bs, len(architect['h_sizes']), num_l, learning_rate, init, '+'.join(loss_which), rseed)

        save_path = PROJ_RT + 'runs/' + fix + f'/{k}/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        check_path = save_path + 'model.pt'

        # save this script
        shutil.copy2(__file__, save_path)
        print(save_path)

        # instantiate model      
        model = MODEL(num_layers = num_l, in_size = in_size, out_size = out_size, **architect, 
                      channel_info = cinfo_tr, dropout = dropout, inner_optim = inner_optim).to(device)
        num_params = count_parameters(model, 1)
        
        # optimization method
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        # construct the trainer and get loaders
        trainer = TRAINER(model, check_path, optimizer, resume = True, l2 = l2, display_step = 1000)
        loaders = trainer.get_loaders(loaders_dict)
        
        print([len(v) for v in loaders.values()])

        # start training, and save the best model (lowest loss)
        try:
            pbar = trange(epochs, desc='%d(/%d) fold'%(k+1, k_fold))
            pbar.update(trainer.epoch)
            
            for epoch in range(trainer.epoch, epochs):
                
                trainer.train(epoch=epoch, criterion=loss_which, loader=loaders['tr'], pbar=pbar)
                trainer.predict(epoch=epoch, criterion=loss_which, loader=loaders['va'], save=True, pbar=pbar)

                if not (epoch+1)%save_freq:
                    trainer._save_latest(True)

                # autostop
                if hasattr(trainer, 'stop') and trainer.stop:
                    print('Terninating because the loss hasn\'t been reducing for a while ...')
                    break
                    
                pbar.update(1)

                # test
                trainer.predict(epoch=epoch, criterion=loss_which, loader=loaders['te'], save=False, pbar=pbar)
                
            pbar.close()

        except (KeyboardInterrupt, SystemExit):
            trainer._save_latest()
            raise

        trainer._save_latest()
        
        # save json logs
        

