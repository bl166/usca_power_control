import os
import gc
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
from models_uf import USCA_MLP, USCA_MLP_R, MLP_ChPm, GCN_ChPt, USCA_GCN
from trainer import Simple_Trainer, Simple_Trainer_G


device = torch.device('cuda')

num_ue = 10
in_size = num_ue**2 +1 # + num_ue
out_size = num_ue

inner_architect = [
#     {'h_sizes': [32, 16, 8],
#      'activs': ['elu', 'relu', 'elu']}#,
    {'h_sizes': [16, 64, 32, 16, 8], 
     'activs': ['elu', 'relu', 'elu', 'relu',  'elu']}#,        
#     {'h_sizes': [128, 64, 32, 32, 20, 16, 8], 
#      'activs': ['elu', 'relu', 'elu', 'relu', 'elu', 'relu',  'elu']},    
#     {'h_sizes': [256, 256, 64, 64, 32, 32, 32, 32, 8], 
#      'activs': ['elu', 'relu', 'elu', 'relu', 'elu', 'relu', 'elu', 'relu',  'elu']}
]

k_fold = 2
bs = 512

num_outer_layers = [0]#[1,2,3,4,5,7,9] 

learning_rate = 0.0005#1
dropout = 0
l2 = 1e-6
epochs = 500
save_freq = 50

init = 'rand'
rseed= 42
loss_options= [['wsee']]
# inner_optim='learned-mlp' 
inner_optim='vanilla' 

models = [MLP_ChPm]#[USCA_MLP] #[MLP_ChPm]
trainers = [Simple_Trainer]

#--- data---

dfn_tr = PROJ_RT+'../../data_my/channels-hataSuburban-10.h5' # train / validation data
X_tr, y_tr, cinfo_tr = ds.load_data_unsup(dfn_tr)
print(X_tr.shape, y_tr.shape)

dfn_te = PROJ_RT+'../../data_my/channels-hataUrban-10.h5' # test data
X_te, y_te, cinfo_te = ds.load_data_unsup(dfn_te)
print(X_te.shape, y_te.shape)

# to torch tensor
y_tr = torch.from_numpy(y_tr).float().to(device)
y_te = torch.from_numpy(y_te).float().to(device)

# # add initial pt (max)
attach_pt = lambda x: torch.from_numpy(np.hstack((init_p(x[:,-1], num_ue, method=init), x))).float().to(device)
X_tr_ = attach_pt(X_tr)
X_te_ = attach_pt(X_te)
# X_tr_ = torch.from_numpy(X_tr).float().to(device)
# X_te_ = torch.from_numpy(X_te).float().to(device)

# assert in_size==X_tr_.shape[1]

# move channel info to device
dict_to_device = lambda x,dev: {k:v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
cinfo_tr = dict_to_device(cinfo_tr, device)
cinfo_te = dict_to_device(cinfo_te, device)

# training:
for num_l in num_outer_layers:

    for arc in inner_architect:
        
        for loss_which in loss_options:

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

                    # generate save paths
                    m_str = MODEL.__name__.replace('_','.')+'+'+inner_optim.replace('-','.')
                    print("\nmodel={ms} ; trainer={tn}".format(ms = m_str, tn = TRAINER.__name__))

                    fix = "{model_name}_kf+{num_folds}_bs+{batch_size}_nl+{outer_nlayers}+{inner_nlayers}+{inner_num_nodes}_lr+{learning_rate:.2e}_l2+{l2:.2e}_pinit+{pt_initial}_loss+{loss_funcs}_rseed+{random_state}".format(
                        model_name=m_str, num_folds=k_fold, batch_size=bs, outer_nlayers=num_l, 
                        inner_nlayers=len(arc['h_sizes']), inner_num_nodes=str(arc['h_sizes']).replace(' ', ''), 
                        learning_rate=learning_rate, l2=l2, pt_initial=init, loss_funcs= '+'.join(loss_which), 
                        random_state=rseed)

                    save_path = PROJ_RT + 'runs_10_rand/' + fix + f'/{k}/'
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    check_path = save_path + 'model.pt'

                    # save this script
                    shutil.copy2(__file__, save_path)
                    print(save_path)

                    """
                    instantiate model      
                    """
#                     # USCA-MLP:
#                     model = MODEL(num_layers = num_l, in_size = in_size, out_size = out_size, **arc, 
#                                   channel_info = cinfo_tr, dropout = dropout, inner_optim = inner_optim).to(device)
#                     # USCA-GCN:
#                     model = MODEL(num_layers = num_l, in_size = 1, out_size = 1, **arc, 
#                                   channel_info = cinfo_tr, dropout = dropout, inner_optim = inner_optim).to(device)   
                    # VANILLA MLP:
                    model = MODEL(in_size = in_size, out_size = out_size, **arc, dropout=dropout).to(device)
#                     # VANILLA GCN:
#                     model = MODEL(in_size=1, out_size=1, **arc, channel_info = cinfo_tr).to(device)
                    
                    num_params = count_parameters(model, 1)

                    # optimization method
                    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

                    # construct the trainer and get loaders
                    trainer = TRAINER(model, check_path, optimizer, resume = True, l2 = l2, display_step = 5, mode='UNSUP')
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

                    model, trainer = None, None
                    gc.collect()

