import os
import gc
import h5py
import shutil
import numpy as np
import itertools as it
from tqdm.auto import trange,tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
DEVICE = torch.device('cuda')

# SET YOUR PROJECT ROOT DIR HERE
PROJ_RT = "/root/usca_power_control_github/"
DATA_RT = "/root/usca_power_control/datasets/"
RSLT_RT = "/root/usca_power_control_github/results/"
import sys; sys.path.append(PROJ_RT)

from utils import *
import dataset as ds
from models import USCA_GCN_Embed_R, USCA_MLP_Embed_R
from trainer import DecayWS_Seq_Trainer_G, DecayWS_Seq_Trainer


GRAPH_MODE = 1  # 0 | 1
NUM_STAB= 1e-12 

num_ue = 8
if GRAPH_MODE: 
    ## For GCN based USCA:
    in_size     = 1
    out_size    = 1
    models      = [USCA_GCN_Embed_R]
    trainers    = [DecayWS_Seq_Trainer_G]
    lrate       = 5e-4 #[2e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    train_folds = [0,1,2,3]

else:
    ## For MLP based USCA:
    in_size     = num_ue**2 +1 
    out_size    = num_ue
    models      = [USCA_MLP_Embed_R] 
    trainers    = [DecayWS_Seq_Trainer]
    lrate       = 1e-3
    train_folds = [0,1,2,3]


inner_architect = [
    {'h_sizes': [16, 64, 64, 64, 16], 
     'activs': ['elu', 'relu', 'elu', 'relu',  'elu']},
]

tr_dist = 'WBS_Alessio' # WBS_Alessio | HataSuburban | HataSuburban-noSF | HataUrban | HataUrban-noSF
te_dist = 'WBS_Alessio' #'HataSuburban'

tr_fade = 'rayleigh'    # rayleigh | rician
te_fade = 'rayleigh'

n_ant = 1
r_path = f'WeightShareSeq_SplitH_{tr_dist}_{tr_fade}_aug_Ant+{n_ant}'

k_fold = 4
bs = 510*4

num_outer_layers = [10]
block_lr_decay   = 0.6
ft_lr_decay      = 0.4

learning_rates = [lrate] 
decay_step     = None    
dropout        = 0.5
l2             = 1e-6
epochs         = 1000
save_freq      = 10
rseed          = 42
auto_stop      = 50

init         = 'full'     # full | rand | last 
loss_options = ['wsee']   # ['mse'] | ['wsee'] | ['wsee+mono']
cweights     = None          


# --- data---
dfn_tr = DATA_RT + f'/{tr_fade}/ue+{num_ue}_bs+4_ant+{n_ant}/channels-PL_{tr_dist}-NS_1000.h5' # train / validation data
X_tr, y_tr, cinfo_tr = ds.load_data_unsup(dfn_tr, hxp=False, num_stab=NUM_STAB)
print(X_tr.shape, y_tr.shape)

dfn_te = DATA_RT + f'/{te_fade}/ue+{num_ue}_bs+4_ant+{n_ant}/test-channels-PL_{te_dist}-NS_1000.h5' # test data
X_te, y_te, cinfo_te = ds.load_data_unsup(dfn_te, hxp=False, num_stab=NUM_STAB)
print(X_te.shape, y_te.shape)

# to torch tensor
y_tr = torch.from_numpy(y_tr).float().to(DEVICE)
y_te = torch.from_numpy(y_te).float().to(DEVICE)

# add initial pt (max)
attach_pt = lambda x: torch.from_numpy(np.hstack((init_p(x[:,-1], num_ue, method=init), x))).float().to(DEVICE)
X_tr_ = attach_pt(X_tr)
X_te_ = attach_pt(X_te)

# move channel info to DEVICE
dict_to_device = lambda x,dev: {k:v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
cinfo_tr = dict_to_device(cinfo_tr, DEVICE)
cinfo_te = dict_to_device(cinfo_te, DEVICE)

# training:
for learning_rate in learning_rates:

    for num_l in num_outer_layers:

        for arc in inner_architect:

            for loss_which in loss_options:

                for MODEL, TRAINER in zip(models, trainers):

                    tr_loss, va_loss, te_loss = {},{},{}
                    tr_wsee, va_wsee, te_wsee = {},{},{}
                    wsee_opt = {}

                    # k fold cross validation
                    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=rseed)
                    nP  = 51

                    for k, (train_index_, valid_index_) in enumerate(
                        skf.split(X_tr_.reshape(X_tr.shape[0]//nP, -1), np.ones(y_tr.shape[0]//nP, dtype=bool))):
                        
                        if k not in train_folds:
                            continue
                        
                        train_index = np.concatenate([np.arange(ti*nP, (ti+1)*nP) for ti in train_index_])
                        valid_index = np.concatenate([np.arange(vi*nP, (vi+1)*nP) for vi in valid_index_])

                        # data loaders
                        loaders_dict = {'tr': (X_tr_[train_index], y_tr[train_index], bs),
                                        'va': (X_tr_[valid_index], y_tr[valid_index], len(valid_index) ),
                                        'te': (X_te_, y_te, len(X_te_) )}

                        # generate save paths
                        m_str = MODEL.__name__.replace('_','.')
                        print("\nmodel={ms} ; trainer={tn}".format(ms = m_str, tn = TRAINER.__name__))

                        fix = "{model_name}_nue+{num_ue}_kf+{num_folds}_bs+{batch_size}_nl+{outer_nlayers}+{inner_nlayers}+{inner_num_nodes}_bd+{block_decay:.1f}_fd+{ft_decay:.1f}_dropout+{drop_out:.1f}_lr+{learning_rate:.2e}_l2+{l2:.2e}_pinit+{pt_initial}_loss+{loss_funcs}_rseed+{random_state}".format(
                            model_name      = m_str, 
                            num_ue          = num_ue, 
                            num_folds       = k_fold, 
                            batch_size      = bs, 
                            outer_nlayers   = num_l, 
                            inner_nlayers   = len(arc['h_sizes']), 
                            inner_num_nodes = str(arc['h_sizes']).replace(' ', ''), 
                            block_decay     = block_lr_decay, 
                            ft_decay        = ft_lr_decay, 
                            drop_out        = dropout,
                            learning_rate   = learning_rate, 
                            l2              = l2, 
                            pt_initial      = init, 
                            loss_funcs      = loss_which, 
                            random_state    = rseed)

                        save_path = RSLT_RT + f'{r_path:s}/{fix:s}/{k:d}/'
                        if not os.path.isdir(save_path):
                            os.makedirs(save_path)
                        else:
                            #continue
                            pass
                        check_path = save_path + 'model.pt'

                        # save this script
                        shutil.copy2(__file__, save_path)
                        print(save_path)

                        # instantiate model      
                        model = MODEL(num_layers   = num_l, 
                                      in_size      = in_size, 
                                      out_size     = out_size, 
                                      **arc, 
                                      channel_info = cinfo_tr,
                                      dropout      = dropout).to(DEVICE)
                        num_params = count_parameters(model, 1)
#                         raise
                        
                        print(model)
                        optimizer = (torch.optim.Adam, {'lr': learning_rate, 'weight_decay': l2})

                        # start training, and save the best model (lowest loss)
                        try:
                            loaders= None
                            nb = model.nblocks
                            
                            # start at block number:
                            completed = sorted([fn for fn in os.listdir(os.path.dirname(check_path)) \
                                           if 'model.pt' in fn and 'latest' in fn])
                            if len(completed):
                                sb = max([int(re.findall(r'seq(\d+)', fn)[0]) for fn in completed])
                            else:
                                sb = 0 

                            for bi in range(sb,nb):
                                
                                # construct the trainer and get loaders
                                curr_seq_path = check_path+f'-seq{bi}' #+ 'ft'*(bi==0)
                                next_seq_path = check_path+f'-seq{bi+1}'
                                
                                if not os.path.exists(next_seq_path):

                                    trainer = TRAINER(model        = model, 
                                                      check_path   = curr_seq_path, 
                                                      optimizer    = None, #optimizer, # SET LATER
                                                      resume       = True, 
                                                      decay_step   = decay_step,
                                                      block_decay  = block_lr_decay,
                                                      display_step = 1, 
                                                      auto_stop    = auto_stop)
                                    trainer.set_criteria(loss_which, cweights=None) # specify what criteria to use

                                    if loaders is None:
                                        loaders = trainer.get_loaders(loaders_dict)

                                    """ freeze all previous blocks """
                                    trainer.set_optimizer(optimizer)
                                    trainer.lr_init = learning_rate

                                    pbar = trange(epochs, desc=f'{bi+1}(/{nb}) blocks: {k+1}(/{k_fold}) fold')
                                    pbar.update(trainer.epoch)                                

                                    """ train the current block """
                                    for epoch in range(trainer.epoch, epochs):
                                        trainer.train(epoch     = epoch, 
                                                      b_curr    = [0,bi+1], 
                                                      criterion = loss_which, 
                                                      loader    = loaders['tr'], 
                                                      pbar      = pbar)
                                        
                                        with torch.no_grad():
                                            trainer.predict(epoch     = epoch, 
                                                            b_curr    = [0,bi+1], 
                                                            criterion = loss_which, 
                                                            loader    = loaders['va'], 
                                                            save      = True, 
                                                            pbar      = pbar)

                                        if not (epoch+1)%save_freq:
                                            trainer.save_checkpoint(True)

                                        # autostop
                                        if hasattr(trainer, 'stop') and trainer.stop:
                                            print(f'Finish training block#{bi} because the loss hasn\'t been reducing for {auto_stop} epochs')
                                            break

                                        pbar.update(1)

                                    trainer.save_checkpoint()
                                    pbar.close()

                        except (KeyboardInterrupt, SystemExit):
                            trainer.save_checkpoint()
                            raise

                        trainer._save_latest()

                        model, trainer = None, None
                        torch.cuda.empty_cache()
                        gc.collect()

