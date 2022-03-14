from tqdm.notebook import tqdm
from tqdm.auto import trange
from copy import deepcopy
import itertools as it
import glob 
import h5py
import json
import gc
import os

import numpy as np
import cvxpy as cp
import networkx as nx

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb 
import matplotlib.patheffects as pe
matplotlib.rcParams['pdf.fonttype'] = 42
PATH_EFF = [pe.Stroke(linewidth=3, foreground='w', alpha=1), pe.Normal()]
PATH_EFF_THIN = [pe.Stroke(linewidth=2, foreground='w', alpha=1), pe.Normal()]

COLORS   = plt.rcParams['axes.prop_cycle'].by_key()['color']
MARKERS  = ['o','v','^','s','D','*','x','+']
LINESTY  = ['-','--','-.',':']
BIGGER_SIZE = 14
NORMAL_SIZE = 12
plt.rc('font', size=BIGGER_SIZE, family='serif')          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=NORMAL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=NORMAL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=NORMAL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


PROJ_RT = "/root/usca_power_control/"
DATA_RT = "/root/usca_power_control/datasets/"
RSLT_RT = "/root/usca_power_control/results/"


## system constants
PDB = np.array(range(-40,10+1,1)) 


## colors
COLOR_CODE_D = {
    'WBS': '#1f77b4',
    'WBS-Rician': '#17becf',
    'Urb-SF': '#ff7f0e',
    'Urb-noSF': '#2ca02c',
    'Sub-SF': '#d62728',
    'Sub-noSF': '#9467bd'
}
COLOR_CODE = {
    'GCN-USCA-NS':'#17becf',
    'GCN-USCA':   '#1f77b4',
    'MLP-USCA-NS':'#e377c2',
    'MLP-USCA':   'brown',
    'GCN':        '#9467bd',
    'SCA':        '#ff7f0e',
    'Tr-SCA':     '#2ca02c',
    'Max-Pow':    'tan',
    'Opt':        '#8c564b',
}
MARKER_CODE = { #['o','v','^','s','D','*','x','+']
    'GCN-USCA-NS':'*',
    'GCN-USCA':   'v',
    'MLP-USCA-NS':'D',
    'MLP-USCA':   '^',
    'GCN':        's',
    'SCA':        '',
    'Tr-SCA':     '',
    'Max-Pow':    'o',
    'Opt':        '',
}
LINE_CODE = {
    'SCA':        '--',
    'Tr-SCA':     '-.',
    'Max-Pow':    ':',
}

### model paths
MODEL_PATHS = {}
MODEL_PATHS['GCN-USCA-NS'] = [
    './results/MR_DECSEQ_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.GCN.Embed2+learned.gcn4_nue+8_kf+4_bs+2040_nl+10+5+[8,32,32,16,8]_bd+0.6_fd+0.4_dropout+0.5_lr+5.00e-04_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/0/model.pt-seq9ft',
    './results/MR_DECSEQ_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.GCN.Embed2+learned.gcn4_nue+8_kf+4_bs+2040_nl+10+5+[8,32,32,16,8]_bd+0.6_fd+0.4_dropout+0.5_lr+5.00e-04_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/1/model.pt-seq9ft',
    './results/MR_DECSEQ_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.GCN.Embed2+learned.gcn4_nue+8_kf+4_bs+2040_nl+10+5+[8,32,32,16,8]_bd+0.6_fd+0.4_dropout+0.5_lr+5.00e-04_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/2/model.pt-seq9ft',
    './results/MR_DECSEQ_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.GCN.Embed2+learned.gcn4_nue+8_kf+4_bs+2040_nl+10+5+[8,32,32,16,8]_bd+0.6_fd+0.4_dropout+0.5_lr+5.00e-04_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/3/model.pt-seq9ft',
]
MODEL_PATHS['GCN-USCA'] = [
    './results/MR_WeightShareSeq_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.GCN.Embed2.R+learned.gcn4_nue+8_kf+4_bs+2040_nl+10+5+[16,64,64,64,16]_bd+0.6_fd+0.4_dropout+0.5_lr+5.00e-04_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/0/model.pt-seq8',
    './results/MR_WeightShareSeq_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.GCN.Embed2.R+learned.gcn4_nue+8_kf+4_bs+2040_nl+10+5+[16,64,64,64,16]_bd+0.6_fd+0.4_dropout+0.5_lr+5.00e-04_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/1/model.pt-seq8',
    './results/MR_WeightShareSeq_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.GCN.Embed2.R+learned.gcn4_nue+8_kf+4_bs+2040_nl+10+5+[16,64,64,64,16]_bd+0.6_fd+0.4_dropout+0.5_lr+5.00e-04_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/2/model.pt-seq9',
    './results/MR_WeightShareSeq_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.GCN.Embed2.R+learned.gcn4_nue+8_kf+4_bs+2040_nl+10+5+[16,64,64,64,16]_bd+0.6_fd+0.4_dropout+0.5_lr+5.00e-04_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/3/model.pt-seq9',
]
MODEL_PATHS['MLP-USCA-NS'] = [
    './results/MR_DECSEQ_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.MLP.Embed2+learned.mlp4_nue+8_kf+4_bs+2040_nl+10+5+[8,32,32,16,8]_bd+0.6_fd+0.4_dropout+0.5_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/0/model.pt-seq9ft',
    './results/MR_DECSEQ_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.MLP.Embed2+learned.mlp4_nue+8_kf+4_bs+2040_nl+10+5+[8,32,32,16,8]_bd+0.6_fd+0.4_dropout+0.5_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/1/model.pt-seq9ft',
    './results/MR_DECSEQ_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.MLP.Embed2+learned.mlp4_nue+8_kf+4_bs+2040_nl+10+5+[8,32,32,16,8]_bd+0.6_fd+0.4_dropout+0.5_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/2/model.pt-seq9ft',
    './results/MR_DECSEQ_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.MLP.Embed2+learned.mlp4_nue+8_kf+4_bs+2040_nl+10+5+[8,32,32,16,8]_bd+0.6_fd+0.4_dropout+0.5_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/3/model.pt-seq9ft',
]
MODEL_PATHS['MLP-USCA'] = [
    './results/MR_WeightShareSeq_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.MLP.Embed2.R+learned.mlp4_nue+8_kf+4_bs+2040_nl+10+5+[16,64,64,64,16]_bd+0.6_fd+0.4_dropout+0.5_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/0/model.pt-seq9',
    './results/MR_WeightShareSeq_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.MLP.Embed2.R+learned.mlp4_nue+8_kf+4_bs+2040_nl+10+5+[16,64,64,64,16]_bd+0.6_fd+0.4_dropout+0.5_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/1/model.pt-seq9',
    './results/MR_WeightShareSeq_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.MLP.Embed2.R+learned.mlp4_nue+8_kf+4_bs+2040_nl+10+5+[16,64,64,64,16]_bd+0.6_fd+0.4_dropout+0.5_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/2/model.pt-seq9',
    './results/MR_WeightShareSeq_L0w0_runs_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/USCA.MLP.Embed2.R+learned.mlp4_nue+8_kf+4_bs+2040_nl+10+5+[16,64,64,64,16]_bd+0.6_fd+0.4_dropout+0.5_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee_rseed+42/3/model.pt-seq9',
]
MODEL_PATHS['GCN'] = [
    './results/MR_VanillaMono_L0w0_runs_Semi_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/GCN.ChPt+vanilla_nue+8_kf+4_bs+2040_nl+0+5+[32,128,128,64,32]_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee(1.)+mono(.25)_rseed+42/0/model.pt',
    './results/MR_VanillaMono_L0w0_runs_Semi_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/GCN.ChPt+vanilla_nue+8_kf+4_bs+2040_nl+0+5+[32,128,128,64,32]_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee(1.)+mono(.25)_rseed+42/1/model.pt',
    './results/MR_VanillaMono_L0w0_runs_Semi_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/GCN.ChPt+vanilla_nue+8_kf+4_bs+2040_nl+0+5+[32,128,128,64,32]_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee(1.)+mono(.25)_rseed+42/2/model.pt',
    './results/MR_VanillaMono_L0w0_runs_Semi_SplitH_WBS_Alessio_rayleigh_aug_Ant+1/GCN.ChPt+vanilla_nue+8_kf+4_bs+2040_nl+0+5+[32,128,128,64,32]_lr+1.00e-03_l2+1.00e-06_pinit+full_loss+wsee(1.)+mono(.25)_rseed+42/3/model.pt',
]