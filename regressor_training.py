import sys
sys.path.append('./rl_agent')
sys.path.append('./regressor_file')
sys.path.append('./reinforce')
sys.path.append('./result')
sys.path.append('./dataset')
sys.path.append('./regressor')
sys.path.append('./fastai1/')


from sklearn.model_selection import train_test_split


from fastai import *
from fastai.text import *
from fastai.vision import *
from fastai.imports import *

import numpy as np
import threading
import random
from sklearn.utils import shuffle
import pandas as pd 
import numpy as np


import os
current_path = os.getcwd()
print(current_path)


device = torch.device('cuda:5' if torch.cuda.is_available() else "cpu")


import tl_Predictor_Reaction
from tl_Predictor_Reaction import pred_init, train_reg, test_performance, test_performance

#Parameter 
seed_tl = 1234
batch_size = 128
reaction_dataset = pd.read_csv('./dataset/Reaction_a.csv')
augm = 100
drp_out = 0.0 
sigm_g = 0.0

#Loading of pre-trained weight using Transfer Learning
reg_learner_pre, train_aug , valid = pred_init(seed_tl, batch_size, reaction_dataset, current_path, augm, drp_out, sigm_g)


unf1 = 6
unf2 = 6
unf3 = 6
unf4 = 10


reg_learner_trained = train_reg(unf1, unf2, unf3, unf4, reg_learner_pre)
model_filename = 'reg_Reaction_a'
# Choose checkpoint file of trained target-task regressor according to your dataset [reg_Reaction_a/reg_Reaction_b/reg_Reaction_c]
reg_learner_trained.save(model_filename)  

test_rmse = test_performance(seed_tl, batch_size, reaction_dataset, train_aug, valid, current_path, drp_out, sigm_g, model_filename)



