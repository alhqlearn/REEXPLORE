
import os
import numpy as np
import pandas as pd
import random
import threading
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm,trange

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import AllChem
from rdkit import RDLogger 


import matplotlib.pyplot as plt
import seaborn as sns
from fcd_torch import FCD
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*') # switch off RDKit warning messages

import argparse

sys.path.append('./rl_agent')
sys.path.append('./regressor_py')
sys.path.append('./reinforce')
sys.path.append('./result')
sys.path.append('./dataset')
sys.path.append('./regressor')
sys.path.append('./fastai1/')
sys.path.append('./scscore/')
sys.path.append('./scscore/scscore/')

from fastai import *
from fastai.text import *
from fastai.vision import *
from fastai.imports import *

current_path = os.getcwd()
print(current_path)

device = torch.device('cuda:5' if torch.cuda.is_available() else "cpu")

from models import RNN, OneHotRNN, EarlyStopping
from datasets import SmilesDataset, SelfiesDataset, SmilesCollate, Vocabulary
from functions import decrease_learning_rate, print_update, track_loss, \
     sample_smiles, write_smiles
     
from utils import load_model, is_valid, novelty_score, dataframe_result, plot_sc_score, simple_moving_average, plot_out


from tdc import Evaluator
import tl_Predictor_Reaction_c
from tl_Predictor_Reaction_c import pred_init, train_reg, test_performance, test_performance, predictor


import functions_rl
from functions_rl import generate_allfragments, join_frag, usable_frag, permute, gen_firstatom_frag, plot_hist, tensor_to_array, canonical_smiles


import standalone_model_numpy
from standalone_model_numpy import SCScorer
scscore_model = SCScorer()
scscore_model.restore('./pre_trained_files/model.ckpt-10654.as_numpy.json.gz')



seed = 0
batch_size = 128
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    print("using cuda")
    torch.cuda.manual_seed_all(seed)
    
# Argument parsing
parser = argparse.ArgumentParser(description='REEXPLORE')
parser.add_argument('--agent_dataset', type=str, choices=['ChEMBL', 'ZINC', 'COCONUT'], default='ChEMBL')
parser.add_argument('--trajectories_method', type=str, choices=['random', 'topp', 'topk'], default='random')
parser.add_argument('--core_smiles', type=str, default='O(*)p1oc2ccc3c(cccc3)c2c2c3ccccc3ccc2o1')
parser.add_argument('--num_trajectories', type=int, default=4000)
args = parser.parse_args()

# Configuration based on the chosen dataset
dataset_paths = {
    'ZINC': ('./pre_trained_files/vocab_zinc', './pre_trained_files/checkpoint_zinc', './dataset/zinc.csv'),
    'ChEMBL': ('./pre_trained_files/vocab_chembl', './pre_trained_files/checkpoint_chembl', './dataset/chembl.csv'),
    'COCONUT': ('./pre_trained_files/vocab_coconut', './pre_trained_files/checkpoint_coconut', './dataset/coconut.csv'),
}

vocab_file, model_path, smiles_file = dataset_paths[args.agent_dataset]

# Initialization
vocabulary_agent = Vocabulary(vocab_file=vocab_file)
rl_agent = RNN(vocabulary=vocabulary_agent,
                rnn_type='GRU',                      
                embedding_size= 128,                 
                hidden_size=512,                     
                n_layers=3,                          
                dropout=0,                           
                bidirectional=False,                 
                tie_weights=False,
                nonlinearity='tanh')



load_model(rl_agent, model_path)

smiles_agent_file = pd.read_csv(smiles_file, header=None)
smiles_agent_file.columns = ['smiles_train']
smiles_ref = list(set(smiles_agent_file.smiles_train))

# Check if the model is on CUDA (GPU)
device_type = 'cuda' if next(rl_agent.parameters()).is_cuda else 'cpu'

print(f"RNN Agent is on {device_type}")

sample_technique = getattr(rl_agent, f'{args.trajectories_method}_sampling')
sampled_smiles = []
sample_size = 500
while len(sampled_smiles) < sample_size:
    sampled_smiles.extend(sample_technique(batch_size, return_smiles=True))

mols_sampled_smiles = list(filter(is_valid, sampled_smiles))
print(f"Percentage of validity for pre-trained RL agent: {(len(mols_sampled_smiles) / len(sampled_smiles)) * 100}")


# Trained_predictor Initialization
#Parameter defining
seed_tl = 1234
batch_size = 128
reaction_dataset = pd.read_csv('./dataset/Reaction_c.csv')
augm = 100
drp_out = 0.2 
sigm_g = 0.5


#Loading of pre-trained weight using Transfer Learning
reg_learner_pre, train_aug , valid = pred_init(seed_tl, batch_size, reaction_dataset, current_path, augm, drp_out, sigm_g) 
reaction_test_performance = test_performance(seed_tl, batch_size, reaction_dataset, train_aug, valid, current_path, drp_out, sigm_g)


# Loading of Unbiased RL settings 
n_to_generate=500

othercomponents = 'C=C(CC(=O)OC)C(=O)OC.ClCCl'
def add_othercomponents(smile, components = 'C=C(CC(=O)OC)C(=O)OC.ClCCl'):
    return smile + '.' + components
    

real_comp = pd.read_csv('./dataset/Reaction_c_real.csv')
real_comp_list = list(real_comp['smiles'])

evaluator = Evaluator(name = 'Diversity')
fcd = FCD(canonize=False)
def rl_adjustment(rl_agent, surrogate_regressor, core_smi, n_to_generate, savedir1 = None, savedir2 = None, **kwargs):
    rng = np.random.default_rng()
    seed_value = rng.integers(low = 10000)

    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False

    generated = []
    generated_mol = []
    #pbar = tqdm(range(n_to_generate))
    
    #pbar.set_description("Generating molecules...")
    no_sample = n_to_generate
    rl_sample_technique = getattr(rl_agent, f'{args.trajectories_method}_sampling')
    sampled_smiles = rl_sample_technique(no_sample, max_len=100, return_smiles=True)
    generated.append(sampled_smiles)
        
    generated = [ y for ys in generated for y in ys]
  
    generated_novel = []
    
    x = 0            
    for j in range(len(generated)):
        if_smile = Chem.MolFromSmiles(generated[j])
        if if_smile is not None:
            x+=1

            fragment = gen_firstatom_frag(generated[j])
            mol = core_smi
            for i in range(core_smi.count('(*)')):
                mol = join_frag(mol, fragment)
                mol = usable_frag(mol)

            generated_novel.append(generated[j])
            generated_mol.append(mol)

    if x==0:
        return [], []

    sanitized = canonical_smiles(generated_mol, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]

    unique_components = []
    for i in range(len(unique_smiles)):
        unique_components.append(add_othercomponents(unique_smiles[i]))


    smiles, prediction = surrogate_regressor.predictor(unique_components, seed_tl, batch_size, reaction_dataset, train_aug, valid, current_path, drp_out, sigm_g)
    
    sc_score_gen= plot_sc_score(scscore_model, unique_smiles, real_comp_list, savedir1)
   

    novel_mols = novelty_score(set(generated_novel), set(smiles_ref))    

    print("Percentage of validity: ", (x/n_to_generate)*100)
    print("Percentage of uniqueness: ", (len(set(generated_novel))/n_to_generate)*100)
    print("Percentage of novelty: ", (len(novel_mols)/n_to_generate)*100)
    print("Internal diversity: ", np.round(evaluator(unique_smiles),2)) 
    plot_hist(prediction, n_to_generate, savedir2)   
    print("Mean synthetic complexeity score: ", np.round(np.mean(sc_score_gen),2))               
    print("FCD distance: ", np.round(fcd(unique_smiles, real_comp_list),2))                                     


    return smiles, prediction, generated, unique_smiles, sc_score_gen
    
sc_path_ub = './result/' + 'Synthetic_score_Reaction_c_unbiased_' + str(args.agent_dataset) + '_' + str(args.trajectories_method)
output_path_ub = './result/' + 'Average_output_Reaction_c_unbiased_' + str(args.agent_dataset) + '_' + str(args.trajectories_method)

print("------------ Devoid of RL... Unbiased setting ---------")
smiles_unbiased, prediction_unbiased, generated_unbiased, molecule_list_unbiased, sc_unbiased = rl_adjustment(rl_agent, tl_Predictor_Reaction_c, args.core_smiles, n_to_generate, sc_path_ub, output_path_ub)


smile_pred_unbiased_df = dataframe_result(smiles_unbiased, prediction_unbiased, molecule_list_unbiased, sc_unbiased)
unbiased_result_path = './result/' + 'Reaction_c_unbiased_' + str(args.agent_dataset) + '_' + str(args.trajectories_method) + '.csv'
smile_pred_unbiased_df.to_csv(unbiased_result_path, index =None)


# RL training 

import reinforce_fragment_Reaction_c
from reinforce_fragment_Reaction_c import Reinforcement_random, Reinforcement_topp, Reinforcement_topk


#Parameter
n_policy = 10
n_iterations = 8

def Reaction_c_get_reward(smiles, surrogate_regressor, invalid_reward=0.0):
    
    molecule_smiles, predicted_value = surrogate_regressor.predictor(smiles, seed_tl, batch_size, reaction_dataset, train_aug, valid, current_path, drp_out, sigm_g) 
    predicted_value = tensor_to_array(predicted_value)
    rewards = np.zeros([len(smiles)])
    for i in range(len(smiles)):
        sample = smiles[i]
        if sample == '':
            rewards[i] = -2
            continue
        else:
            a = sample.find('.')
            smile_ = sample[:a]
            #print(smile_)
            mol = Chem.MolFromSmiles(smile_)
            if mol is None:
                rewards[i] = -2
                continue
            else:
                charge = 0
                for atom in mol.GetAtoms():
                    atom_charge = atom.GetFormalCharge()
                    if atom_charge != 0:
                        charge = atom_charge 
                if charge != 0:
                    rewards[i] = 1
                    continue
                else:
                    if Descriptors.MolWt(mol) > 550:
                        rewards[i] = 2
                        continue
                        
                    else:
                        if Lipinski.NumHDonors(mol) > 1 or Lipinski.NumHAcceptors(mol) > 10:
                            rewards[i] = 3
                            continue
                        else:  
                            sm, SCscore = scscore_model.get_score_from_smi(smile_)
                            if SCscore > 3.5:            
                                rewards[i] = 4 + ((4.5-SCscore)*2)

                            else:
                                s = int((predicted_value[i]-90))
                                rewards[i] = 7 + ((2*s)+1)
   
    return rewards

def get_pred_val(smiles, surrogate_regressor):
    generated_novel = []
    for j in range(len(smiles)):
        if_smile = Chem.MolFromSmiles(smiles[j])
        if if_smile is not None:
            generated_novel.append(smiles[j])  
    unique_components = list(np.unique(generated_novel))
    mol, predicted_tensor = surrogate_regressor.predictor(unique_components, seed_tl, batch_size, reaction_dataset, train_aug, valid, current_path, drp_out, sigm_g)
    predicted_array = tensor_to_array(predicted_tensor)
    return predicted_array
    

if args.trajectories_method == 'random':
    RL_max = Reinforcement_random(rl_agent, tl_Predictor_Reaction_c, Reaction_c_get_reward, get_pred_val)
if args.trajectories_method =='topp':
    RL_max = Reinforcement_topp(rl_agent, tl_Predictor_Reaction_c, Reaction_c_get_reward, get_pred_val)
if args.trajectories_method =='topk':
    RL_max = Reinforcement_topk(rl_agent, tl_Predictor_Reaction_c, Reaction_c_get_reward, get_pred_val)
    

rewards_max = []
rl_losses_max = []
pred_Reinforce_max_plot = []


for i in range(n_iterations):
    for j in trange(n_policy, desc='Policy gradient...'):
        cur_reward, cur_loss, cur_pred = RL_max.policy_gradient(vocabulary_agent, n_batch = args.num_trajectories, core_smi = args.core_smiles, others = othercomponents)
        pred_Reinforce_max_plot.append(cur_pred)
        rewards_max.append(simple_moving_average(rewards_max, cur_reward)) 
        rl_losses_max.append(simple_moving_average(rl_losses_max, cur_loss))


print("------------ RL with sequential reward function ---------")

sc_path_bias = './result/' + 'Synthetic_score_Reaction_c_biased_' + str(args.agent_dataset) + '_' + str(args.trajectories_method)
output_path_bias = './result/' + 'Average_output_Reaction_c_biased_' + str(args.agent_dataset) + '_' + str(args.trajectories_method)


smiles_biased, prediction_biased, generated_biased, molecule_list_biased, sc_biased =rl_adjustment(RL_max.generator, tl_Predictor_Reaction_c,args.core_smiles, n_to_generate, sc_path_bias, output_path_bias)
smile_pred_biased_df = dataframe_result(smiles_biased, prediction_biased, molecule_list_biased, sc_biased)
biased_result_path = './result/' + 'Reaction_c_biased_' + str(args.agent_dataset) + '_' + str(args.trajectories_method) + '.csv'
smile_pred_biased_df.to_csv(biased_result_path, index =None)
plot_out(pred_Reinforce_max_plot, './result/' + 'Episodic_performance_Reaction_c_biased_' + str(args.agent_dataset) + '_' + str(args.trajectories_method))

print('Reaction discovery task using RL is now finished. All the best ...!!')
                    