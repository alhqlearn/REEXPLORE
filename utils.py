
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import RDLogger 
import numpy as np
import pandas as pd
from tqdm import tqdm,trange
import seaborn as sns
import matplotlib.pyplot as plt


import functions_rl
from functions_rl import generate_allfragments, join_frag, usable_frag, permute, gen_firstatom_frag, plot_hist, tensor_to_array, canonical_smiles

def plot_out(value_list, path_dir):
    x_ah = []
    y_ah = []
    for i in range(len(value_list)):
        x_ah.append(value_list[i])
        y_ah.append(float(i))
    plt.figure()     
    p1 = sns.lineplot(x=y_ah, y=x_ah, linewidth=2, color='green', alpha=0.8)
    p1.set(xlabel="Number of episodes", ylabel="Average %yield/ee", title = 'Elevation in %yield/$ee$ over RL episodes')
    plt.tight_layout() 
    plt.savefig(path_dir, dpi=300)
    #plt.show()

def load_model(model, path):
        weights = torch.load(path)
        model.load_state_dict(weights)


def is_valid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and mol.GetNumAtoms()>0:
        return smiles
        

def combinatorial_evaluation_Reaction_a(alc_list, base_list, fluor_list, tl_Predictor_Reaction_a):
    alc_list =  np.asarray(alc_list, dtype=np.str_)
    SF_base_pred_list = []
    for i in range(len(fluor_list)):
        for j in range(len(base_list)):
            current_list = []
            for k in range(len(alc_list)):
                #x = alc_list[k] + '.' + fluor_list[i] + '.' + base_list[j]
                #print(x)
                current_list.append(alc_list[k] + '.' + fluor_list[i] + '.' + base_list[j])
            
            smiles, prediction = tl_Predictor_Reaction_a.predictor(current_list, seed_tl, batch_size, filename, train_aug, valid, current_path, drp_out, sigm_g)
            SF_base_pred_list.append(tensor_to_array(prediction).mean())
            print("Fluorinating agent: ", fluor_list[i])
            print("Base:", base_list[j])
            plot_hist(prediction, 1)

    return SF_base_pred_list       
    


def combinatorial_evaluation_Reaction_b(lig_list, thiol_list, imine_list, tl_Predictor_Reaction_b):
    lig_list =  np.asarray(lig_list, dtype=np.str_)
    IM_thiol_pred_list = []
    for i in range(len(imine_list)):
        for j in range(len(thiol_list)):
            current_list = []
            for k in range(len(lig_list)):
                #x = lig_list[k] + '.' + imine_list[i] + '.' + thiol_list[j]
                #print(x)
                current_list.append(lig_list[k] + '.' + imine_list[i] + '.' + thiol_list[j])
            
            smiles, prediction = tl_Predictor_Reaction_b.predictor(current_list, seed_tl, batch_size, filename, train_aug, valid, current_path, drp_out, sigm_g)
            IM_thiol_pred_list.append(tensor_to_array(prediction).mean())
            print("imine: ", imine_list[i])
            print("thiol:", thiol_list[j])
            plot_hist(prediction, 1)

    return SF_thiol_pred_list    
    
def novelty_score(mols,ref_mols): 
    return set.difference(mols,ref_mols)
    
    
def dataframe_result(smile, pred_val, all_smiles, sc_score_list):
    smile_df = pd.DataFrame(smile, columns = ['smiles'])
    prediction_array = list(tensor_to_array(pred_val))
    prediction_df = pd.DataFrame(prediction_array, columns = ['predicted_value'])
    smile_pred_df  = pd.concat([smile_df,prediction_df], axis =1)
    all_smiles_df = pd.DataFrame(all_smiles, columns = ['explored_smiles'])
    smile_pred_df1 = pd.concat([smile_pred_df,all_smiles_df], axis =1)
    all_smiles_sc_df = pd.DataFrame(sc_score_list, columns = ['sc_score'])
    smile_pred_df2 = pd.concat([smile_pred_df1,all_smiles_sc_df], axis =1)
    return smile_pred_df2
    
    
def plot_sc_score(scscore_model, gen_smiles, alcohol_list, path_dir):
    scscore_all = []
    for i in range(len(gen_smiles)):
        sm, score = scscore_model.get_score_from_smi(gen_smiles[i])
        scscore_all.append(score)
    scscore_real = []
    for i in range(len(alcohol_list)):
        sm, score = scscore_model.get_score_from_smi(alcohol_list[i])
        scscore_real.append(score)
    #scs_plot
    plt.figure() #Create a new figure for the plot
    p1 = sns.kdeplot(scscore_real, color='blue', fill=True, label='Experimental = ' +str(np.round(np.mean(scscore_real),2)))
    p1 = sns.kdeplot(scscore_all, color='Green', fill=True, label='Generated = ' + str(np.round(np.mean(scscore_all), 2))) 
    p1.set(xlabel='Synthetic complexeity score', title='SC Score distribution with mean')
    p1.legend(bbox_to_anchor=(1.02, 1.02), loc='best')
    plt.tight_layout() 
    if path_dir is not None:
      plt.savefig(path_dir, dpi=300)
    return scscore_all
    
def cal_rdkit_prop(smile):
    mol = Chem.MolFromSmiles(smile)
    # Calculate molecular weight
    mw = Descriptors.MolWt(mol)
    # Calculate number of rotatable bonds
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    # Calculate number of H-bond donors and acceptors
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    # Calculate number of rings
    rings = Lipinski.RingCount(mol)
    # Calculate number of aromatic rings
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)
    
    # Check if molecule is uncharged
    charge = 0
    for atom in mol.GetAtoms():
        atom_charge = atom.GetFormalCharge()
        if atom_charge != 0:
            #print(f"Atom with index {atom.GetIdx()} is charged with a charge of {atom_charge}")
            charge = atom_charge
    # Calculate volume
    #print(smile)
    '''
    mol1 = Chem.AddHs(Chem.MolFromSmiles(smile))
    AllChem.EmbedMolecule(mol1)
    volume = AllChem.ComputeMolVolume(mol1)
    '''
    # Calculate logP
    logp = rdkit.Chem.Crippen.MolLogP(mol)
     
    all_prop = np.array([mw,rot_bonds,hbd,hba,rings,num_aromatic_rings,charge,logp])
    

    return all_prop
    
    
    
def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma




