
import argparse
import os
import numpy as np
import pandas as pd
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import functions_rl
from functions_rl import generate_allfragments, join_frag, usable_frag, permute, gen_firstatom_frag, plot_hist, tensor_to_array, canonical_smiles


class Reinforcement_random(object):
    def __init__(self, generator, predictor, get_reward, get_pred_val):
       
        super(Reinforcement_random, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward
        self.get_pred_val = get_pred_val

    def policy_gradient(self, vocab, n_batch=4000, gamma=0.99,
                        std_smiles=False, grad_clipping=None, core_smi='O(*)p1oc2ccc3c(cccc3)c2c2c3ccccc3ccc2o1', others = 'C=C(CC(=O)OC)C(=O)OC.ClCCl', **kwargs):

        # New part begins
        optimizer = optim.Adam(self.generator.parameters(),
                       betas=(0.9, 0.999), ## default
                       eps=1e-08, ## default
                       lr=0.001)
        rl_loss = 0
        optimizer.zero_grad()
        total_reward = 0
        samples = []

        trajectories = self.generator.random_sampling(n_batch, max_len=100, return_smiles = True)
        for i in range(len(trajectories)):
            if_smile = Chem.MolFromSmiles(trajectories[i])
            if if_smile is None:
                mol = ''
                samples.append(mol)
            else:
                fragment = gen_firstatom_frag(trajectories[i])
                mol = core_smi
                for i in range(core_smi.count('(*)')):
                    mol = join_frag(mol, fragment)
                    mol = usable_frag(mol)

                mol = mol + '.' + others
                samples.append(mol)

        rewards = self.get_reward(samples, self.predictor, **kwargs)
        reward_arr = np.array(rewards, dtype='float64')
        #print(reward_arr)
        pred_Reinforce = self.get_pred_val(samples, self.predictor, **kwargs)
        mean_pred_Reinforce = np.mean(pred_Reinforce)
        tensors = []
        self.generator.train()
        hidden = self.generator.init_hidden(1)
        
        def accumulate_loss(smile, reward, gamma, hidden):
            loss = 0
            traj_tensor = vocab.encode(vocab.tokenize(smile)).cuda()
            discounted = reward
            for p in range(len(traj_tensor)-1):
              logits, hidden = self.generator(traj_tensor[p].view(-1,1), hidden)
              log_probs = F.log_softmax(logits, dim=2)
              next_char = traj_tensor[p+1]
              loss += (log_probs[0, 0, next_char] * discounted)
              discounted = discounted * gamma
            return loss
        
        unq_factor = 0.75

        unique = list(set(trajectories))
        counter = np.zeros([len(unique)])



        for k in range(len(trajectories)):
            counter[unique.index(trajectories[k])] +=1

            rewards[k] = rewards[k] * (unq_factor ** (counter[unique.index(trajectories[k])]-1))
            l = accumulate_loss(trajectories[k], rewards[k], gamma, hidden)
            rl_loss -= l

        total_reward = np.sum(rewards)

        rl_loss = rl_loss / n_batch
        rl_loss.backward()
        
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                           grad_clipping)

        optimizer.step()
        
        return total_reward/n_batch, rl_loss.item(), mean_pred_Reinforce

class Reinforcement_topp(object):
    def __init__(self, generator, predictor, get_reward, get_pred_val):
       
        super(Reinforcement_topp, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward
        self.get_pred_val = get_pred_val

    def policy_gradient(self, vocab, n_batch=4000, gamma=0.99,
                        std_smiles=False, grad_clipping=None, core_smi='O(*)p1oc2ccc3c(cccc3)c2c2c3ccccc3ccc2o1', others = 'C=C(CC(=O)OC)C(=O)OC.ClCCl', **kwargs):

        # New part begins
        optimizer = optim.Adam(self.generator.parameters(),
                       betas=(0.9, 0.999), ## default
                       eps=1e-08, ## default
                       lr=0.001)
        rl_loss = 0
        optimizer.zero_grad()
        total_reward = 0
        samples = []

        trajectories = self.generator.topp_sampling(n_batch, max_len=100, return_smiles = True)
        for i in range(len(trajectories)):
            if_smile = Chem.MolFromSmiles(trajectories[i])
            if if_smile is None:
                mol = ''
                samples.append(mol)
            else:
                fragment = gen_firstatom_frag(trajectories[i])
                mol = core_smi
                for i in range(core_smi.count('(*)')):
                    mol = join_frag(mol, fragment)
                    mol = usable_frag(mol)

                mol = mol + '.' + others
                samples.append(mol)

        rewards = self.get_reward(samples, self.predictor, **kwargs)
        reward_arr = np.array(rewards, dtype='float64')
        #print(reward_arr)
        pred_Reinforce = self.get_pred_val(samples, self.predictor, **kwargs)
        mean_pred_Reinforce = np.mean(pred_Reinforce)
        tensors = []
        self.generator.train()
        hidden = self.generator.init_hidden(1)
        
        def accumulate_loss(smile, reward, gamma, hidden):
            loss = 0
            traj_tensor = vocab.encode(vocab.tokenize(smile)).cuda()
            discounted = reward
            for p in range(len(traj_tensor)-1):
              logits, hidden = self.generator(traj_tensor[p].view(-1,1), hidden)
              log_probs = F.log_softmax(logits, dim=2)
              next_char = traj_tensor[p+1]
              loss += (log_probs[0, 0, next_char] * discounted)
              discounted = discounted * gamma
            return loss
        
        unq_factor = 0.75

        unique = list(set(trajectories))
        counter = np.zeros([len(unique)])



        for k in range(len(trajectories)):
            counter[unique.index(trajectories[k])] +=1

            rewards[k] = rewards[k] * (unq_factor ** (counter[unique.index(trajectories[k])]-1))
            l = accumulate_loss(trajectories[k], rewards[k], gamma, hidden)
            rl_loss -= l

        total_reward = np.sum(rewards)

        rl_loss = rl_loss / n_batch
        rl_loss.backward()
        
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                           grad_clipping)

        optimizer.step()
        
        return total_reward/n_batch, rl_loss.item(), mean_pred_Reinforce
        
        
class Reinforcement_topk(object):
    def __init__(self, generator, predictor, get_reward, get_pred_val):
       
        super(Reinforcement_topk, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward
        self.get_pred_val = get_pred_val

    def policy_gradient(self, vocab, n_batch=4000, gamma=0.99,
                        std_smiles=False, grad_clipping=None, core_smi='O(*)p1oc2ccc3c(cccc3)c2c2c3ccccc3ccc2o1', others = 'C=C(CC(=O)OC)C(=O)OC.ClCCl', **kwargs):

        # New part begins
        optimizer = optim.Adam(self.generator.parameters(),
                       betas=(0.9, 0.999), ## default
                       eps=1e-08, ## default
                       lr=0.001)
        rl_loss = 0
        optimizer.zero_grad()
        total_reward = 0
        samples = []

        trajectories = self.generator.topk_sampling(n_batch, max_len=100, return_smiles = True)
        for i in range(len(trajectories)):
            if_smile = Chem.MolFromSmiles(trajectories[i])
            if if_smile is None:
                mol = ''
                samples.append(mol)
            else:
                fragment = gen_firstatom_frag(trajectories[i])
                mol = core_smi
                for i in range(core_smi.count('(*)')):
                    mol = join_frag(mol, fragment)
                    mol = usable_frag(mol)

                mol = mol + '.' + others
                samples.append(mol)

        rewards = self.get_reward(samples, self.predictor, **kwargs)
        reward_arr = np.array(rewards, dtype='float64')
        #print(reward_arr)
        pred_Reinforce = self.get_pred_val(samples, self.predictor, **kwargs)
        mean_pred_Reinforce = np.mean(pred_Reinforce)
        tensors = []
        self.generator.train()
        hidden = self.generator.init_hidden(1)
        
        def accumulate_loss(smile, reward, gamma, hidden):
            loss = 0
            traj_tensor = vocab.encode(vocab.tokenize(smile)).cuda()
            discounted = reward
            for p in range(len(traj_tensor)-1):
              logits, hidden = self.generator(traj_tensor[p].view(-1,1), hidden)
              log_probs = F.log_softmax(logits, dim=2)
              next_char = traj_tensor[p+1]
              loss += (log_probs[0, 0, next_char] * discounted)
              discounted = discounted * gamma
            return loss
        
        unq_factor = 0.75

        unique = list(set(trajectories))
        counter = np.zeros([len(unique)])



        for k in range(len(trajectories)):
            counter[unique.index(trajectories[k])] +=1

            rewards[k] = rewards[k] * (unq_factor ** (counter[unique.index(trajectories[k])]-1))
            l = accumulate_loss(trajectories[k], rewards[k], gamma, hidden)
            rl_loss -= l

        total_reward = np.sum(rewards)

        rl_loss = rl_loss / n_batch
        rl_loss.backward()
        
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                           grad_clipping)

        optimizer.step()
        
        return total_reward/n_batch, rl_loss.item(), mean_pred_Reinforce