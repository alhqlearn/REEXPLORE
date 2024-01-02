
# %env CUDA_VISIBLE_DEVICES=4


import sys
sys.path.append('./rl_agent')


import argparse
import os
import numpy as np
import pandas as pd
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import Chem
from models import RNN, OneHotRNN, EarlyStopping
from datasets import SmilesDataset, SelfiesDataset, SmilesCollate
from functions import decrease_learning_rate, print_update, track_loss, \
     sample_smiles, write_smiles


device = torch.device('cuda:4' if torch.cuda.is_available() else "cpu")


## seed all RNGs
seed = 0    # Mention seed value
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    print("using cuda")
    torch.cuda.manual_seed_all(seed)


# suppress Chem.MolFromSmiles error output
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


#Output directory
output_dir = './pre_trained_files'

# make output directories
if not os.path.isdir(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

# sample a set of SMILES from the final, trained model
sample_size = 1000          # (type=int, default=100000)
batch_size = 128              # (type=int, default=128)

dataset = SmilesDataset(smiles_file='./dataset/chembl.csv') # Dataset file name


# set up batching

batch_size = 128
loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=SmilesCollate(dataset.vocabulary))


model = RNN(vocabulary=dataset.vocabulary,
                rnn_type='GRU',                      # str; RNN type choices=['RNN', 'LSTM', 'GRU']
                embedding_size= 128,                 # int; embedding size
                hidden_size=512,                     # int; size of language model hidden layers
                n_layers=3,                          # int; number of layers in language model
                dropout=0,                           # float; amount of dropout (0-1) to apply to model
                bidirectional=False,                 # bool; for LSTMs only, train a bidirectional mode
                tie_weights=False,
                nonlinearity='tanh')


# set up optimizer


# optimization parameters
learning_rate = 0.001   # initial learning rate
learning_rate_decay = None   #amount (0-1) to decrease learning rate by every ' + \ 'fixed number of steps')
learning_rate_decay_steps = 10000       # Number of steps between learning rate decrements
log_every_epochs = 1000     #log training/validation losses every n epochs


optimizer = optim.Adam(model.parameters(),
                       betas=(0.9, 0.999), ## default
                       eps=1e-08, ## default
                       lr=learning_rate)


# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# set up early stopping
patience = 1000
early_stop = EarlyStopping(patience)

# set up training schedule file
sample_idx = 0   #index of the model being trained (zero-indexed)
sched_filename = "sched_file" + str(sample_idx + 1) + ".csv"
sched_file = os.path.join(output_dir, sched_filename)




max_epochs = 1000   #maximum number of epochs to train for
gradient_clip = None, # type=float, amount to which to clip the gradients

# manually deal with gradient clipping
try:
    gradient_clip = float(gradient_clip)
except (ValueError, TypeError):
    gradient_clip = None


smiles_filename = "sample_" + str(3) + "_SMILES.smi"
smiles_file = os.path.join(output_dir, smiles_filename)

def training_model_rnn():
    # iterate over epochs
    counter = 0
    for epoch in range(max_epochs):
        # iterate over batches
        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
            batch, lengths = batch

            # increment counter
            counter += 1

            # calculate loss
            log_p = model.loss(batch, lengths)
            loss = log_p.mean()

            # zero gradients, calculate new gradients, and take a step
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            if gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            # check learning rate decay
            if learning_rate_decay is not None and \
                    counter % learning_rate_decay_steps == 0:
                decrease_learning_rate(optimizer,
                                      multiplier=learning_rate_decay)

            # calculate validation loss
            validation, lengths = dataset.get_validation(batch_size)
            validation_loss = model.loss(validation, lengths).mean().detach()
            # check early stopping
            model_filename = "model_validation" + str(sample_idx + 1) + ".pth" ##model filename
            model_file = os.path.join(output_dir, model_filename)
            early_stop(validation_loss.item(), model, model_file, counter)

            if early_stop.stop:
                break

        # print update and write training schedule?
        if log_every_epochs is not None:
            #print_update(model, dataset, epoch, 'NA', loss.item(), batch_size)
            track_loss(sched_file, model, dataset, epoch,
                      counter, loss.item(), batch_size)

        if early_stop.stop:
            break

    # append information about final training step
    if log_every_epochs is not None:
        sched = pd.DataFrame({'epoch': [None],
                              'step': [early_stop.step_at_best],
                              'outcome': ['training loss'],
                              'value': [early_stop.best_loss]})
        sched.to_csv(sched_file, index=False, mode='a', header=False)


    # load the best model
    model.load_state_dict(torch.load(model_file))
    model.eval() ## enable evaluation modes

    # sample a set of SMILES from the final, trained model
    sampled_smiles = []
    while len(sampled_smiles) < sample_size:
        sampled_smiles.extend(model.sample(batch_size, return_smiles=True))

    # write sampled SMILES
    write_smiles(sampled_smiles, smiles_file)
    #print(sampled_smiles)

    def is_valid(smiles):
      mol = Chem.MolFromSmiles(smiles)
      if mol is not None and mol.GetNumAtoms()>0:
         return smiles


    mols = list(filter(is_valid,sampled_smiles))

    print('Percentage of validity = ' + str((len(mols)/len(sampled_smiles))*100))

## Save  vocab file
dataset.vocabulary.write('./pre_trained_files/vocab_chembl1')

## training the model
training_model_rnn()

## Save the model and vocab file

def save_model(model, path):
        torch.save(model.state_dict(), path)

path = './pre_trained_files/checkpoint_chembl1'
save_model(model, path)

print(dataset.vocabulary)







