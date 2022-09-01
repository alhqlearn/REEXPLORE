"""
Clean and canonicalize SMILES from chemical structure databases and write them
to a line-delimited file.
"""

import os
import numpy as np
import pandas as pd
import sys
from itertools import chain
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from tqdm import tqdm

# set working directory
git_dir = os.path.expanduser("~/git/low-data-generative-models")
python_dir = git_dir + "/python"
os.chdir(python_dir)
sys.path.append(python_dir)

# import functions
from functions import clean_mols, remove_salts_solvents, read_smiles, \
    NeutraliseCharges
# import Vocabulary
from datasets import Vocabulary

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# read SMILES
basename = os.path.basename(input_file)
if "chembl" in basename.lower():
    # read ChEMBL chemical representations
    chembl = pd.read_csv(input_file, sep='\t')
    # get canonical SMILES
    smiles = chembl['canonical_smiles'].values
elif "gdb" in basename.lower():
    # read GDB chemical representations
    gdb13 = pd.read_csv(input_file, sep='\t', header=None)
    # get canonical SMILES
    smiles = gdb13[[0]].values
elif ".sdf" in basename.lower():
    # read all metabolites
    suppl = SDMolSupplier(input_file)
    mols = [x for x in suppl if not x is None]
    # convert back to SMILES for the rest of the preprocessing steps
    smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in mols]
elif "ymdb" in input_file.lower():
    # read YMDB
    ymdb = pd.read_csv(input_file)
    ymdb = ymdb.dropna()
    smiles = ymdb['smiles']
elif "coconut" in input_file.lower():
    # read COCONUT (MetFrag-format CSV)
    coconut = pd.read_csv(input_file)
    smiles = coconut['clean_smiles'].values
else:
    smiles = read_smiles(input_file)

# remove duplicated SMILES
smiles = np.unique(smiles)
# record original count
initial_count = len(smiles)
print("parsing " + str(initial_count) + " unique SMILES")

# convert to molecules
mols = clean_mols(smiles, stereochem=False)
# remove molecules that could not be parsed
mols = [mol for mol in mols if mol]
print("parsed " + str(len(mols)) + " unique, valid canonical SMILES")

# remove salts/solvents
mols = [remove_salts_solvents(mol, hac=3) for mol in tqdm(mols)]
# remove molecules that could not be parsed
mols = [mol for mol in mols if mol]
# remove charges
mols = [NeutraliseCharges(mol) for mol in tqdm(mols)]
print("parsed " + str(len(mols)) + \
      " molecules with >3 heavy atoms and 1 fragment")

# remove molecules with invalid atoms
## what unique atoms are present in any molecule?
elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in mols]
counts = np.unique(list(chain(*elements)), return_counts=True)
## define valid symbols
valid = set(['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S'])
mols = [mols[idx] for idx, atoms in enumerate(elements) if \
        len(set(atoms) - valid) == 0]
print("parsed " + str(len(mols)) + \
      " molecules with all valid atoms (C/N/O/P/S/F/Br/Cl/I)")

# convert back to SMILES
smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in tqdm(mols)]
smiles = np.unique(smiles)

# print the vocabulary
vocabulary = Vocabulary(smiles=smiles)
print("vocabulary of {} characters:".format(len(vocabulary)))
print(vocabulary.characters)

# remove any molecules containing tokens found in <0.01% of molecules,
# or five or fewer molecules
vocab_before = len(vocabulary)
n_smiles = len(smiles)
for token in vocabulary.characters:
    token_smiles = [sm for sm in smiles if token in vocabulary.tokenize(sm)]
    pct_smiles = len(token_smiles) / n_smiles
    if pct_smiles < 0.01 / 100 or len(token_smiles) <= 10:
        # remove from SMILES
        smiles = list(set(smiles).difference(token_smiles))

# recreate the vocabulary and print new dataset size
vocabulary = Vocabulary(smiles=smiles)
vocab_after = len(vocabulary)
print("after removing tokens found in <0.01% of molecules, {} remain".format(
        len(smiles)))
print("updated vocabulary of {} (of {}) characters:".format(
        vocab_after, vocab_before))
print(vocabulary.characters)

# write to line-delimited file
with open(output_file, 'w') as f:
    for sm in smiles:
        f.write(sm + '\n')

print("wrote " + str(len(smiles)) + " SMILES to output file: " + output_file)
