import numpy as np
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem, DataStructs


import random
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(prediction, n_to_generate, path_dir = None):
    #prediction = np.asarray(prediction)
    prediction = tensor_to_array(prediction)
    print("Mean %yield/ee: ", np.round(prediction.mean(),2))
    #print("Proportion of valid SMILES:", len(prediction)/n_to_generate)
    plt.figure() #Create a new figure for the plot
    ax = sns.kdeplot(prediction, shade=True, label='Generated = ' + str(np.round(np.mean(prediction), 2)))
    ax.set(xlabel='Predicted output', 
           title='Distribution of predicted %yield/$ee$')
    plt.legend(bbox_to_anchor=(1.02, 1.02), loc='best')
    plt.tight_layout()        
    if path_dir is not None:
      plt.savefig(path_dir, dpi=300)
    return plt
    


def add_othercomponents(smile, components = 'C(C(C(F)(F)S(=O)(=O)F)(F)F)(C(F)(F)F)(F)F.CC(C)(C)N=P(N1CCCC1)(N2CCCC2)N3CCCC3'):
    return smile + '.' + components
    

def tensor_to_array(prediction):
    a = np.zeros([len(prediction)])
    for k in range(len(prediction)):
        a[k] = float(prediction[k][0])
    return a

def canonical_smiles(smiles, sanitize=True, throw_warning=False):
    new_smiles = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles
    



def generate_allfragments(smile):
    arr_smi = []
    frag = '(*)'
    for i in range(len(smile)):
        array = smile[:i] + frag + smile[i:]
        attempt = Chem.MolFromSmiles(array)
        if attempt is not None:
            arr_smi.append(array)

    return arr_smi

def gen_firstatom_frag(smile): 
    frag = '(*)'
    if len(smile) == 0:
        return smile
    if len(smile) == 1:
        return smile + frag
    else:
        atoms = ['C', 'B', 'O', 'N', 'P', 'S', 'c', 'o', 'n', 'p', 's']
        i= 0
        for i in range(len(smile)):
            if smile[i] in atoms:
                array = smile[:i] + smile[i] + frag + smile[i+1:]
                attempt = Chem.MolFromSmiles(array)
                if attempt is not None:
                    return array 
                else:
                    continue
            else:
                continue
        if i == len(smile)-1:
            return ''

            
def usable_frag(f):
    if f.find('*') == -1:
        return f
    if f.find('(*)') != -1:
        return f
    else:
        idx = f.find('*')
        joined = f[idx+1] + '(*)' + f[(idx+2):]
        return joined


def join_frag_list(core_smi, frag_list):
    combined = []
    for l in range(len(frag_list)):
        frag_list[l] = frag_list[l].replace('(*)', '9')
        combined.append(core_smi + '.' + frag_list[l])

    for n in range(len(frag_list)):
        combined[n] = combined[n].replace('(*)', '9', 1)

    all_list = []
    for m in range(len(frag_list)):
        mol = Chem.MolFromSmiles(combined[m])
        all_list.append(Chem.MolToSmiles(mol))
    return all_list


def join_frag(core_smi, frag):
    
    if frag == '':
        return ''
    frag = frag.replace('(*)', '%99')
    combined = core_smi + '.' + frag
    combined = combined.replace('(*)', '%99', 1)
    mol = Chem.MolFromSmiles(combined)
    
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else: 
        return 'This is an error' + 'Core smile: ' + core_smi + 'Frag: ' + frag


def permute(og_core, flist):
   
    fragmentlist = []
    for a in range(len(flist)):
        fragmentlist += generate_allfragments(flist[a])
    print("Total nm value:", len(fragmentlist))

    # Combined approach
    allfinal = []
    first = og_core
    for d in range(len(fragmentlist)):
        first = og_core
        first = join_frag(first, fragmentlist[d])
        first = usable_frag(first)
        for e in range(len(fragmentlist)):
            second = join_frag(first, fragmentlist[e])
            second = usable_frag(second)
            for f in range(len(fragmentlist)):
                third = join_frag(second, fragmentlist[f])
                third = usable_frag(third)
                allfinal.append(third)

    allfinal = np.array(allfinal)
    allfinal = np.unique(allfinal)
    return allfinal


