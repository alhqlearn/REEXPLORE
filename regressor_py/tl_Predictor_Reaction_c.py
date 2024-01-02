from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') # switch off RDKit warning messages

from sklearn.model_selection import train_test_split

import sys
sys.path.append('./fastai1/')
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


#Define a custom tokenizer


# Don't include the defalut specific token of fastai, only keep the padding token
BOS,EOS,FLD,UNK,PAD = 'xxbos','xxeos','xxfld','xxunk','xxpad'
TK_MAJ,TK_UP,TK_REP,TK_WREP = 'xxmaj','xxup','xxrep','xxwrep'
defaults.text_spec_tok = [PAD]



special_tokens = ['[BOS]', '[C@H]', '[C@@H]','[C@]', '[C@@]','[C-]','[C+]', '[c-]', '[c+]','[cH-]',
                   '[nH]', '[N+]', '[N-]', '[n+]', '[n-]' '[NH+]', '[NH2+]', '[O-]', '[S+]', '[s+]',
                   '[S-]', '[O+]', '[SH]', '[B-]','[BH2-]', '[BH3-]','[b-]', '[PH]','[P+]', '[I+]', 
                   '[Si]','[SiH2]', '[Se]','[SeH]', '[se]', '[Se+]', '[se+]','[te]','[te+]', '[Te]',
                   '[Pd]' , '[Ag]','[Cs]','[Li]','[K]','[Na]', '[N@]', '[N@@]', '[S@+]', '[K+]', 
                   '[Ni+2]', '[Mg]','[Li+]', '[Cl-]', '[Ni]','[Cs+]', '[Cu+2]', '[Zn+2]', '[Al]', '[Cu]']




class MolTokenizer(BaseTokenizer):
    def __init__(self, lang = 'en', special_tokens = special_tokens):
        self.lang = lang
        self.special_tokens = special_tokens
        
    def tokenizer(self, smiles):
        # add specific token '[BOS]' to represetences the start of SMILES
        smiles = '[BOS]' + smiles
        regex = '(\[[^\[\]]{1,10}\])'
        char_list = re.split(regex, smiles)
        tokens = []
        
        if self.special_tokens:
            for char in char_list:
                if char.startswith('['):
                    if char in special_tokens:
                        tokens.append(str(char))
                    else:
                        tokens.append('[UNK]')
                else:
                    chars = [unit for unit in char]
                    [tokens.append(i) for i in chars]                    
        
        if not self.special_tokens:
            for char in char_list:
                if char.startswith('['):
                    tokens.append(str(char))
                else:
                    chars = [unit for unit in char]
                    [tokens.append(i) for i in chars]
                
        #fix the 'Br' be splited into 'B' and 'r'
        if 'B' in tokens:
            for index, tok in enumerate(tokens):
                if tok == 'B':
                    if index < len(tokens)-1: # make sure 'B' is not the last character
                        if tokens[index+1] == 'r':
                            tokens[index: index+2] = [reduce(lambda i, j: i + j, tokens[index : index+2])]
        
        #fix the 'Cl' be splited into 'C' and 'l'
        if 'l' in tokens:
            for index, tok in enumerate(tokens):
                if tok == 'l':
                    if tokens[index-1] == 'C':
                            tokens[index-1: index+1] = [reduce(lambda i, j: i + j, tokens[index-1 : index+1])]
        return tokens    
    
    def add_special_cases(self, toks):
        pass


def randomize_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True, kekuleSmiles=False)

def test_smiles_augmentation(df, N_rounds):
    dist_aug = {col_name: [] for col_name in df}

    for i in range(df.shape[0]):
        for j in range(N_rounds):
            dist_aug['smiles'].append(randomize_smiles(df.iloc[i].smiles))
            dist_aug['ee'].append(df.iloc[i]['ee'])
    df_aug = pd.DataFrame.from_dict(dist_aug)

    return pd.DataFrame.from_dict(dist_aug)    

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False   


def fn_smiles_augmentation(df, N_rounds, noise):
    '''
    noise: add gaussion noise to the label
    '''
    dist_aug = {col_name: [] for col_name in df}

    for i in range(df.shape[0]):
        for j in range(N_rounds):
            dist_aug['smiles'].append(randomize_smiles(df.iloc[i].smiles))
            dist_aug['ee'].append(df.iloc[i]['ee'] + np.random.normal(0,noise))
    df_aug = pd.DataFrame.from_dict(dist_aug)
    df_aug = df_aug.append(df, ignore_index=True)
    return df_aug.drop_duplicates('smiles')




def pred_init(seed, batch_size, filename, current_path,augm, drp_out, sigm_g):

    #data_path = Path(current_path)
    name = 'regressor'
    path = Path(name)
    path.mkdir(exist_ok=True, parents=True)

    data = filename
    print('Dataset:', data.shape)


    """### Target task regressor fine-tuning on target task LM

    Train-validation-test splits

    - Split the data into train-validation-test sets
    - Validation set is used for hyperparameter tuning
    - Test set is used for the final performance evaluation
    """

    random_seed(seed, True)

    train_ , test = train_test_split(data, test_size=0.20, random_state=0)
    train, valid = train_test_split(train_, test_size=0.125, random_state=0)

    print(train_.shape)
    print(train.shape)
    print(test.shape)
    print(valid.shape)


    """### SMILES augmentation for regression task

    - For the regression task, a gaussian noise (with mean zero and standard deviation, Ïƒg_noise) is added to the labels of the augmented SMILES during the training
    - The number of augmented SMILES and Ïƒg_noise is tuned on the validation set
    """

    random_seed(seed, True)

    train_aug = fn_smiles_augmentation(train, augm, noise=sigm_g)
    #print("Train_aug: ", train_aug.shape)

    ### Data pre-processing

   
    bs = batch_size
    tok = Tokenizer(partial(MolTokenizer, special_tokens = special_tokens), n_cpus=6, pre_rules=[], post_rules=[])

    """Adpot the encoder of the pre-trained LM according to the target dataset


    """

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
    random_seed(seed, True)

    lm_vocab = TextLMDataBunch.from_df(path, train_aug, valid, bs=bs, tokenizer=tok, 
                                  chunksize=50000, text_cols=0, label_cols=1, max_vocab=60000, include_bos=False, min_freq=1, num_workers=0)
    #print(f'Vocab Size: {len(lm_vocab.vocab.itos)}')



    pretrained_model_path = Path('./pre_trained_files/')

    pretrained_fnames = ['regressor_pre_trained_wt', 'regressor_pre_trained_vocab']
    fnames = [pretrained_model_path/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]

    random_seed(seed, True)

    lm_learner = language_model_learner(lm_vocab, AWD_LSTM, config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.1,
                              hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True), drop_mult=drp_out, pretrained=False)
    lm_learner = lm_learner.load_pretrained(*fnames)
    lm_learner.freeze()
    lm_learner.save_encoder(f'lm_encoder')

    lm_learner.model

    """Create a text databunch for regression:

    - It takes as input the train and validation data
    - Pass the vocab of the pre-trained LM as defined in the previous step
    - Specify the column containing text data and output
    - Define the batch size according to the GPU memory available

    """

    random_seed(seed, True)

    data_clas = TextClasDataBunch.from_df(path, train_aug, valid, bs=bs, tokenizer=tok, 
                                              chunksize=50000, text_cols='smiles',label_cols='ee', 
                                              vocab=lm_vocab.vocab, max_vocab=60000, include_bos=False, min_freq=1, num_workers=0)

    #print(f'Vocab Size: {len(data_clas.vocab.itos)}')

    """### Training the regression model

    Create a learner for regression:

    - Pass the databunch
    - Load the encoder of the pre-trained LM
    - The drop_mult hyperparameter can be tuned
    - The model is evaluated using RMSE and R-squared value as error metric
    """

    random_seed(seed, True)

    reg_learner = text_classifier_learner(data_clas, AWD_LSTM,  config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.4,
                                hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5), pretrained=False, drop_mult=drp_out, metrics = [r2_score, rmse])
    reg_learner.load_encoder(f'lm_encoder')
    reg_learner.freeze()
    
    return reg_learner, train_aug, valid

# Training the regressor with stepwise unfreezing 

def train_reg(unf1, unf2, unf3, unf4, reg_learner_tr):
    reg_learner_tr.fit_one_cycle(unf1, 3e-2, moms=(0.8,0.7))

    reg_learner_tr.freeze_to(-2)
    reg_learner_tr.fit_one_cycle(unf2, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

    reg_learner_tr.freeze_to(-3)
    reg_learner_tr.fit_one_cycle(unf3, slice(5e-4/(2.6**4),5e-4), moms=(0.8,0.7))

    """The regressor is fine-tuned all at once without any frozen weights (i.e., no gradual unfreezing)"""

    reg_learner_tr.unfreeze()
    reg_learner_tr.fit_one_cycle(unf4, slice(5e-5/(2.6**4),5e-5), moms=(0.8,0.7))
    
    return reg_learner_tr


def train_reg1(unf1, unf2, unf3, unf4, lr1, lr2, lr3, lr4, reg_learner_tr):
    reg_learner_tr.fit_one_cycle(unf1, lr1, moms=(0.8,0.7))

    reg_learner_tr.freeze_to(-2)
    reg_learner_tr.fit_one_cycle(unf2, lr2, moms=(0.8,0.7))

    reg_learner_tr.freeze_to(-3)
    reg_learner_tr.fit_one_cycle(unf3, lr3, moms=(0.8,0.7))

    """The regressor is fine-tuned all at once without any frozen weights (i.e., no gradual unfreezing)"""

    reg_learner_tr.unfreeze()
    reg_learner_tr.fit_one_cycle(unf4, lr4, moms=(0.8,0.7))
    
    return reg_learner_tr


def test_performance(seed, batch_size, filename, train_aug, valid, current_path, drp_out, sigm_g):
    
    #data_path = Path(current_path)
    name = 'regressor'
    path = Path(name)
    path.mkdir(exist_ok=True, parents=True)

    
    random_seed(seed, True)

    bs = batch_size
    tok = Tokenizer(partial(MolTokenizer, special_tokens = special_tokens), n_cpus=6, pre_rules=[], post_rules=[])
    lm_vocab = TextLMDataBunch.from_df(path, train_aug, valid, bs=bs, tokenizer=tok, 
                                  chunksize=50000, text_cols=0, label_cols=1, max_vocab=60000, include_bos=False, min_freq=1, num_workers=0)
    #print(f'Vocab Size: {len(lm_vocab.vocab.itos)}')


    tok_new = TokenizeProcessor(tokenizer=tok, chunksize=50000, include_bos=False)
    num_new = NumericalizeProcessor(vocab=lm_vocab.vocab, max_vocab=60000, min_freq=1) 


    train_ , test_ = train_test_split(filename, test_size=0.2, random_state=0)
    train , test = train_test_split(filename, test_size=0.2, random_state= 0)

    train_.insert(2, 'valid', False)
    test_.insert(2, 'valid', True)
    df = pd.concat([train_, test_])

    random_seed(seed, True)

    preds = []

    # Randomized SMILES Predictions
    for i in range(4):
        np.random.seed(12*i)
        test_aug = test_smiles_augmentation(test, 1)

        #model

        test_aug.insert(2, 'valid', True)
        df_aug = test_aug
        #train.insert(2, 'valid', False)
        #df_aug = pd.concat([train, test_aug])
        test_db = (TextList.from_df(df_aug, path, cols='smiles', processor=[tok_new, num_new]).split_from_df(col='valid').label_from_df(cols='ee', label_cls=FloatList).databunch(bs=bs))

        learner = text_classifier_learner(test_db, AWD_LSTM, config=None, pretrained=False, drop_mult=drp_out, metrics = [r2_score, rmse])

        learner.load(f'reg_Reaction_c'); 

        #get predictions
        pred,lbl = learner.get_preds(ordered=True)

        #print(len(pred),len(lbl), 'augmented')
        #print(pred,lbl)
        preds.append(pred)

    # Canonical SMILES Predictions

    test_db = (TextList.from_df(df, path, cols='smiles', processor=[tok_new, num_new]).split_from_df(col='valid').label_from_df(cols='ee', label_cls=FloatList).databunch(bs=bs))

    learner = text_classifier_learner(test_db, AWD_LSTM, config=None, pretrained=False, drop_mult=drp_out, metrics = [r2_score, rmse])

    learner.load(f'reg_Reaction_c');


    #get predictions
    pred_canonical,lbl = learner.get_preds(ordered=True)
    #print(len(pred_canonical),len(lbl), 'canonical')    
    preds.append(pred_canonical)



    """The test set performance is evaluated using the predictions based on the canonical SMILES as well as that employing test-time augmentation"""

    print('Test Set (Canonical)')
    print('RMSE:', root_mean_squared_error(pred_canonical,lbl))
    print('MAE:', mean_absolute_error(pred_canonical,lbl))
    print('R2:', r2_score(pred_canonical,lbl))

    avg_preds = sum(preds)/len(preds)
    #print('\n')
    print('Test Set (Average)')
    print('RMSE:', root_mean_squared_error(avg_preds,lbl))
    print('MAE:', mean_absolute_error(avg_preds,lbl))
    print('R2:', r2_score(avg_preds,lbl))

    return root_mean_squared_error(pred_canonical,lbl), root_mean_squared_error(avg_preds,lbl)

def predictor(smiles, seed, batch_size, filename, train_aug, valid, current_path, drp_out, sigm_g):
    
    #data_path = Path(current_path)
    name = 'regressor'
    path = Path(name)
    path.mkdir(exist_ok=True, parents=True)

    
    random_seed(seed, True)

    bs = batch_size
    tok = Tokenizer(partial(MolTokenizer, special_tokens = special_tokens), n_cpus=6, pre_rules=[], post_rules=[])
    lm_vocab = TextLMDataBunch.from_df(path, train_aug, valid, bs=bs, tokenizer=tok, 
                                  chunksize=50000, text_cols=0, label_cols=1, max_vocab=60000, include_bos=False, min_freq=1, num_workers=0)
    #print(f'Vocab Size: {len(lm_vocab.vocab.itos)}')


    tok_new = TokenizeProcessor(tokenizer=tok, chunksize=50000, include_bos=False)
    num_new = NumericalizeProcessor(vocab=lm_vocab.vocab, max_vocab=60000, min_freq=1) 


    
    train_ , test_ = train_test_split(filename, test_size=0.2, random_state=0)
    
    test_ = test_[:1]
    train , test = train_test_split(filename, test_size=0.2, random_state= 0)

    train_.insert(2, 'valid', False)
    test_.insert(2, 'valid', True)
    df = pd.concat([train_, test_])
    
    gen_smile = smiles
    df_gen_smile = pd.DataFrame(gen_smile, columns = ['smiles'])
   
    df_gen_smile.insert(1, 'ee', '0')
    df_gen_smile.insert(2, 'valid', True)
   
    df_smile = pd.concat([df,df_gen_smile])
    
    test_db = (TextList.from_df(df_smile, path, cols='smiles', processor=[tok_new, num_new]).split_from_df(col='valid').label_from_df(cols='ee', label_cls=FloatList).databunch(bs=bs))

    learner = text_classifier_learner(test_db, AWD_LSTM, config=None, pretrained=False, drop_mult=drp_out, metrics = [r2_score, rmse])

    learner.load(f'reg_Reaction_c');


    #get predictions
    pred_canonical,lbl = learner.get_preds(ordered=True)
    #print(len(pred_canonical),len(lbl), 'canonical')    
    

    return gen_smile, pred_canonical[1:]



