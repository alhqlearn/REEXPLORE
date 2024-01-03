## Overview
<img align="middle" width="100%" src="reexplore.png">
In this work, we demonstrate how reward shaping can render a policy gradient reinforcement learning (RL) approach a valuable tool for reaction discovery. Whereas we deploy RL to navigate the generation of novel practical molecules towards higher yield/selectivity regions, yield due to the newly generated molecules is predicted using a transfer learning model.

### Prerequisites
- Python 3.7 (Anaconda)
- PyTorch 1.12.1
- CUDA 11.3

### Environmental Setup

```
conda create --name REEXPLORE python=3.7.16
conda activate REEXPLORE
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install seaborn
pip install scikit-learn
pip install tqdm
pip install yaml
pip install fastprogress
pip install spacy
pip install PyTDC
pip install networkx
pip install fcd-torch
```
### Git repository
Please clone two existing repositories (fastai and synthetic complexity score) after creating your environment with Python 3.7.16.
```
https://github.com/fastai/fastai1.git
```
```
https://github.com/connorcoley/scscore.git
```
### Preparation
We provided our pre-trained model and large datasets in the following link (due to the heavy file size)--
## Training
### Pre-training of RL agent 
```
python rl_agent_training.py --dataset <type>
```
`<type>` is one of the large datasets ['ChEMBL', 'ZINC', 'COCONUT'].
### Pre-training of surrogate regressor 
```
python regressor_training.py --dataset <type>
```
`<type>` is one of the reaction datasets ['Reaction_a', 'Reaction_b', 'Reaction_c']. 

**NB:** *Change the hyper-parameters regarding the datasets like the number of augmented smiles, dropout ratios, etc.*
### Training of RL agent under reaction-tailored reward functionalities
Once you are ready with the trained RL agent and regressor, use the following commands-

Reaction A
```
python Reaction_A_rl_training.py --agent_dataset ChEMBL --trajectories_method random --num_trajectories 1000
```
Reaction B
```
python Reaction_B_rl_training.py --agent_dataset ChEMBL --trajectories_method random --num_trajectories 1000
```
Reaction C
```
python Reaction_C_rl_training.py --agent_dataset ChEMBL --trajectories_method random --num_trajectories 1000
```
