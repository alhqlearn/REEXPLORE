
import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem


class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward):
       
        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward

    def policy_gradient(self, vocab, n_batch=2000, gamma=0.99,
                        std_smiles=False, grad_clipping=None, core_smi='OC(*)', others = 'C(C(C(F)(F)S(=O)(=O)F)(F)F)(C(F)(F)F)(F)F.CC(C)(C)N=P(N1CCCC1)(N2CCCC2)N3CCCC3', **kwargs):

        # New part begins
        optimizer = optim.Adam(self.generator.parameters(),
                       betas=(0.9, 0.999), ## default
                       eps=1e-08, ## default
                       lr=0.001)
        rl_loss = 0
        optimizer.zero_grad()
        total_reward = 0
        samples = []

        trajectories = self.generator.sample(n_batch, max_len=100, return_smiles = True)
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
        
        return total_reward/n_batch, rl_loss.item()