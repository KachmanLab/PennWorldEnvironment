import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, num_units_actor, num_units_critic, num_layers_actor, num_layers_critic, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
            self.action_std_init = action_std_init
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, num_units_actor),
                            nn.Tanh(),
                            nn.Linear(num_units_actor, num_units_actor),
                            nn.Tanh(),
                            nn.Linear(num_units_actor, action_dim),
                        )
        else:
            # create list of layers
            layers = nn.ModuleList()
            input_size = state_dim
            for _ in range(num_layers_actor):
                layers.append(nn.Linear(input_size, num_units_actor))
                input_size = num_units_actor
                layers.append(nn.Tanh())

            # output layer
            layers.append(nn.Linear(num_units_actor, action_dim))
            layers.append(nn.Softmax(dim=-1))

            # convert to Sequential module
            self.actor = nn.Sequential(*layers)

                # if activation is not None:
                #     assert isinstance(activation, Module), \
                #         "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                #         self.layers.append(activation)

            ##### OLD ########
            # self.actor = nn.Sequential(
            #                 nn.Linear(state_dim, 64),
            #                 nn.Tanh(),
            #                 nn.Linear(64, 64),
            #                 nn.Tanh(),
            #                 nn.Linear(64, action_dim),
            #                 nn.Softmax(dim=-1)
            #             )
            ##### OLD ########

        # critic
        # create list of layers
        layers = nn.ModuleList()
        input_size = state_dim
        for _ in range(num_layers_critic):
            layers.append(nn.Linear(input_size, num_units_critic))
            input_size = num_units_critic  
            layers.append(nn.Tanh())

        # output layer with estimate of future discounted expected return of state as output 
        layers.append(nn.Linear(num_units_critic, 1))

        # convert to Sequential module
        self.critic = nn.Sequential(*layers)

        ##### OLD ########
        # self.critic = nn.Sequential(
        #                 nn.Linear(state_dim, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 1)
        #             )
        ##### OLD ########
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, action_mask=None, debug=False):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            # mask actions for variable action spaces
            if action_mask is not None:
                # action_mask = torch.tensor(action_mask).bool().to(self.device)
                action_probs = torch.masked_fill(action_probs, action_mask, 0)
                # print(f"action probs in act(): {action_probs}")

                if debug: 
                    print(f"action_probs: {action_probs}")

            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if debug: 
            print(f"action log_prob: {action_logprob}")
            print(f"action: {action}")
        
        return action.detach(), action_logprob.detach()
    
    
    def evaluate(self, state, action, action_mask):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            # print(f"state in evaluate: {state}")
            action_probs = self.actor(state)
            # mask actions for variable action spaces
            if action_mask is not None:
                # print(f"action mask: {action_mask}")
                # print(f"action probs in evaluate before mask(): {action_probs}")
                action_probs = torch.masked_fill(action_probs, action_mask, 0)
                # print(f"action_probs in evaluate: {action_probs}")
            torch.set_printoptions(profile="full")
            # print(f"action probs in evaluate(): {action_probs}")
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        dist_probs = dist.probs
        state_values = self.critic(state)
        # print(f"dist_probs in AC: {dist_probs}")
        
        return action_logprobs, state_values, dist_entropy, dist_probs
