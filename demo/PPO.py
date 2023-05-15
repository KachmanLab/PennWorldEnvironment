import torch
import torch.nn as nn
from RolloutBuffer import RolloutBuffer
from ActorCritic import ActorCritic
from copy import deepcopy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, 
                 K_epochs, eps_clip, has_continuous_action_space, 
                 variable_action_space, device, num_units_actor=64, num_units_critic=64, 
                 num_layers_actor=2, num_layers_critic=2, action_std_init=0.6):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.device = device
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.has_continuous_action_space = has_continuous_action_space
        self.variable_action_space = variable_action_space
        self.device = device
        self.num_units_actor = num_units_actor
        self.num_units_critic = num_units_critic
        self.num_layers_actor = num_layers_actor
        self.num_layers_critic = num_layers_actor
        

        if has_continuous_action_space:
            self.action_std = action_std_init

        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, num_units_actor, num_units_critic, num_layers_actor, num_layers_critic, self.device).to(self.device)
        # self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, num_units_actor, num_units_critic, num_layers_actor, num_layers_critic, self.device).to(self.device)
        # self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state, action_mask=None, debug=False):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state, debug)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                if action_mask is not None:
                    action_mask = torch.BoolTensor(action_mask).to(self.device)
                action, action_logprob = self.policy_old.act(state, action_mask, debug)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.action_masks.append(action_mask)

            if debug:
                print(f"action: {action.item()}")

            # if there is masking, offset needs to be applied to map index in action space to index in graph
            if self.variable_action_space:
                offset = (action_mask == False).nonzero(as_tuple=True)[0][0].item()
            else:
                offset = 0
            return action.item() - offset

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        if self.variable_action_space:
            old_masks = torch.squeeze(torch.stack(self.buffer.action_masks, dim=0)).detach().to(self.device)
        else:
            old_masks = None

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy, _ = self.policy.evaluate(old_states, old_actions, old_masks)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        kwargs = {
            "state_dim": self.policy.state_dim,
            "action_dim": self.policy.action_dim,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "gamma": self.gamma,
            "K_epochs": self.K_epochs,
            "eps_clip": self.eps_clip,
            "has_continuous_action_space": self.has_continuous_action_space,
            "variable_action_space": self.variable_action_space,
            "device": self.device,
            "num_units_actor": self.num_units_actor,
            "num_units_critic": self.num_units_critic,
            "num_layers_actor": self.num_layers_actor,
            "num_layers_critic": self.num_layers_critic
        }

        torch.save([kwargs, deepcopy(self.policy_old.state_dict())], checkpoint_path)

    @classmethod
    def load(PPO, path):

        kwargs, state_dict = torch.load(path)
        ppo_agent = PPO(**kwargs)
        ppo_agent.policy_old.load_state_dict(state_dict)

        return ppo_agent