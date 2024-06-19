import torch
from torch import Tensor
from torch.autograd import Variable
import torch as T
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import numpy as np


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr_a=0.01, lr_c=0.01, discrete_action=True):
        # lr = 0.01
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.num_out_pol = num_out_pol
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_a)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_c)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, noise_std, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                # action += Variable(Tensor(self.exploration.noise()),
                #                    requires_grad=False)

                # 将OU噪声改为高斯噪声
                # noise = np.random.normal(0, noise_std, size=action.shape)
                # action += Variable(Tensor(noise), requires_grad=False)
                noise = T.tensor(np.random.normal(0, noise_std, size=action.shape), dtype=T.float32)
                action += noise

            action = action.clamp(-1, 1)

        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class TD3Agent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """

    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr_a=0.01, lr_c=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic_1 = MLPNetwork(num_in_critic, 1,
                                   hidden_dim=hidden_dim,
                                   constrain_out=False)
        self.critic_2 = MLPNetwork(num_in_critic, 1,
                                   hidden_dim=hidden_dim,
                                   constrain_out=False)

        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic_1 = MLPNetwork(num_in_critic, 1,
                                          hidden_dim=hidden_dim,
                                          constrain_out=False)
        self.target_critic_2 = MLPNetwork(num_in_critic, 1,
                                          hidden_dim=hidden_dim,
                                          constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic_1, self.critic_1)
        hard_update(self.target_critic_2, self.critic_2)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_a)
        self.critic_optimizer = Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr_c)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, noise_std, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                # action += Variable(Tensor(self.exploration.noise()),
                #                    requires_grad=False)

                # 将OU噪声改为高斯噪声
                noise = T.tensor(np.random.normal(0, noise_std, size=action.shape), dtype=T.float32)
                action += noise

            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic_1': self.critic_1.state_dict(),
                'critic_2': self.critic_2.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic_1': self.target_critic_1.state_dict(),
                'target_critic_2': self.target_critic_2.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic_1.load_state_dict(params['critic_1'])
        self.critic_2.load_state_dict(params['critic_2'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic_1.load_state_dict(params['target_critic_1'])
        self.target_critic_2.load_state_dict(params['target_critic_2'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
