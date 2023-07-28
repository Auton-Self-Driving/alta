"""pytorch networks for multi-agent RL algo
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical, Normal
#from transformers import BertConfig, BertModel

from agent_utils import set_init

class Basic_Discrete(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Basic_Discrete, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s, deterministic=False):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).detach()
        m = self.distribution(prob)
        if deterministic:
            return m.mean.cpu().numpy()[0]
        return m.sample().cpu().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

class PPOActorCritic_Continuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_std=.5, use_transformer=False, squash=False):
        super(PPOActorCritic_Continuous, self).__init__()
        # action mean range -1 to 1
        self.N_S = state_dim
        self.N_A = action_dim
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.N_S = 800
            state_dim = 128
            # state_dim = 800
            self.transformer = TransformerAgent(state_dim)
            # self.transformer = TransformerAgent(128)
        self.squash = squash
        self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh()
            )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        self.action_var = torch.full(
            (action_dim,) , action_std * action_std)

    def forward(self):
        raise NotImplementedError('please use act and eval instead')

    def act(self, state, deterministic=False):
        device = next(self.actor.parameters()).device

        if self.use_transformer:
            state = self.transformer(state.to(device))

        action_mean = self.actor(state.to(device))
        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean.to(torch.device('cpu')), cov_mat.to(torch.device('cpu')))

        action = dist.mean if deterministic else dist.sample()

        if self.squash:
            action = torch.tanh(action)
        action_logprobs = get_action_log_prob(dist,self.squash,action)

        action = action.detach().cpu().numpy()
        action_logprobs = action_logprobs.detach().cpu().numpy()

        return action, action_logprobs

    def evaluate(self, state, action):
        device = next(self.actor.parameters()).device
        if self.use_transformer:
            state = self.transformer(state.to(device))
        action_mean = self.actor(state.to(device))
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        
        dist = MultivariateNormal(action_mean.to(torch.device('cpu')), cov_mat.to(torch.device('cpu')))

        action_logprobs = get_action_log_prob(dist,self.squash,action.to(torch.device('cpu')))

        dist_entropy = dist.entropy()
        dist_entropy = dist_entropy.to(device)
        action_logprobs = action_logprobs.to(device)

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy.view(-1)

    def __str__(self):
        device = next(self.actor.parameters()).device
        return 'PPOActorCritic_Continuous:\n ' + \
            'device: {}\n actor: {}\n critic: {}\n'.format(device,
            self.actor, self.critic)

class printer(nn.Module):

    def __init__(self):
        super(printer, self).__init__()

    def forward(self,x):
        print(x[...,:,:10])
        return x

class PPOActorCritic_Mixed(nn.Module):
    def __init__(self, state_dim, action_dim, discrete_actions, action_std=.5, use_transformer=False, squash=False):
        super(PPOActorCritic_Mixed, self).__init__()
        # action mean range -1 to 1
        self.N_S = state_dim
        self.N_A = action_dim
        self.discrete_actions = discrete_actions
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.N_S = 800
            state_dim = 128
            # state_dim = 800
            self.transformer = TransformerAgent(state_dim)
            # self.transformer = TransformerAgent(128)
        self.squash = squash
        
        # First action_dim - 1 outputs are continuous 
        # Last discrete_actions outputs make a categorical distribution
        self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim + discrete_actions - 1),
            )

        self.actor_continuous_head = nn.Sequential(
            nn.Tanh()
        )
        self.actor_discrete_dist = Categorical

        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        self.action_var = torch.full(
            (action_dim-1,) , action_std * action_std)

    def forward(self):
        raise NotImplementedError('please use act and eval instead')

    def act(self, state, deterministic=False):
        device = next(self.actor.parameters()).device

        if self.use_transformer:
            state = self.transformer(state.to(device))

        raw_action = self.actor(state.to(device))

        prob = F.softmax(raw_action[...,self.N_A-1:], dim=1).detach()
        m = self.actor_discrete_dist(prob.cpu())
        # print(prob, raw_action)
        if deterministic:
            discrete_action =  torch.argmax(prob).cpu()
            if len(list(discrete_action.shape)) == 0:
                discrete_action = discrete_action.reshape((1,))
        else:
            discrete_action = m.sample().cpu() #.numpy()[0]

        discrete_logprobs = m.log_prob(discrete_action).cpu().numpy()
        discrete_action = discrete_action.cpu().numpy()

        action_mean = self.actor_continuous_head(raw_action[...,:self.N_A-1])
        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean.to(torch.device('cpu')), cov_mat.to(torch.device('cpu')))

        action = dist.mean if deterministic else dist.sample()

        if self.squash:
            action = torch.tanh(action)
        action_logprobs = get_action_log_prob(dist,self.squash,action)

        action = action.detach().cpu().numpy()
        action_logprobs = action_logprobs.detach().cpu().numpy() 

        action_logprobs += discrete_logprobs # Summation of logs is multiplation of probs
        action = np.concatenate((action, discrete_action[...,np.newaxis]),axis=-1)

        return action, action_logprobs

    def evaluate(self, state, action):
        device = next(self.actor.parameters()).device
        if self.use_transformer:
            state = self.transformer(state.to(device))


        raw_action = self.actor(state.to(device))

        prob = F.softmax(raw_action[...,self.N_A-1:], dim=1).detach()
        disc_dist = self.actor_discrete_dist(prob.cpu())
        discrete_logprobs = disc_dist.log_prob(action[:,-1].cpu()).cpu()

        action_mean = self.actor_continuous_head(raw_action[...,:self.N_A-1])
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean.to(torch.device('cpu')), cov_mat.to(torch.device('cpu')))

        action_logprobs = get_action_log_prob(dist,self.squash,action[...,:self.N_A-1].to(torch.device('cpu')))

        action_logprobs += discrete_logprobs # Summation of logs is multiplation of probs

        dist_entropy = dist.entropy() + disc_dist.entropy() ### NOTE :  is addition the right thing to do
        dist_entropy = dist_entropy.to(device)
        action_logprobs = action_logprobs.to(device)

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy.view(-1)

    def __str__(self):
        device = next(self.actor.parameters()).device
        return 'PPOActorCritic_Continuous:\n ' + \
            'device: {}\n actor: {}\n critic: {}\n'.format(device,
            self.actor, self.critic)


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=64, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    # def __init__(self, num_inputs, num_actions, action_range,
    def __init__(self, num_inputs, num_actions,
        hidden_size=64, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.N_S = num_inputs
        self.N_A = num_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # self.action_range = action_range

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mean, std)
        z = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(z)

        log_pi = dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi
        # return z, log_pi

    def act(self, state, deterministic=True):
        assert deterministic is True
        mean, log_std = self.forward(state)
        action = torch.tanh(mean)
        action = mean.cpu().detach().squeeze(0).numpy()
        return action, log_std

    # def rescale_action(self, action):
    #     return action * (self.action_range[1] - self.action_range[0]) / 2 +\
    #         (self.action_range[1] + self.action_range[0]) / 2


def positional_encoding(p, L=10):
    return torch.stack([torch.sin(2**i * np.pi * p) for i in range(L)] + [torch.cos(2**i * np.pi * p) for i in range(L)], dim=-1)


def get_action_log_prob(dist, squash, actions):
    """ Squashing policy output via a Tanh. Implementations based on the Stable Baselines implementation.
    """

    if not squash:
        action_logprob = dist.log_prob(actions)
    else:
        eps = torch.finfo(actions.dtype).eps
        clamped_actions = actions.clamp(min=-1.0 + eps, max=1.0 - eps)
        gaussian_action = 0.5 * (clamped_actions.log1p() - (-clamped_actions).log1p())

        action_logprob = dist.log_prob(gaussian_action)

        # Squash correction (action_logprob original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        action_logprob -= torch.sum(torch.log(1 - actions**2 + 1e-6), dim=1)

    return action_logprob       


class TransformerAgent(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.embedding_size = embedding_size

        config = BertConfig(
            vocab_size=1, # we do our own embeddings
            num_attention_heads=4,
            hidden_size=self.embedding_size,
            intermediate_size=128,
        )
        # config = BertConfig(
        #     vocab_size=1, # we do our own embeddings
        #     num_attention_heads=4,
        #     num_hidden_layers=2,
        #     hidden_size=self.embedding_size,
        #     intermediate_size=128,
        # )
        self.model = BertModel(config)
        # layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8, dim_feedforward=1024)
       # self.model = nn.TransformerEncoder(layer, num_layers=6)

        # self.predictor = nn.Sequential(
        #     nn.Linear(embedding_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 2),
        #     nn.Tanh()
        # )

        self.segment_embedding = nn.Embedding(3, embedding_size)

        self.vehicle_encoder = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_size)
        )

        self.ego_encoder = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_size)
        )

        self.waypoint_encoder = nn.Linear(1, self.embedding_size)


    def forward(self, obs_batch, return_encoding=False):
        obs_batch = obs_batch.reshape(-1, 100, 8)

        ego_indices = torch.where(obs_batch[:,:,0]==1)
        vehicle_indices = torch.where(obs_batch[:,:,0]==2)
        waypoint_indices = torch.where(obs_batch[:,:,0]==3)
        padding_indices = torch.where(obs_batch[:,:,0]==0)

        ego_features = obs_batch[ego_indices][:,1:]
        vehicle_features = obs_batch[vehicle_indices][:,1:]
        waypoint_features = obs_batch[waypoint_indices][:,1:]

        # Separate positions and encoding features
        ego_positions, ego_encodings = ego_features[:,:2], self.ego_encoder(ego_features[:,2:7])
        vehicle_positions, vehicle_encodings = vehicle_features[:,:2], self.vehicle_encoder(vehicle_features[:,2:4])
        waypoint_positions, waypoint_encodings = waypoint_features[:,:2], self.waypoint_encoder(waypoint_features[:,2:3])

        ego_position_encodings = positional_encoding(ego_positions.view(-1,2), L=4).view(-1,16).repeat(1,8).view(len(ego_positions),self.embedding_size)
        vehicle_position_encodings = positional_encoding(vehicle_positions.view(-1,2), L=4).view(-1,16).repeat(1,8).view(len(vehicle_positions),self.embedding_size)
        waypoint_position_encodings = positional_encoding(waypoint_positions.view(-1,2), L=4).view(-1,16).repeat(1,8).view(len(waypoint_positions),self.embedding_size)

        # Segment encoding (indicates token type, e.g. ego, vehicle, waypoint)
        ego_segment_encodings = self.segment_embedding(torch.tensor([0]).to(ego_features.device))
        vehicle_segment_encodings = self.segment_embedding(torch.tensor([1]).to(ego_features.device))
        waypoint_segment_encodings = self.segment_embedding(torch.tensor([2]).to(ego_features.device))

        # Construct tokens
        ego_tokens = ego_encodings + ego_position_encodings + ego_segment_encodings
        vehicle_tokens = vehicle_encodings + vehicle_position_encodings + vehicle_segment_encodings
        waypoint_tokens = waypoint_encodings + waypoint_position_encodings + waypoint_segment_encodings

        # Use token indices to construct token sequences in the original order
        all_tokens = torch.zeros((obs_batch.shape[0], obs_batch.shape[1], self.embedding_size)).to(obs_batch.device)
        masks = torch.ones((obs_batch.shape[0], obs_batch.shape[1])).to(obs_batch.device)

        all_tokens[ego_indices] = ego_tokens
        all_tokens[vehicle_indices] = vehicle_tokens
        all_tokens[waypoint_indices] = waypoint_tokens
        masks[padding_indices] = 0

        output = self.model(
            inputs_embeds=all_tokens.float(),
            attention_mask=masks
        )

        # hidden_state = output[0]
        hidden_state = output[0][:,0] # hidden state of ego token only
        return hidden_state
        # pred_action = self.predictor(hidden_state)

        # if return_encoding:
        #     return pred_action, hidden_state
        # else:
        #     return pred_action

    def predict(self, obs):
        obs = torch.FloatTensor(obs).cuda()
        action = self.forward(obs)
        return action.detach().cpu().numpy().reshape(2)


if __name__ == '__main__':
    glb_net = Basic_Discrete(12, 24).to('cpu')
    print(glb_net)
    policy = PPOActorCritic_Continuous(12, 24)
    print(policy)
    q_net = SoftQNetwork(12, 24)
    # p_net = PolicyNetwork(12, 24, (-1, 1))
    p_net = PolicyNetwork(12, 24)
    print(q_net, '\n', p_net)
