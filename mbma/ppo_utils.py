import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import dmc2gym

class RunningMeanStd2:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class ObsNormalizer:
    def __init__(self, envs_shape, device):
        self.device = device
        #self.rms = gym.wrappers.normalize.RunningMeanStd(shape=envs_shape)
        self.rms = RunningMeanStd2(shape=envs_shape)
        self.shape = envs_shape
        
    def add(self, obs):
        self.rms.update(np.array(obs).reshape(1, self.shape[0]))
        
    def forward(self, obs):
        
        mean = torch.from_numpy(self.rms.mean).to(self.device).unsqueeze(0)
        std = torch.sqrt(torch.from_numpy(self.rms.var).to(self.device) + 1e-8).unsqueeze(0)
        return ((obs - mean.float())/(std.float() + 1e-8)).clip(-10,10)
        
        #return obs
        
    def get_meanstd(self):
        pass
    
class RewardNormalizer:
    def __init__(self, device):
        self.device = device
        #self.rms = gym.wrappers.normalize.RunningMeanStd(shape=())
        self.rms = RunningMeanStd2(shape=())
        
    def add(self, reward):
        returns = np.zeros(1)
        returns += reward
        self.rms.update(returns)
        
    def forward(self, reward):
        '''
        mean = torch.from_numpy(np.asarray(self.rms.mean)).to(self.device).unsqueeze(0)
        std = torch.sqrt(torch.from_numpy(np.asarray(self.rms.var)).to(self.device) + 1e-8).unsqueeze(0)
        return ((reward - mean.float())/(std.float() + 1e-8)).clip(-10,10)
        '''
        return reward
        
    def get_meanstd(self):
        pass

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), args.hidden_dim)), nn.LeakyReLU(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)), nn.LeakyReLU(),
            layer_init(nn.Linear(args.hidden_dim, 1), std=1.0),)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), args.hidden_dim)), nn.LeakyReLU(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)), nn.LeakyReLU(),
            layer_init(nn.Linear(args.hidden_dim, np.prod(envs.action_space.shape)), std=0.01),)
        
        self.record = []
        self.args = args 
        self.simple_logstd = args.simple_logstd
        self.rpo_alpha = args.rpo_alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        
        if args.simple_logstd:
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))
        else:
            self.actor_logstd = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod(), args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, np.array(envs.action_space.shape).prod()))
        
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        action_mean = torch.tanh(self.actor_mean(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class transition_net(nn.Module):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod() + np.array(envs.action_space.shape).prod(), hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, np.array(envs.observation_space.shape).prod()))
        self.apply(weight_init)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return self.transition(state_action)
    
class transition_nets(nn.Module):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        self.t1 = transition_net(envs, hidden_dim)
        self.t2 = transition_net(envs, hidden_dim)
        
    def forward(self, state, action):
        ns1 = self.t1(state, action)
        ns2 = self.t2(state, action)
        return (ns1 + ns2) / 2
    
class reward_net(nn.Module):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        self.reward = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod() + np.array(envs.action_space.shape).prod(), hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return self.reward(state_action)
    
class reward_nets(nn.Module):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        self.r1 = reward_net(envs, hidden_dim)
        self.r2 = reward_net(envs, hidden_dim)
        
    def forward(self, state, action):
        r1 = self.r1(state, action)
        r2 = self.r2(state, action)
        return (r1 + r2) / 2
    
def evaluate(args, agent, num_episodes, device):
    args.seed += 1
    envs_copy = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
    rews = 0
    for i in range(num_episodes):
        obs = torch.tensor(envs_copy.reset()).float().to(device)
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
            obs, reward, done, _ = envs_copy.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
            obs = torch.tensor(obs).float().to(device)
            rews += reward
    agent.record.append(rews/num_episodes)
    return rews/num_episodes

def evaluate2(args, agent, num_episodes, device, s_normalizer):
    args.seed += 1
    envs_copy = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
    rews = 0
    for i in range(num_episodes):
        obs = torch.tensor(envs_copy.reset()).float().to(device)
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(s_normalizer.forward(obs.unsqueeze(0)))
            obs, reward, done, _ = envs_copy.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
            obs = torch.tensor(obs).float().to(device)
            rews += reward
    agent.record.append(rews/num_episodes)
    return rews/num_episodes

def generate_trajectory2(states, horizon, t_net, r_net, agent, device):
    with torch.no_grad():
        states_ = torch.zeros(states.size(0), states.size(1), horizon).float().to(device)
        final_state_values_ = torch.zeros(states.size(0), 1).float().to(device)
        rewards_ = torch.zeros(states.size(0), horizon).float().to(device)
        values_ = torch.zeros(states.size(0), horizon).float().to(device)
        log_probs_ = torch.zeros(states.size(0), horizon).float().to(device)
        actions, logprobs, _, value = agent.get_action_and_value(states)
        actions_ = torch.zeros(states.size(0), actions.size(1), horizon).float().to(device)
        for i in range(horizon):
            states_[:,:,i] = states
            actions_[:,:,i] = actions
            log_probs_[:,i] = logprobs
            values_[:,i] = value.squeeze()
            new_states = t_net(states, actions.clip(min=-1.0, max=1.0))
            rewards_[:,i] = r_net(states, actions.clip(min=-1.0, max=1.0)).squeeze()
            actions, logprobs, _, value = agent.get_action_and_value(new_states)
            states = new_states
        final_state_values_[:] = value
    return states_, actions_, rewards_, final_state_values_.squeeze(), log_probs_, values_

def generate_trajectory(states, horizon, t_net, r_net, agent, device, s_normalizer):
    with torch.no_grad():
        states_ = torch.zeros(states.size(0), states.size(1), horizon).float().to(device)
        final_state_values_ = torch.zeros(states.size(0), 1).float().to(device)
        rewards_ = torch.zeros(states.size(0), horizon).float().to(device)
        values_ = torch.zeros(states.size(0), horizon).float().to(device)
        log_probs_ = torch.zeros(states.size(0), horizon).float().to(device)
        actions, logprobs, _, value = agent.get_action_and_value(s_normalizer.forward(states))
        actions_ = torch.zeros(states.size(0), actions.size(1), horizon).float().to(device)
        for i in range(horizon):
            states_[:,:,i] = states
            actions_[:,:,i] = actions
            log_probs_[:,i] = logprobs
            values_[:,i] = value.squeeze()
            new_states = t_net(states, actions.clip(min=-1.0, max=1.0))
            rewards_[:,i] = r_net(states, actions.clip(min=-1.0, max=1.0)).squeeze()
            actions, logprobs, _, value = agent.get_action_and_value(s_normalizer.forward(new_states))
            states = new_states
        final_state_values_[:] = value
    return states_, actions_, rewards_, final_state_values_.squeeze(), log_probs_, values_

def view_chunk(tensor, chunks, dim=0):
    assert tensor.shape[dim] % chunks == 0
    if dim < 0:  # Support negative indexing
        dim = len(tensor.shape) + dim
    cur_shape = tensor.shape
    new_shape = cur_shape[:dim] + (chunks, tensor.shape[dim] // chunks) + cur_shape[dim + 1:]
    return tensor.reshape(*new_shape).transpose(0,1)

def get_returns(args, rewards, values, dones, next_done, next_value, device):
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return returns, advantages

def calculate_advantage(args, final_state_values_, rewards_, values_, device, normalizer=None):
    if args.normalize_rewards:
        rewards_ = normalizer.forward(rewards_)
    next_value = final_state_values_
    if args.gae:
        advantages = torch.zeros_like(rewards_).to(device)
        lastgaelam = 0
        for idx in reversed(range(rewards_.size(1))):
            if idx == rewards_.size(1) - 1:
                nextvalues = next_value
            else:
                nextvalues = values_[:, idx+1]
            delta = rewards_[:, idx] + args.gamma * nextvalues - values_[:, idx]
            advantages[:, idx] = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam
        returns = advantages + values_
    if args.gae is False:
        returns = torch.zeros_like(rewards_).to(device)
        for idx in reversed(range(rewards_.size(1))):
            if idx == rewards_.size(1) - 1:
                next_return = next_value
            else:
                next_return = returns[:, idx+1]
            returns[:, idx] = rewards_[:, idx] + args.gamma * next_return
        advantages = returns - values_
    return advantages.detach(), returns.detach()

class ExperienceBuffer(object):
    
    def __init__(self, capacity, env, device):
        super().__init__()
        self.states = torch.zeros(capacity, np.array(env.observation_space.shape).prod())
        self.actions = torch.zeros(capacity, np.array(env.action_space.shape).prod())
        self.rewards = torch.zeros(capacity, 1)
        self.next_states = torch.zeros(capacity, np.array(env.observation_space.shape).prod())
        self.capacity = capacity
        self.full = False
        self.idx = 0
        self.device = device
        
    def add(self, state, action, reward, next_state):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.idx += 1
        if self.idx >= self.capacity:
            self.full = True
            self.idx = 0
            
    def sample(self, batch_size):
        idx = np.random.permutation(self.capacity)[:batch_size] if self.full else np.random.permutation(self.idx-1)[:batch_size]
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        new_states = self.next_states[idx]
        return states.to(self.device), actions.to(self.device), rewards.to(self.device), new_states.to(self.device)

def get_q_value(qpos, qvel, state, args, agent, device, normalizer):
    env_copy = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
    s = []
    r = []
    v = []
    _ = torch.tensor(env_copy.reset()).float().to(device)
    qpqv = np.concatenate((qpos,qvel))
    with env_copy.env.physics.reset_context():
      env_copy.env.physics.set_state(qpqv)
    s.append(state)
    with torch.no_grad():
        action, logprob, _, val = agent.get_action_and_value(state.unsqueeze(0))
    obs, reward, done, _ = env_copy.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
    obs = torch.tensor(obs).float().to(device)
    r.append(torch.tensor(reward).float().to(device))
    v.append(val)
    for i in range(args.horizon-1):
        s.append(obs)
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
        obs, reward, done, _ = env_copy.step(a.cpu().clip(min=-1, max=1).squeeze(0).numpy())
        obs = torch.tensor(obs).float().to(device)
        r.append(torch.tensor(reward).float().to(device))
        v.append(val)
        if done:
            break
    final_state = obs
    s = torch.stack(s)
    r = torch.stack(r)
    v = torch.stack(v).squeeze()
    if args.normalize_rewards:
        r = normalizer.forward(r)
    with torch.no_grad():
        next_value = agent.get_value(final_state.unsqueeze(0)).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(r).to(device)
            lastgaelam = 0
            for t in reversed(range(args.horizon)):
                if t == args.horizon - 1:
                    nextvalues = next_value
                else:
                    nextvalues = v[t + 1]
                delta = r[t] + args.gamma * nextvalues - v[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam
            returns = advantages + v
        else:
            returns = torch.zeros_like(r).to(device)
            for t in reversed(range(args.horizon)):
                if t == args.horizon - 1:
                    next_return = next_value
                else:
                    next_return = returns[t + 1]
                returns[t] = r[t] + args.gamma * next_return
            advantages = returns - v
    return action, logprob, advantages[0]

def get_q_value2(qpos, qvel, state, args, agent, device):
    env_copy = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
    s = []
    r = []
    v = []
    _ = torch.tensor(env_copy.reset()).float().to(device)
    qpqv = np.concatenate((qpos,qvel))
    with env_copy.env.physics.reset_context():
      env_copy.env.physics.set_state(qpqv)
    s.append(state)
    with torch.no_grad():
        action, logprob, _, val = agent.get_action_and_value(state.unsqueeze(0))
    obs, reward, done, _ = env_copy.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
    obs = torch.tensor(obs).float().to(device)
    r.append(torch.tensor(reward).float().to(device))
    v.append(val)
    for i in range(50):
        s.append(obs)
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
        obs, reward, done, _ = env_copy.step(a.cpu().clip(min=-1, max=1).squeeze(0).numpy())
        obs = torch.tensor(obs).float().to(device)
        r.append(torch.tensor(reward).float().to(device))
        v.append(val)
        if done:
            break
    final_state = obs
    s = torch.stack(s)
    r = torch.stack(r)
    v = torch.stack(v).squeeze()
    with torch.no_grad():
        if done:
            next_value = 0
        else:
            next_value = agent.get_value(final_state.unsqueeze(0)).reshape(1, -1)
        returns = torch.zeros_like(r).to(device)
        for t in reversed(range(51)):
            if t == 50:
                next_return = next_value
            else:
                next_return = returns[t + 1]
            returns[t] = r[t] + args.gamma * next_return
    return action, logprob, returns[0]

class StateNormalizer:
    def __init__(self, device, shape):
        self.mean = torch.zeros(1, shape).float().to(device)
        self.var = torch.ones(1, shape).float().to(device)
        self.count = 1

    def add(self, x):
        batch_mean = x.mean(0).unsqueeze(0)
        batch_var = x.std(0).pow(2).unsqueeze(0)
        batch_count = x.size(0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def update_mean_var_count_from_moments(self,
        mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
    
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count
    
    def forward(self, x):
        out = (x - self.mean)/(torch.sqrt(self.var) + 1e-8)
        return out.clip(min=-10,max=10)
