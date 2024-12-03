import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
from datetime import datetime
import matplotlib.pyplot as plt
import os
import gin
import argparse
import pickle
from tqdm import tqdm

def fibonacci_seeds(total_seeds):
    seeds = [0, 1]
    for i in range(2, total_seeds + 2):
        next_seed = seeds[i - 1] + seeds[i - 2]
        seeds.append(next_seed)
    seeds = list(set(seeds))[:total_seeds]
    
    assert len(seeds) == len(np.unique(seeds))
    assert len(seeds) == total_seeds
    
    return seeds

def plot_with_confidence(item, ax, color, label, marker):
    mean_item = np.mean(item, axis=0)
    
    lower_bound = np.percentile(item, 5, axis=0)
    upper_bound = np.percentile(item, 95, axis=0)
    
    episodes = np.arange(item.shape[1])

    ax.plot(episodes, mean_item, color=color, label=label, zorder=2)
    ax.fill_between(episodes, lower_bound, upper_bound, color=color, alpha=0.1, zorder=1)
    
    return ax

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        
        # Actor network layers
        self.actor_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Critic network layers
        self.critic_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor output layers
        self.policy_head = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic output layer
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        
        # Process through actor and critic networks separately
        actor_features = self.actor_network(x)
        critic_features = self.critic_network(x)
        
        # Get value prediction
        value = self.value_head(critic_features)
        
        # Get policy outputs
        policy = self.policy_head(actor_features)
        return policy, value

@gin.configurable
def eval(env, actor_critic, EVAL_EPI):
    eval_returns = []
    eval_lens = []

    seeds = fibonacci_seeds(EVAL_EPI)

    for i in range(EVAL_EPI):
        obs, _ = env.reset(seed=seeds[i])
        done = False
        e_ret = 0
        e_len = 0
        
        while not done:
            act_probs, _ = actor_critic(obs)
            act = np.argmax(act_probs.detach().numpy())
            next_obs, rew, term, trun, _ = env.step(act)
            
            done = term or trun
            
            e_ret += rew
            e_len += 1
            
            obs = next_obs
        
        eval_returns.append(e_ret)
        eval_lens.append(e_len)
        
    return np.mean(eval_returns, -1), np.mean(eval_lens, -1)

def update(reward, done, obs, next_obs, log_prob, actor_critic, optimizer, gamma):
    curr_policy, curr_value = actor_critic(obs)
    _, next_value = actor_critic(next_obs)
    
    target = reward + (1 - int(done)) * gamma * next_value.detach()
    td_error = target - curr_value
    
    policy_loss = -log_prob * td_error.detach()
    value_loss = td_error ** 2
    total_loss = policy_loss + value_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

@gin.configurable
def train(env, eval_env, actor_critic, gamma, optimizer, MAX_STEPS, EVAL_EVERY, EXP_INIT, EXP_DECAY_UNTILL, EXP_MIN):
    
    MAX_STEPS = int(MAX_STEPS)
    
    eval_rets, eval_lens = np.zeros(MAX_STEPS), np.zeros(MAX_STEPS)
    train_rets, train_lens = np.zeros(MAX_STEPS), np.zeros(MAX_STEPS)
    
    done, truncated = True, True
    t_len, t_ret = 0, 0
    e_len, e_ret = 0, 0
    
    t_update_idx, e_update_idx = 0, 0 
    
    DECAY_TILL = int(MAX_STEPS * EXP_DECAY_UNTILL)
        
    for step in range(MAX_STEPS):
        
        exp_factor = EXP_INIT * (DECAY_TILL - step) / DECAY_TILL
        exp_factor = max(exp_factor, EXP_MIN)
                    
        if (done or truncated):
            obs, _ = env.reset()
            done = False
            truncated = False
            
            train_rets[t_update_idx:] = t_ret
            train_lens[t_update_idx:] = t_len
            
            t_update_idx = step            
            t_len, t_ret = 0, 0
            
        if step % EVAL_EVERY == 0 and step > 0:
            e_ret, e_len = eval(eval_env, actor_critic)            
            eval_rets[e_update_idx:] = e_ret
            eval_lens[e_update_idx:] = e_len
            
            e_update_idx = step
        
        action_probs, _ = actor_critic(obs)
        dist = Categorical(action_probs)

        if np.random.rand() < exp_factor:
            action = torch.tensor(np.random.choice(env.action_space.n))
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
            
        next_obs, reward, done, truncated, _ = env.step(action.numpy())
        
        update(reward, done, obs, next_obs, log_prob, actor_critic, optimizer, gamma)
        
        obs = next_obs
        
        t_len += 1
        t_ret += reward
    
    # print(f'FINAL PERFORMANCE: TRAIN = {t_ret}; EVAL = {e_ret}')
    return (np.array(eval_rets), np.array(eval_lens)), (np.array(train_rets), np.array(train_lens))

@gin.configurable
def init_stuff(ENV_ID, LR):
    env = gym.make(ENV_ID)
    eval_env = gym.make(ENV_ID)
    input_dim = env.observation_space.shape[0] 
    output_dim = env.action_space.n
    
    actor_critic = ActorCriticNetwork(input_dim, output_dim)
    optimizer = optim.Adam(actor_critic.parameters(), lr=LR)
    
    return env, eval_env, actor_critic, optimizer

@gin.configurable
def main(addn_name, GAMMAS, NAME, NUM_SEEDS):
    current_time = datetime.now()
    formatted_time = current_time.strftime('%b%d_%H_%M_%S')

    main_dir = f"AC_Results/" + formatted_time.upper() + f'_{addn_name}/'
    os.makedirs(main_dir, exist_ok=True)
    
    set_of_seeds = fibonacci_seeds(NUM_SEEDS)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    colormap = plt.cm.viridis
    colors = [colormap(i) for i in np.linspace(0, 1, len(GAMMAS))]
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    
    # Dictionary to store all results
    results = {
        'env_id' : None,
        'hyperparams' : {
            'gammas' : GAMMAS,
            'env_name' : NAME,
            'num_seeds' : NUM_SEEDS,
            'lr' : None  # Will be set after first optimizer initialization
        },
        'scores': {}
    }

    for idx, gamma in enumerate(GAMMAS):
        print(f"RUNNING: Gamma = {gamma}")
        
        set_of_eval_rets, set_of_eval_lens = [], []
        set_of_train_rets, set_of_train_lens = [], []
        
        for seed in tqdm(set_of_seeds):
            torch.manual_seed(seed)
            
            env, eval_env, actor_critic, optimizer = init_stuff()
            
            if results['hyperparams']['lr'] is None:
                results['hyperparams']['lr'] = optimizer.param_groups[0]['lr']
                results['env_id'] = env.spec.id
            
            eval_data, train_data = train(env, eval_env, actor_critic, gamma, optimizer)
            
            set_of_eval_rets.append(eval_data[0])
            set_of_eval_lens.append(eval_data[1])
            
            set_of_train_rets.append(train_data[0])
            set_of_train_lens.append(train_data[1])   
        
        set_of_eval_rets = np.array(set_of_eval_rets)
        set_of_eval_lens = np.array(set_of_eval_lens)
        
        set_of_train_rets = np.array(set_of_train_rets)
        set_of_train_lens = np.array(set_of_train_lens)
        
        # Store scores for this gamma
        results['scores'][gamma] = {
            'eval_returns': np.mean(set_of_eval_rets, 0),
            'eval_lengths': np.mean(set_of_eval_lens, 0),
            'train_returns': np.mean(set_of_train_rets, 0),
            'train_lengths': np.mean(set_of_train_lens, 0)
        }
                
        ax1 = plot_with_confidence(set_of_eval_rets, ax1, colors[idx], str(gamma), markers[idx])
        ax2 = plot_with_confidence(set_of_eval_lens, ax2, colors[idx], str(gamma), markers[idx])

        ax3 = plot_with_confidence(set_of_train_rets, ax3, colors[idx], str(gamma), markers[idx])
        ax4 = plot_with_confidence(set_of_train_lens, ax4, colors[idx], str(gamma), markers[idx])

        
        print('-' * 10)
        print()
        
    ax1.set_title('Eval Episodic Return')
    ax2.set_title('Eval Episodic Lengths')
    ax3.set_title('Train Episodic Return')
    ax4.set_title('Train Episodic Lengths')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    
    fig.text(0.5, 0.01, f"ENV : {NAME} - LR : {optimizer.param_groups[0]['lr']}", ha='center', fontsize=10)

    fig.savefig(main_dir + 'Results.png')
    
    # Save results and figure
    with open(os.path.join(main_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

# Add argument parsing
parser = argparse.ArgumentParser(description="Actor-Critic Training Script")
parser.add_argument('--config', type=str, default='AC_config.gin', help="Path to the gin configuration file.")
parser.add_argument('--gin_bindings', type=str, default='', help="Gin bindings to override during runtime.")
args = parser.parse_args()

# Parse gin configurations
gin.parse_config_files_and_bindings(
    config_files=[args.config],  # Use the specified config file
    bindings=[args.gin_bindings]  # Allow runtime overrides
)

main(args.config[:-4])