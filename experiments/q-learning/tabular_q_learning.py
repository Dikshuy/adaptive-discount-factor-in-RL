from gym_simplegrid.envs import SimpleGridEnv

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from maps import load_map
import gin
import wandb
import os
from datetime import datetime
import matplotlib.colors as mcolors

from tqdm import tqdm


@gin.configurable
def fibonacci_seeds(TOTAL_SEEDS):
    seeds = [0, 1]
    for i in range(2, TOTAL_SEEDS):
        next_seed = seeds[i - 1] + seeds[i - 2]
        seeds.append(next_seed)
    seeds = list(set(seeds))
    return seeds[:TOTAL_SEEDS]


@gin.configurable
def smoothen(returns, WINDOW_SIZE):

    smoothed_returns = []
    for i in range(len(returns)):
        window = returns[max(0, i - WINDOW_SIZE + 1):i + 1]
        smoothed_returns.append(sum(window) / len(window))
    
    return smoothed_returns

def plot_with_confidence(item, ax, color, label, marker):
    
    mean_item = np.mean(item, axis=0)
    
    lower_bound = np.percentile(item, 5, axis=0)
    upper_bound = np.percentile(item, 95, axis=0)
    
    episodes = np.arange(item.shape[1])

    ax.plot(episodes, mean_item, color = color, label = label, zorder = 2) #, marker = marker)
    ax.fill_between(episodes, lower_bound, upper_bound, color = color, alpha = 0.1, zorder = 1)
    
    return ax
    
@gin.configurable
def define_env(NAME, GAMMA):
    
    MAP = load_map(NAME)
    
    length = len(MAP)
    width = len(MAP[0])

    options ={
        'start_loc': (length - 1, 0),
        'goal_loc': (0, width - 1)
    }

    env = gym.make(
        'SimpleGrid-v0', 
        obstacle_map = MAP,
    )
    
    return env, options, GAMMA


@gin.configurable
def eval(env, options, Q, EVAL_EPI):
    
    eval_returns = []
    eval_lens = []
    
    seeds = fibonacci_seeds(EVAL_EPI)
    
    for i in range(EVAL_EPI):
        
        obs, _ = env.reset(options = options, seed = seeds[i])
        done = False
        e_ret = 0
        e_len = 0
        
        while not done:
            
            act = np.argmax(Q[obs])
            next_obs, rew, term, trun, _ = env.step(act)
            
            done = term or trun
            
            e_ret += rew
            e_len += 1
            
            obs = next_obs
        
        eval_returns.append(e_ret)
        eval_lens.append(e_len)
        
    return np.mean(eval_returns, -1), np.mean(eval_lens, -1)
        

@gin.configurable
def qlearning(env, eval_env, options, seed, gamma, LR, Q_INIT, TOTAL_STEPS, EPS_INIT, EPS_MIN, EPS_DECAY_UNTILL, EVAL_EVERY):
    
    if Q_INIT is not None:
        Q = np.ones((env.observation_space.n, env.action_space.n)) * Q_INIT
    else:
        Q = np.random.randn((env.observation_space.n, env.action_space.n))
        
    done = True
    
    train_rets = []
    train_lens = []

    eval_rets = []
    eval_lens = []

    t_len = 0
    t_ret = 0
    
    DECAY_TILL = int(TOTAL_STEPS * EPS_DECAY_UNTILL)
    
    for step in range(TOTAL_STEPS):
        
        if step > 0 and step % EVAL_EVERY == 0:
            e_ret, e_len = eval(eval_env, options, Q)
            eval_rets.append(e_ret)
            eval_lens.append(e_len)
        
        if done:
            obs, _ = env.reset(seed = seed, options = options)
            
            train_lens.append(t_len)
            train_rets.append(t_ret)
            
            t_len = 0
            t_ret = 0
            
            eps = EPS_INIT * (DECAY_TILL - step) / DECAY_TILL
            eps = max(eps, EPS_MIN)
        
        # Epsilon-greedy action selection
        if np.random.rand() < eps:
            act = env.action_space.sample()
        else:
            act = np.argmax(Q[obs])
        
        next_obs, rew, term, trun, _ = env.step(act)
        
        t_len += 1
        t_ret += rew
        
        Q[obs, act] += LR * (rew + (gamma * np.max(Q[next_obs]) * (1 - term)) - Q[obs, act])
        
        obs = next_obs
        
        done =  term or trun 
        
    return (np.array(eval_rets), np.array(eval_lens)), (np.array(train_rets), np.array(train_lens))


@gin.configurable
def main(GAMMAS, Q_INIT, NAME):

    current_time = datetime.now()
    formatted_time = current_time.strftime('%b%d_%H_%M_%S')

    main_dir = f"MG_Results/" + formatted_time.upper() + '/'
    os.makedirs(main_dir, exist_ok = True)
    
    set_of_seeds = fibonacci_seeds()
    env, options, env_gamma = define_env()
    eval_env, _, _ = define_env()
    
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 8))
    ax1, ax2 = axes.flatten()
    
    colormap = plt.cm.viridis  # You can choose any colormap
    colors = [colormap(i) for i in np.linspace(0, 1, len(GAMMAS))]
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    
    for idx, gamma in enumerate(GAMMAS):
        
        set_of_eval_rets, set_of_eval_lens = [], []
        set_of_train_rets, set_of_train_lens = [], []
        
        for seed in set_of_seeds:
            
            eval_data, train_data = qlearning(env, eval_env, options, seed, gamma)
            
            set_of_eval_rets.append(eval_data[0])
            set_of_eval_lens.append(eval_data[1])
            
            set_of_train_rets.append(train_data[0])
            set_of_train_lens.append(train_data[1])   
            
    
        set_of_eval_rets = np.array(set_of_eval_rets)
        set_of_eval_lens = np.array(set_of_eval_lens)
        # set_of_train_rets = np.array(set_of_train_rets)
        # set_of_train_lens = np.array(set_of_train_lens)
                
        ax1 = plot_with_confidence(set_of_eval_rets, ax1, colors[idx], str(gamma), markers[idx])
        ax2 = plot_with_confidence(set_of_eval_lens, ax2, colors[idx], str(gamma), markers[idx])
        
        
    ax1.set_title('Eval Episodic Return')
    ax2.set_title('Eval Episodic Lengths')
    
    ax1.legend()
    ax2.legend()
    
    fig.text(0.5, 0.01, f"Q_INIT : {Q_INIT} - MAP : {NAME}", ha = 'center', fontsize = 10)
    
    fig.savefig(main_dir + 'Comparision.png')
    
gin.parse_config_file('config.gin')
main()