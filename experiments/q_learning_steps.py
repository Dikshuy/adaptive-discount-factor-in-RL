from gym_simplegrid.envs import SimpleGridEnv

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from maps import load_map

def fibonacci_seeds(TOTAL_SEEDS):
    seeds = [0, 1]
    for i in range(2, TOTAL_SEEDS):
        next_seed = seeds[i - 1] + seeds[i - 2]
        seeds.append(next_seed)
    seeds = list(set(seeds))
    return seeds[:TOTAL_SEEDS]


def plot_with_confidence(item, ax, color, label, marker):
    
    mean_item = np.mean(item, axis=0)
    
    lower_bound = np.percentile(item, 5, axis=0)
    upper_bound = np.percentile(item, 95, axis=0)
    
    episodes = np.arange(item.shape[1])

    ax.plot(episodes, mean_item, color = color, label = label, zorder = 2) #, marker = marker)
    ax.fill_between(episodes, lower_bound, upper_bound, color = color, alpha = 0.1, zorder = 1)
    
    return ax


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

# def eps_greedy_action(Q, s, eps):
#     if np.random.rand() < eps:
#         action = env.action_space.sample()
#     else:
#         # action = np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])
#         action = np.argmax(Q[s])
#     return action


# def expected_return(env, Q, options, gamma, episodes=1):
#     G = np.zeros(episodes)
#     episode_steps = np.zeros(episodes)
#     for e in range(episodes):
#         np.random.seed(e)
#         s, _ = env.reset(seed = int(seed), options = options)
#         done = False
#         t = 0
#         while not done:
#             # a = eps_greedy_action(Q, s, 0.0)
#             a = np.argmax(Q[s])
#             s_next, r, terminated, truncated, _ = env.step(a)
#             done = terminated or truncated
#             G[e] += r
#             s = s_next
#             t += 1
#             if done:
#                 episode_steps[e] = t
#     return G.mean(), episode_steps.mean()

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


# def Q_learning(env, options, Q, gamma, eps, alpha, max_steps, _seed):
#     exp_ret = []
#     steps_per_episode = []
#     eps_decay = eps / (10*max_steps)
#     alpha_decay = alpha / max_steps
#     tot_steps = 0

#     while tot_steps < max_steps:  
#         s, _ = env.reset(seed = _seed, options = options)
#         # a = eps_greedy_action(Q, s, eps)
#         done = False
#         # print("#####")
#         # print("eps:", eps)
#         while not done and tot_steps < max_steps:
#             tot_steps += 1
#             a = eps_greedy_action(Q, s, eps)
#             s_next, r, terminated, truncated, _ = env.step(a)

#             done = terminated or truncated
#             # eps = max(eps - eps_decay, 0.01)
#             # alpha = max(alpha - alpha_decay, 0.001)

#             td_err = r + gamma * np.max(Q[s_next]) * (1 - terminated) - Q[s, a]

#             Q[s,a] += alpha * td_err

#             if tot_steps % eval_steps == 0:
#                 # G, episode_steps = expected_return(env, Q, options, gamma)
#                 G, episode_steps = eval(env, options, Q, 1)
#                 # print("G:", G)
#                 # print("episode_steps", episode_steps)
#                 # print("total steps:", tot_steps)
#                 exp_ret.append(G)
#                 steps_per_episode.append(episode_steps)

#             if done:
#                 eps = 0.99 * (int(max_steps * 0.9)-tot_steps) / int(max_steps * 0.9)
#                 eps = max(eps, 0.0)

#             s = s_next

#     return Q, exp_ret, steps_per_episode


def qlearning(env, eval_env, options, seed, gamma, alpha, Q_INIT, TOTAL_STEPS):
    
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
    
    DECAY_TILL = int(TOTAL_STEPS * 0.9)
    
    for step in range(TOTAL_STEPS):
        
        if step > 0 and step % eval_steps == 0:
            e_ret, e_len = eval(eval_env, options, Q, EVAL_EPI)
            eval_rets.append(e_ret)
            eval_lens.append(e_len)
        
        if done:
            obs, _ = env.reset(seed = seed, options = options)
            
            train_lens.append(t_len)
            train_rets.append(t_ret)
            
            t_len = 0
            t_ret = 0
            
            eps = 0.99 * (DECAY_TILL - step) / DECAY_TILL
            eps = max(eps, 0.0)
        
        # Epsilon-greedy action selection
        if np.random.rand() < eps:
            act = env.action_space.sample()
        else:
            act = np.argmax(Q[obs])
        
        next_obs, rew, term, trun, _ = env.step(act)
        
        t_len += 1
        t_ret += rew
        
        Q[obs, act] += alpha * (rew + (gamma * np.max(Q[next_obs]) * (1 - term)) - Q[obs, act])
        
        obs = next_obs
        
        done =  term or trun 
        
    return (np.array(eval_rets), np.array(eval_lens)), (np.array(train_rets), np.array(train_lens))


# def smooth(arr, span):
#     re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
#     re[0] = arr[0]
#     for i in range(1, span + 1):
#         re[i] = np.average(arr[: i + span])
#         re[-i] = np.average(arr[-i - span:])
#     return re


# def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
#     y = np.nanmean(data, 0)
#     x = np.arange(len(y))
#     x = [stepsize * step for step in range(len(y))]
#     if smoothing_window > 1:
#         y = smooth(y, smoothing_window)

#     (line,) = ax.plot(x, y, **kwargs)
#     error = np.nanstd(data, axis=0)
#     if smoothing_window > 1:
#         error = smooth(error, smoothing_window)
#     error = 1.96 * error / np.sqrt(data.shape[0])
#     ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())


alpha = 0.5
# eps = 1.0
max_steps = 75000
eval_steps = 250

TOTAL_SEEDS = 10

# init_values = [0.0]#, 5.0, 10.0]
# gamma_values = [0.1, .25, 0.5, 0.75, 0.8, 0.9, 0.99]
gamma_values = [0.1, 0.5, 0.75, 0.99]
# seeds = np.arange(10)

# results_exp_ret = np.zeros((
#     len(gamma_values),
#     len(init_values),
#     len(seeds),
#     max_steps // eval_steps,
# ))

# results_steps= np.zeros((
#     len(gamma_values),
#     len(init_values),
#     len(seeds),
#     max_steps // eval_steps,
# ))

# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# plt.ion()
# plt.show()

# for ax in axs:
#     ax.set_prop_cycle(color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"])
#     ax.set_xlabel("Steps", fontsize=10)
#     ax.grid(True, which="both", linestyle="--", linewidth=0.5)
#     ax.minorticks_on()

# env = gym.make(
#     'SimpleGrid-v0', 
#     obstacle_map=obstacle_map,
# )

Q_INIT = 0

env, options, env_gamma = define_env("EASY_SPARSE", 0.99)
eval_env, _, _ = define_env("EASY_SPARSE", 0.99)
# for i, gamma in enumerate(gamma_values):
#     for j, init_value in enumerate(init_values):
#         for seed in seeds:
#             np.random.seed(seed)
#             Q = np.zeros((n_states, n_actions)) + init_value
#             Q, exp_ret, steps = Q_learning(env, options, Q, gamma, eps, alpha, max_steps, int(seed))
#             eval_data, train_data = qlearning(env, env, options, seed, gamma)
#             results_exp_ret[i, j, seed] = exp_ret
#             results_steps[i, j, seed] = steps

#             print(gamma, init_value, seed)

#         error_shade_plot(
#             axs[0],
#             results_exp_ret[i, j],
#             stepsize=1,
#             smoothing_window=10,
#             label=f'γ={gamma:.2f}, $Q_o$={init_value:.2f}'
#         )
#         axs[0].set_ylabel("Average Return", fontsize=10)
#         axs[0].set_title("Q-Learning Performance Across Different Gamma Values")
#         axs[0].legend()
#         # axs[0].set_ylim([-5,1.4])

#         error_shade_plot(
#             axs[1],
#             results_steps[i, j],
#             stepsize=1,
#             smoothing_window=10,
#             label=f'γ={gamma:.2f}, $Q_o$={init_value:.2f}'
#         )
#         axs[1].set_ylabel("Steps to Goal", fontsize=10)
#         axs[1].set_title("Steps per Episode Across Different Gamma Values")
#         axs[1].legend()
#         # axs[1].set_ylim([0, 200]) 

#         plt.tight_layout()
#         plt.draw()
#         plt.pause(0.001)

# plt.savefig("steps_q.png", dpi=300)
# plt.ioff()
# plt.show() for idx, gamma in enumerate(GAMMAS):

set_of_seeds = fibonacci_seeds(TOTAL_SEEDS)

EVAL_EPI = 1
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 8))
ax1, ax2 = axes.flatten()

colormap = plt.cm.viridis  # You can choose any colormap
colors = [colormap(i) for i in np.linspace(0, 1, len(gamma_values))]
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    

for idx, gamma in enumerate(gamma_values):       
    set_of_eval_rets, set_of_eval_lens = [], []
    set_of_train_rets, set_of_train_lens = [], []
    
    for seed in set_of_seeds:
        
        eval_data, train_data = qlearning(env, env, options, seed, gamma, alpha, Q_INIT, max_steps)
        
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

# fig.text(0.5, 0.01, f"Q_INIT : {Q_INIT} - MAP : {NAME}", ha = 'center', fontsize = 10)

fig.savefig('Comparison.png')

