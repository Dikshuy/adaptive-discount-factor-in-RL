import gymnasium as gym
from gym_simplegrid.envs import SimpleGridEnv
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

obstacle_map = [
    "00000",
    "00000",
    "00000",
    "00000",
    "00000",
]

length = len(obstacle_map)
width = len(obstacle_map[0])

options ={
    'start_loc': (length - 1, 0),
    'goal_loc': (0, width - 1)
}

env = gym.make(
    'SimpleGrid-v0', 
    obstacle_map=obstacle_map,
)

obs, _ = env.reset(seed=1, options = options)
rew = env.unwrapped.reward
done = env.unwrapped.done

n_states = (length + 1) * (width + 1)
n_actions = 4

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))


def eps_greedy_probs(Q, eps):
    pi = np.ones((n_states, n_actions)) * (eps / n_actions)
    best_actions = np.argmax(Q, axis=1)
    for s in range(n_states):
        pi[s, best_actions[s]] += (1 - eps)
    return pi


def eps_greedy_action(Q, s, eps):
    if np.random.rand() < eps:
        action = np.random.choice(n_actions)
    else:
        action = np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])
    return action


def expected_return(env, Q, gamma, episodes = 10):
    G = np.zeros(episodes)
    for e in range(episodes):
        np.random.seed(e)
        s, _ = env.reset(seed = int(seed), options = options)
        done = False
        t = 0
        while not done:
            a = eps_greedy_action(Q, s, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()


def Q_learning(env, Q, gamma, eps, alpha, max_steps, _seed):
    exp_ret = []
    steps_per_episode = []
    G = np.zeros(num_episodes)
    eps_decay = eps / max_steps
    alpha_decay = alpha / max_steps
    tot_steps = 0
    episodes = 0

    while episodes < num_episodes:
        s, _ = env.reset(seed = _seed, options = options)
        a = eps_greedy_action(Q, s, eps)
        done = False
        episode_steps = 0

        while not done:# and tot_steps < max_steps:
            tot_steps += 1
            episode_steps += 1
            a = eps_greedy_action(Q, s, eps)
            s_next, r, terminated, truncated, _ = env.step(a)

            done = terminated or truncated
            G[episodes] += gamma ** episode_steps * r
            eps = max(eps - eps_decay, 0.01)
            alpha = max(alpha - alpha_decay, 0.001)

            best_actions = np.where(Q[s_next] == np.max(Q[s_next]))[0]
            a_next = np.random.choice(best_actions)
            td_err = r + gamma * np.max(Q[s_next]) * (1 - terminated) - Q[s, a]

            Q[s,a] += alpha * td_err

            if done:
                exp_ret.append(G[episodes])
                steps_per_episode.append(episode_steps)
                episodes += 1
                if adpative_gamma:
                    gamma = min(gamma + episodes * gamma / max_steps, 0.99)

            s = s_next
            a = a_next

    return Q, exp_ret, steps_per_episode


def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span:])
    return re


def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    # if smoothing_window > 1:
    #     y = smooth(y, smoothing_window)

    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    # if smoothing_window > 1:
    #     error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())


alpha = 0.1
eps = 1.0
max_steps = 10000
num_episodes = 500

init_values = [0.0, 5.0, 10.0, 15.0]
gamma_values = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99]
seeds = np.arange(30)

adpative_gamma = True
if adpative_gamma:
    gamma_values = [0.1]

results_exp_ret = np.zeros((
    len(gamma_values),
    len(init_values),
    len(seeds),
    num_episodes,
))

results_steps = np.zeros((
    len(gamma_values),
    len(init_values),
    len(seeds),
    num_episodes,
))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.ion()
plt.show()

for ax in axs:
    ax.set_prop_cycle(color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"])
    ax.set_xlabel("Episodes", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.minorticks_on()

env = gym.make(
    'SimpleGrid-v0', 
    obstacle_map=obstacle_map,
)

for i, gamma in enumerate(gamma_values):
    for j, init_value in enumerate(init_values):
        for seed in seeds:
            np.random.seed(seed)
            Q = np.zeros((n_states, n_actions)) + init_value
            Q, exp_ret, steps = Q_learning(env, Q, gamma, eps, alpha, max_steps, int(seed))
            
            results_exp_ret[i, j, seed] = exp_ret
            results_steps[i, j, seed] = steps

            print(gamma, init_value, seed)

        error_shade_plot(
            axs[0],
            results_exp_ret[i, j],
            stepsize=1,
            smoothing_window=20,
            label=f'γ={gamma:.2f}, $Q_o$={init_value:.2f}'
        )
        axs[0].set_ylabel("Average Return", fontsize=10)
        axs[0].set_title("Q-Learning Performance Across Different Gamma Values")
        axs[0].legend()
        # axs[0].set_ylim([-5,1.4])

        error_shade_plot(
            axs[1],
            results_steps[i, j],
            stepsize=1,
            smoothing_window=20,
            label=f'γ={gamma:.2f},  $Q_o$={init_value:.2f}'
        )
        axs[1].set_ylabel("Steps to Goal", fontsize=10)
        axs[1].set_title("Steps per Episode Across Different Gamma Values")
        axs[1].legend()
        # axs[1].set_ylim([0, 200]) 

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

plt.savefig("adaptive_q.png", dpi=300)
plt.ioff()
plt.show()
