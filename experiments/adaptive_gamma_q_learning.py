import gymnasium as gym
from gym_simplegrid.envs import SimpleGridEnv
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

obstacle_map = [
    "00000100000",
    "00000100000",
    "00000100000",
    "00000000000",
    "00000100000",
    "11011111101",
    "00000100000",
    "00000100000",
    "00000100000",
    "00000000000",
    "00000100000",
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

env_eval = gym.make(
    'SimpleGrid-v0', 
    obstacle_map=obstacle_map,
)

obs, _ = env.reset(seed=1, options = options)
rew = env.unwrapped.reward
done = env.unwrapped.done

n_states = env.observation_space.n
n_actions = env.action_space.n


def eps_greedy_action(Q, s, eps):
    if np.random.rand() < eps:
        action = np.random.choice(n_actions)
    else:
        action = np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])
    return action


def expected_return(env, Q, gamma, episodes=1):
    G = np.zeros(episodes)
    episode_steps = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed = int(seed), options = options)
        done = False
        t = 0
        while not done:
            a = eps_greedy_action(Q, s, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += r
            s = s_next
            t += 1
            if done:
                episode_steps[e] = t
    return G.mean(), episode_steps.mean()


def Q_learning(env, Q, gamma, eps, alpha, max_steps, _seed):
    exp_ret = []
    steps_per_episode = []
    eps_decay = eps / max_steps
    alpha_decay = alpha / max_steps
    tot_steps = 0
    episodes = 0

    while tot_steps < max_steps:  
        s, _ = env.reset(seed = _seed, options = options)
        done = False

        while not done and tot_steps < max_steps:
            tot_steps += 1
            a = eps_greedy_action(Q, s, eps)
            s_next, r, terminated, truncated, _ = env.step(a)

            done = terminated or truncated
            eps = max(eps - eps_decay, 0.01)
            alpha = max(alpha - alpha_decay, 0.001)

            td_err = r + gamma * np.max(Q[s_next]) * (1 - terminated) - Q[s, a]

            Q[s,a] += alpha * td_err

            if tot_steps % eval_steps == 0:
                G, episode_steps = expected_return(env_eval, Q, gamma)
                exp_ret.append(G)
                steps_per_episode.append(episode_steps)

            if done:
                episodes += 1
                if adaptive_gamma:
                    gamma = min(gamma + episodes * gamma / max_steps, 1.0) # check the cutoff parameter

            s = s_next

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
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)

    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())


alpha = 0.5
eps = 1.0
max_steps = 100000
eval_steps = 100

init_values = [0.0, 10.0]
gamma_values = [0.1, 0.5, 0.75, 0.9, 0.99]
seeds = np.arange(30)

adaptive_gamma = True
if adaptive_gamma:
    gamma_values = [0.1]

results_exp_ret = np.zeros((
    len(gamma_values),
    len(init_values),
    len(seeds),
    max_steps // eval_steps,
))

results_steps= np.zeros((
    len(gamma_values),
    len(init_values),
    len(seeds),
    max_steps // eval_steps,
))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.ion()
plt.show()

for ax in axs:
    ax.set_prop_cycle(color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"])
    ax.set_xlabel("Steps", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.minorticks_on()


for i, gamma in enumerate(gamma_values):
    for j, init_value in enumerate(init_values):
        for seed in seeds:
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
        axs[0].set_title("Q-Learning Performance with adaptive gamma")
        axs[0].legend()
        axs[0].set_ylim([-5,1.4])

        error_shade_plot(
            axs[1],
            results_steps[i, j],
            stepsize=1,
            smoothing_window=20,
            label=f'γ={gamma:.2f}, $Q_o$={init_value:.2f}'
        )
        axs[1].set_ylabel("Steps to Goal", fontsize=10)
        axs[1].set_title("Steps per Episode with adaptive gamma")
        axs[1].legend()
        # axs[1].set_ylim([0, 200]) 

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

plt.savefig("adaptive_gamma_q.png", dpi=300)
plt.ioff()
plt.show()