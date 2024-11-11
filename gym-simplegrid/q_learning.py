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
    render_mode='rgb_array'
)

obs, info = env.reset(seed=1, options=options)
rew = env.unwrapped.reward
done = env.unwrapped.done

n_states = (length + 1) * (width + 1)
n_actions = 4

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, done, _, info = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = done

# next state probability for terminal transitions is 0
P = P * (1.0 - T[..., None])


def bellman_q(pi, gamma, max_iter=1000):
    delta = np.inf
    iter = 0
    Q = np.zeros((n_states, n_actions))
    be = np.zeros((max_iter))
    while delta > 1e-5 and iter < max_iter:
        Q_new = R + (np.dot(P, gamma * (Q * pi)).sum(-1))
        delta = np.abs(Q_new - Q).sum()
        be[iter] = delta
        Q = Q_new
        iter += 1
    return Q


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


def expected_return(env, Q, gamma, episodes=10):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
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
    be = []
    exp_ret = []
    tde = []
    eps_decay = eps / max_steps
    alpha_decay = alpha / max_steps
    tot_steps = 0
    while tot_steps < max_steps:
        s, info = env.reset(seed = _seed)
        a = eps_greedy_action(Q, s, eps)
        done = False
        while not done and tot_steps < max_steps:
            tot_steps += 1
            a = eps_greedy_action(Q, s, eps)
            s_next, r, terminated, truncated, info = env.step(a)

            done = terminated or truncated
            eps = max(eps - eps_decay, 0.01)
            alpha = max(alpha - alpha_decay, 0.001)

            best_actions = np.where(Q[s_next] == np.max(Q[s_next]))[0]
            a_next = np.random.choice(best_actions)
            td_err = r + gamma * np.max(Q[s_next]) * (1 - terminated) - Q[s, a]

            Q[s,a] += alpha * td_err

            tde.append(abs(td_err))

            if tot_steps % 100 == 0:
                Q_true = bellman_q(eps_greedy_probs(Q, 0), gamma)
                be.append(np.mean(np.abs(Q - Q_true)))
                exp_ret.append(expected_return(env, Q, gamma))  # update it to env_eval?

            s = s_next
            a = a_next

    return Q, be, tde, exp_ret


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


gamma = 0.99
alpha = 0.1
eps = 1.0
max_steps = 10000
horizon = 10

init_values = [-10, 0.0, 10]
seeds = np.arange(5)

results_be = np.zeros((
    len(init_values),
    len(seeds),
    max_steps // 100,
))
results_tde = np.zeros((
    len(init_values),
    len(seeds),
    max_steps,
))
results_exp_ret = np.zeros((
    len(init_values),
    len(seeds),
    max_steps // 100,
))

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plt.ion()
plt.show()

reward_noise_std = 0.0  # re-run with 3.0

for ax in axs:
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black",
               "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.minorticks_on()

env = gym.make(
    'SimpleGrid-v0', 
    obstacle_map=obstacle_map, 
    render_mode='rgb_array'
)

for i, init_value in enumerate(init_values):
    for seed in seeds:
        np.random.seed(seed)
        Q = np.zeros((n_states, n_actions)) + init_value
        Q, be, tde, exp_ret = Q_learning(
            env, Q, gamma, eps, alpha, max_steps, int(seed))
        results_be[i, seed] = be
        results_tde[i, seed] = tde
        results_exp_ret[i, seed] = exp_ret
        print(i, seed)
    label = f"$Q_0$: {init_value}"
    axs[0].set_title("TD Error", fontsize=12)
    error_shade_plot(
        axs[0],
        results_tde[i],
        stepsize=1,
        smoothing_window=20,
        label=label,
    )
    axs[0].legend()
    axs[0].set_ylim([0, 5])
    axs[1].set_title("Bellman Error", fontsize=12)
    error_shade_plot(
        axs[1],
        results_be[i],
        stepsize=100,
        smoothing_window=20,
        label=label,
    )
    axs[1].legend()
    axs[1].set_ylim([0, 50])
    axs[2].set_title("Expected Return", fontsize=12)
    error_shade_plot(
        axs[2],
        results_exp_ret[i],
        stepsize=100,
        smoothing_window=20,
        label=label,
    )
    axs[2].legend()
    axs[2].set_ylim([-5, 1])
    plt.tight_layout() 
    plt.draw()
    plt.pause(0.001)

plt.savefig("q.png", dpi=300)
plt.ioff()
plt.show()
